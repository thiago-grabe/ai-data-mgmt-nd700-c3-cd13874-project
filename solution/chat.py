import warnings

warnings.filterwarnings("ignore", message="Field.*conflict with protected namespace")

import os

# Must run before anything opens an HTTPS connection — notably the Hugging Face
# download of the CLIP weights. See tls_trust.py for why.
import tls_trust

tls_trust.enable()

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DeviceCodeCredential,
)
from azure.keyvault.secrets import SecretClient

from agents.multimodal_data_agent import MultimodalDataAgent
from agents.structured_data_agent import StructuredDataAgent
from agents.timeseries_data_agent import TimeSeriesDataAgent
from agents.unstructured_data_agent import UnstructuredDataAgent
from audit import (
    AuditLog,
    summarize_anomaly,
    summarize_bias,
    summarize_content_safety,
    summarize_pii,
)
from router_llm import LLMRouter

# ============================================================
#  Secrets
# ============================================================

# The vault NAME is a resource identifier, not a secret: it grants nothing on its
# own, and every read below is gated by Entra ID. Overridable by environment so
# the same code runs against a differently-named vault without an edit.
keyVaultName = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
KVUri = f"https://{keyVaultName}.vault.azure.net/"

print("Connecting to Azure for authentication.")

# DeviceCodeCredential is the authentication path this project documents, and it
# is what an interactive user gets. AzureCliCredential is tried first so an
# existing `az login` session is reused: without it, any non-interactive run (the
# evaluation harness, CI) blocks forever on a device-code prompt nobody can answer.
credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())
client = SecretClient(vault_url=KVUri, credential=credential)

# Every credential in this system is fetched from Key Vault at runtime. Nothing
# below is stored on disk, in source, or in an environment variable. Each agent
# receives only the secrets for its own data source, so no agent is technically
# capable of reaching another agent's store.
#
#   Structured   -> PostgreSQL          + Azure OpenAI (key1)
#   Unstructured -> MongoDB             + Azure OpenAI (key2)
#   Multimodal   -> Blob Storage        + Content Safety
#   Time-series  -> Table Storage       + Azure OpenAI (key1)

# ============================================================
# Structured Data Agent Secrets
# ============================================================

structuredazureendpoint = retrieved_secret = client.get_secret("structuredazureendpoint").value
structuredazureapikey = retrieved_secret = client.get_secret("structuredazureapikey").value
structuredpostgresqlpassword = retrieved_secret = client.get_secret("structuredpostgresqlpassword").value
structuredpostgresqluser = retrieved_secret = client.get_secret("structuredpostgresqluser").value
structuredpostgresqldbname = retrieved_secret = client.get_secret("structuredpostgresqldbname").value
structuredpostgresqlhost = retrieved_secret = client.get_secret("structuredpostgresqlhost").value

unstructuredmongourl = retrieved_secret = client.get_secret("unstructuredmongourl").value
unstructureddbname = retrieved_secret = client.get_secret("unstructureddbname").value
unstructuredcollectionname = retrieved_secret = client.get_secret("unstructuredcollectionname").value
unstructuredazureendpoint = retrieved_secret = client.get_secret("unstructuredazureendpoint").value
unstructuredazurekey = retrieved_secret = client.get_secret("unstructuredazurekey").value

multimodalazureconnstring = retrieved_secret = client.get_secret("multimodalazureconnstring").value
multimodalazurecontentsafetyendpoint = retrieved_secret = client.get_secret("multimodalazurecontentsafetyendpoint").value
multimodalazurecontentsafetykey = retrieved_secret = client.get_secret("multimodalazurecontentsafetykey").value

timeseriesazureconnstring = retrieved_secret = client.get_secret("timeseriesazureconnstring").value
timeseriestablename = retrieved_secret = client.get_secret("timeseriestablename").value


class AgentChatManager:
    def __init__(self, use_llm_router=True, figure_dir=None):
        self.audit = AuditLog()
        self.figure_dir = figure_dir
        self._figure_count = 0
        self._load_agents()

        # LLM routing with the keyword router as deterministic fallback.
        self.llm_router = (
            LLMRouter(self.structured.model, self._route_query_keywords)
            if use_llm_router
            else None
        )

    # -------------------------
    #  Agents
    # -------------------------
    def _load_agents(self):

        print("\nLoading Structured Data Agent")
        print("-------------------------------")
        self.structured = StructuredDataAgent(
            azure_endpoint=structuredazureendpoint,
            api_key=structuredazureapikey,
            deployment="gpt-4.1-mini",
            db_config={
                "host": structuredpostgresqlhost,
                "dbname": structuredpostgresqldbname,
                "user": structuredpostgresqluser,
                "password": structuredpostgresqlpassword,
                "port": 5432,
            }
        )

        print("\nLoading Unstructured Data Agent")
        print("---------------------------------")
        self.unstructured = UnstructuredDataAgent(
            mongo_uri=unstructuredmongourl,
            db_name=unstructureddbname,
            collection_name=unstructuredcollectionname,
            chroma_path="./data/chroma_db_storage",
            azure_endpoint=unstructuredazureendpoint,
            azure_key=unstructuredazurekey,
            azure_api_version="2023-06-01-preview",
            embedding_deployment="text-embedding-ada-002",
            chat_deployment="gpt-4.1-mini",
        )
        self.unstructured.build_index()

        print("\nLoading Multimodal Data Agent")
        print("-------------------------------")
        self.multimodal = MultimodalDataAgent(
            azure_conn_str=multimodalazureconnstring,
            container_name="houses",
            content_safety_endpoint=multimodalazurecontentsafetyendpoint,
            content_safety_key=multimodalazurecontentsafetykey
        )

        print("\nLoading Time-Series Data Agent")
        print("-------------------------------")
        self.timeseries = TimeSeriesDataAgent(
            connection_string=timeseriesazureconnstring,
            table_name=timeseriestablename,
            azure_endpoint=structuredazureendpoint,
            api_key=structuredazureapikey,
            deployment="gpt-4.1-mini",
        )

    # -------------------------
    #  Routing
    # -------------------------
    def _route_query(self, user_message: str) -> str:
        """Route a query to one of: structured, unstructured, multimodal, timeseries.

        Uses the LLM classifier when available, falling back to keyword matching.
        """
        if self.llm_router is not None:
            return self.llm_router.route(user_message)
        return self._route_query_keywords(user_message)

    def _route_query_keywords(self, user_message: str) -> str:
        """
        Route user query to one of:
        - structured
        - unstructured
        - multimodal
        - timeseries

        This is a rule-based router. You can extend it later.
        """
        msg = user_message.lower().strip()

        multimodal_keywords = [
            "house like mine",
            "find me a house like mine",
            "show me a house like mine",
            "similar house",
            "similar image",
            "image",
            "photo",
            "picture",
            "looks like",
            "visual",
            # Added: the project's own example query "Show me houses similar to my
            # house" matched none of the above ("similar house" does not occur in
            # "houses similar to") and fell through to the default, reaching the
            # unstructured agent instead of this one.
            "houses similar",
            "similar to my house",
            "similar to mine",
            "like my house",
        ]

        timeseries_keywords = [
            "sensor",
            "air quality",
            "pm2.5",
            "pm25",
            "particulate",
            "noise level",
            "decibel",
            "water usage",
            "reading",
            "readings",
            "anomaly",
            "anomalies",
            "unusual",
            "spike",
            "over time",
            "trend",
        ]

        structured_keywords = [
            "demographic",
            "demographics",
            "price",
            "home price",
            "most expensive",
            "average",
            "median",
            "count",
            "how many",
            "highest",
            "lowest",
            "neighborhood",
            "income",
            "population",
            "sql",
            "database",
        ]

        unstructured_keywords = [
            "permit",
            "approved",
            "document",
            "documents",
            "text",
            "restaurant",
            "permit approved",
            "notes",
            "report",
            "permit document",
        ]

        # Priority 1: multimodal
        if any(keyword in msg for keyword in multimodal_keywords):
            return "multimodal"

        # Priority 2: time-series. Checked before structured because sensor
        # questions naturally contain "average" and "neighborhood".
        if any(keyword in msg for keyword in timeseries_keywords):
            return "timeseries"

        # Priority 3: unstructured
        if any(keyword in msg for keyword in unstructured_keywords):
            return "unstructured"

        # Priority 4: structured
        if any(keyword in msg for keyword in structured_keywords):
            return "structured"

        # Default fallback
        return "unstructured"

    # -------------------------
    #  Agent wrappers
    # -------------------------
    def _run_structured(self, prompt: str) -> str:
        result = self.structured.ask(prompt, verbose=False, run_bias_audit=True)

        if isinstance(result, dict):
            self._last_checks = [summarize_bias(result.get("bias_findings"))]
            self._last_modified = bool(result.get("bias_findings"))
            return result.get("response", str(result))

        return str(result)

    def _run_unstructured(self, prompt: str) -> str:
        result = self.unstructured.ask(prompt, run_pii_audit=True)

        if isinstance(result, dict):
            self._last_checks = [summarize_pii(result.get("pii_findings"))]
            self._last_modified = bool(result.get("pii_findings"))
            return result.get("response", str(result))

        return str(result)

    def _run_multimodal(self, prompt: str) -> str:
        # The shipped version ignored the prompt entirely and always used
        # query-house-2.jpg, so two different multimodal questions produced
        # byte-identical answers. The default is unchanged; naming an image now
        # selects it.
        query_image = "query-house-2.jpg"
        for candidate in ("query-house-1.jpg", "query-house-2.jpg"):
            if candidate.replace(".jpg", "") in prompt.lower().replace(" ", "-"):
                query_image = candidate
                break

        matches, query_img = self.multimodal.find_similar(query_image)

        self._last_checks = [summarize_content_safety(
            self.multimodal.last_safety_findings,
            self.multimodal.blocked_blobs,
        )]
        self._last_modified = bool(self.multimodal.blocked_blobs)

        print("\nTop matches:")
        for score, path, address in matches:
            print(f"{address} | similarity: {score:.4f}")

        # In an interactive session this pops a window; the evaluation harness
        # sets figure_dir so the figure is saved as committed evidence instead.
        save_path = None
        if self.figure_dir:
            self._figure_count += 1
            os.makedirs(self.figure_dir, exist_ok=True)
            save_path = os.path.join(
                self.figure_dir, f"multimodal_matches_{self._figure_count}.png"
            )
        self.multimodal.show_results(query_img, matches, save_path=save_path)

        if not matches:
            return "I could not find similar houses."

        lines = [f"Here are the top similar houses I found (query image: {query_image}):"]
        for score, path, address in matches[:5]:
            lines.append(f"- {address} (similarity: {score:.4f})")

        if self.multimodal.blocked_blobs:
            lines.append(
                f"\n{len(self.multimodal.blocked_blobs)} image(s) were excluded by "
                "Azure AI Content Safety."
            )

        return "\n".join(lines)

    def _run_timeseries(self, prompt: str) -> str:
        result = self.timeseries.ask(prompt, run_anomaly_audit=True)

        if isinstance(result, dict):
            self._last_checks = [summarize_anomaly(
                result.get("anomalies"), result.get("refused", False)
            )]
            self._last_modified = bool(result.get("anomalies")) or result.get("refused", False)
            return result.get("response", str(result))

        return str(result)

    # -------------------------
    #  Chat Interface
    # -------------------------
    def chat(self, user_message: str) -> str:
        self._last_checks = []
        self._last_modified = False

        route = self._route_query(user_message)
        method = self.llm_router.last_method if self.llm_router else "keyword"
        print(f"[Router] Selected agent: {route} (via {method})")

        try:
            if route == "structured":
                answer = self._run_structured(user_message)
            elif route == "unstructured":
                answer = self._run_unstructured(user_message)
            elif route == "multimodal":
                answer = self._run_multimodal(user_message)
            elif route == "timeseries":
                answer = self._run_timeseries(user_message)
            else:
                answer = "I could not determine the correct agent."

            self.audit.record(
                query=user_message,
                route=route,
                router_method=method,
                checks=self._last_checks,
                response_modified=self._last_modified,
            )
            return answer

        except Exception as e:
            self.audit.record(
                query=user_message,
                route=route,
                router_method=method,
                checks=self._last_checks,
                response_modified=False,
                error=f"{type(e).__name__}: {e}",
            )
            return f"An error occurred while processing your request with the {route} agent: {e}"


# ============================================================
#  Run Chat
# ============================================================

if __name__ == "__main__":
    banner = r"""
                                   /\
                                  /  \
                                 /____\
                ________________/______\_______________________
               /                                               \
              /_________________________________________________\
             |   ____      ____      ____      ____      ____   |
             |  | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |  |
             |  | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |    | ▇▇ |  |
             |  |____|    |____|    |____|    |____|    |____|  |
             |                                                  |
             |        NEIGHBORHOOD INSIGHTS & DATA ASSISTANT    |
             |__________________________________________________|
    """
    print(banner)
    print("\n This AI-powered assistant provides automatically generated responses. "
          "Please use discretion, as answers may contain inaccuracies or errors.\n")
    print("Data comes from a structured demographics database, permit documents in NoSQL, images stored in Azure, and environmental sensor readings.")
    print("Data remains in its original systems and is not copied into the assistant.")
    print("Embeddings and other AI artifacts are generated at runtime and are not permanently stored.")

    chat = AgentChatManager()

    print("\n\n--------------------------------")
    print("\n Agent Chat Ready.\n")
    print("Here are some examples of questions you can ask:")
    print("  - Which demographic group has the most expensive homes?")
    print("  - Was a permit approved for a restaurant in any of the neighborhoods?")
    print("  - Find me a house like mine")
    print("  - Are there any unusual air quality readings in Maplewood?")
    print("\n Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break

        answer = chat.chat(user_input)
        print(f"Agent: {answer}\n")
