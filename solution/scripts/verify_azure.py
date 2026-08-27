"""Phase 0 exit gate: prove every Azure dependency is live before writing code against it.

Runs the same SDK calls, with the same arguments, that the agents use — so a pass
here means the agents will connect, not that the resources merely exist.

    python scripts/verify_azure.py

Exits non-zero if any probe fails.
"""

import os
import sys
import traceback
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOLUTION_DIR))

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DeviceCodeCredential,
)
from azure.keyvault.secrets import SecretClient

VAULT = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")

SECRET_NAMES = [
    "structuredazureendpoint",
    "structuredazureapikey",
    "structuredpostgresqlpassword",
    "structuredpostgresqluser",
    "structuredpostgresqldbname",
    "structuredpostgresqlhost",
    "unstructuredmongourl",
    "unstructureddbname",
    "unstructuredcollectionname",
    "unstructuredazureendpoint",
    "unstructuredazurekey",
    "multimodalazureconnstring",
    "multimodalazurecontentsafetyendpoint",
    "multimodalazurecontentsafetykey",
    "timeseriesazureconnstring",
    "timeseriestablename",
]

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"

results = []


def probe(label):
    """Decorator turning a function into a recorded pass/fail probe."""
    def wrap(fn):
        print(f"\n--- {label} ---")
        try:
            detail = fn()
            print(f"{GREEN}PASS{RESET}  {label}: {detail}")
            results.append((label, True, detail))
        except Exception as e:
            print(f"{RED}FAIL{RESET}  {label}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
            results.append((label, False, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


print(f"Verifying Azure resources via Key Vault '{VAULT}'")
credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())
kv = SecretClient(vault_url=f"https://{VAULT}.vault.azure.net/", credential=credential)

S = {}


@probe("A3 Key Vault — all 16 secrets readable")
def _a3():
    missing = []
    for name in SECRET_NAMES:
        try:
            S[name] = kv.get_secret(name).value
        except Exception:
            missing.append(name)
    if missing:
        raise AssertionError(f"unreadable secrets: {missing}")
    return f"{len(SECRET_NAMES)} secrets"


if len(S) != len(SECRET_NAMES):
    print(f"\n{RED}Cannot continue without all secrets.{RESET}")
    sys.exit(1)


@probe("A1 Azure OpenAI chat — gpt-4.1-mini")
def _a1():
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_endpoint=S["structuredazureendpoint"],
        api_key=S["structuredazureapikey"],
        azure_deployment="gpt-4.1-mini",
        api_version="2024-12-01-preview",
        temperature=0,
    )
    reply = llm.invoke("Reply with the single word: pong")
    return f"responded {reply.content.strip()[:20]!r}"


@probe("A2 Azure OpenAI embeddings — text-embedding-ada-002 @ 2023-06-01-preview")
def _a2():
    from langchain_openai import AzureOpenAIEmbeddings

    # Deliberately the exact api_version chat.py hardcodes for the unstructured
    # agent. If this version has been retired, the RAG pipeline is dead and we
    # need to know now, not during the evaluation run.
    emb = AzureOpenAIEmbeddings(
        azure_endpoint=S["unstructuredazureendpoint"],
        azure_deployment="text-embedding-ada-002",
        openai_api_version="2023-06-01-preview",
        api_key=S["unstructuredazurekey"],
    )
    vector = emb.embed_query("ping")
    assert len(vector) == 1536, f"unexpected embedding dimension {len(vector)}"
    return f"dimension {len(vector)}"


@probe("A4 PostgreSQL — 4 tables with expected row counts")
def _a4():
    import psycopg

    expected = {
        "neighborhood_houses": 5540,
        "education_survey": 2580,
        "neighborhood_schools": 10,
        "neighborhood_hospitals": 6,
    }
    conn_args = {
        "host": S["structuredpostgresqlhost"],
        "dbname": S["structuredpostgresqldbname"],
        "user": S["structuredpostgresqluser"],
        "password": S["structuredpostgresqlpassword"],
        "port": 5432,
        "sslmode": "require",
    }
    found = {}
    with psycopg.connect(**conn_args) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        for table in expected:
            cur.execute(f"SELECT count(*) FROM {table}")
            found[table] = cur.fetchone()[0]

    wrong = {t: (found[t], expected[t]) for t in expected if found[t] != expected[t]}
    if wrong:
        # A count above expected almost always means a loader was run twice, which
        # silently doubles every group in the Fairlearn audit.
        raise AssertionError(f"row count mismatch (found, expected): {wrong}")
    return ", ".join(f"{t}={found[t]}" for t in expected)


@probe("A5 MongoDB — 21 permit documents")
def _a5():
    from pymongo import MongoClient

    client = MongoClient(S["unstructuredmongourl"], serverSelectionTimeoutMS=20000)
    client.admin.command("ping")
    coll = client[S["unstructureddbname"]][S["unstructuredcollectionname"]]
    count = coll.count_documents({})
    assert count == 21, f"expected 21 documents, found {count}"
    return f"{count} documents"


@probe("A6 Blob Storage — 12 house images")
def _a6():
    from azure.storage.blob import BlobServiceClient

    container = BlobServiceClient.from_connection_string(
        S["multimodalazureconnstring"]
    ).get_container_client("houses")
    blobs = [b.name for b in container.list_blobs()]
    assert len(blobs) == 12, f"expected 12 blobs, found {len(blobs)}"
    return f"{len(blobs)} images"


@probe("A7 Content Safety — analyze_image on the largest image")
def _a7():
    from azure.ai.contentsafety import ContentSafetyClient
    from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
    from azure.core.credentials import AzureKeyCredential

    sys.path.insert(0, str(SOLUTION_DIR))
    from io import BytesIO

    from PIL import Image

    from agents.multimodal_data_agent import CONTENT_SAFETY_MAX_DIM

    # 274_Maplewood_Crescent.jpg is 2268x1676, past the 2048px service limit.
    # Probing with this specific file proves the downscale path works rather than
    # that a small image happens to pass.
    path = SOLUTION_DIR / "data" / "multimodal" / "274_Maplewood_Crescent.jpg"
    raw = path.read_bytes()
    image = Image.open(BytesIO(raw))
    if max(image.size) > CONTENT_SAFETY_MAX_DIM:
        resized = image.convert("RGB")
        resized.thumbnail((CONTENT_SAFETY_MAX_DIM, CONTENT_SAFETY_MAX_DIM), Image.LANCZOS)
        buf = BytesIO()
        resized.save(buf, format="JPEG", quality=92)
        raw = buf.getvalue()

    client = ContentSafetyClient(
        endpoint=S["multimodalazurecontentsafetyendpoint"],
        credential=AzureKeyCredential(S["multimodalazurecontentsafetykey"]),
    )
    response = client.analyze_image(AnalyzeImageOptions(image=ImageData(content=raw)))
    cats = {str(c.category): c.severity for c in response.categories_analysis}
    assert len(cats) >= 4, f"expected 4 categories, got {cats}"
    return ", ".join(f"{k}={v}" for k, v in cats.items())


@probe("A8 Table Storage — sensor readings present")
def _a8():
    from azure.data.tables import TableServiceClient

    table = TableServiceClient.from_connection_string(
        S["timeseriesazureconnstring"]
    ).get_table_client(S["timeseriestablename"])
    rows = list(table.list_entities())
    assert len(rows) == 1680, f"expected 1680 readings, found {len(rows)}"
    partitions = {r["PartitionKey"] for r in rows}
    assert len(partitions) == 5, f"expected 5 neighborhoods, found {sorted(partitions)}"
    return f"{len(rows)} readings across {len(partitions)} neighborhoods"


@probe("A9 Source isolation — each credential reaches only its own store")
def _a9():
    # The rubric requires each agent to connect exclusively to its own data source.
    # These are distinct storage accounts and distinct services, so a credential
    # crossing over should be impossible; assert it rather than assume it.
    from azure.storage.blob import BlobServiceClient

    def account_of(conn_str):
        return dict(
            part.split("=", 1) for part in conn_str.split(";") if "=" in part
        )["AccountName"]

    images_account = account_of(S["multimodalazureconnstring"])
    sensors_account = account_of(S["timeseriesazureconnstring"])
    assert images_account != sensors_account, (
        f"multimodal and time-series share storage account {images_account}; "
        "per-source credential isolation is not real"
    )

    # The sensor credential must not be able to read the house images.
    try:
        container = BlobServiceClient.from_connection_string(
            S["timeseriesazureconnstring"]
        ).get_container_client("houses")
        list(container.list_blobs())
        raise AssertionError(
            "the time-series storage credential can read the multimodal container"
        )
    except AssertionError:
        raise
    except Exception:
        pass  # expected: the container does not exist under that account

    assert S["structuredpostgresqlhost"] not in S["unstructuredmongourl"]
    return f"images={images_account}, sensors={sensors_account}, distinct"


# ------------------------------------------------------------------- summary
print("\n" + "=" * 72)
print("PHASE 0 VERIFICATION SUMMARY")
print("=" * 72)
for label, passed, detail in results:
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"{mark}  {label}")
    if not passed:
        print(f"        {detail}")

failed = [r for r in results if not r[1]]
print("=" * 72)
if failed:
    print(f"{RED}{len(failed)} of {len(results)} probes failed.{RESET}")
    sys.exit(1)
print(f"{GREEN}All {len(results)} probes passed. Azure is ready.{RESET}")
