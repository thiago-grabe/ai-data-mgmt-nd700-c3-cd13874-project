"""LLM-based query routing, with the keyword router as a deterministic fallback.

Keyword routing is brittle in ways this project's own example queries expose:
"Show me houses similar to my house" matches no multimodal keyword (the list has
"similar house"; the sentence reads "houses similar to") and falls through the
default straight to the wrong agent. Substring matching also misfires the other
way — "text" matches inside "context", "count" inside "account".

An LLM classifier reads intent rather than substrings. But it can time out, rate
limit, or hallucinate a label, so it is never the only mechanism: any failure
falls back to the keyword router, and an unrecognized label is discarded.
"""

VALID_ROUTES = ("structured", "unstructured", "multimodal", "timeseries")

SYSTEM_PROMPT = """You are a query router for a neighborhood data assistant.
Classify the user's question into exactly one category and reply with that single
word, lowercase, nothing else.

structured   - demographics, housing, income, schools, hospitals; counts, averages,
               comparisons. Backed by a PostgreSQL database.
unstructured - construction and building permits, approvals, inspection notes,
               permit documents. Backed by PDF documents in MongoDB.
multimodal   - anything about what a house looks like: photos, images, visual
               similarity, "a house like mine". Backed by images in Blob Storage.
timeseries   - environmental sensor readings over time: air quality, PM2.5, noise
               levels, water usage, unusual or anomalous readings, trends.

Reply with exactly one of: structured, unstructured, multimodal, timeseries"""


class LLMRouter:
    def __init__(self, llm, keyword_router):
        """
        llm            - an AzureChatOpenAI (or anything with .invoke returning .content)
        keyword_router - callable(str) -> str, the deterministic fallback
        """
        self.llm = llm
        self.keyword_router = keyword_router
        self.last_method = "keyword"

    def route(self, user_message: str) -> str:
        try:
            response = self.llm.invoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", user_message),
                ]
            )
            label = (response.content or "").strip().lower().split()[0].strip(".,")

            if label in VALID_ROUTES:
                self.last_method = "llm"
                return label

            print(f"[Router] LLM returned unrecognized label {label!r}; using keywords.")
        except Exception as e:
            print(f"[Router] LLM routing unavailable ({type(e).__name__}); using keywords.")

        self.last_method = "keyword"
        return self.keyword_router(user_message)
