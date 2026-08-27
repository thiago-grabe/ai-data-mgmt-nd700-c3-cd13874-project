from bson import ObjectId
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from presidio_analyzer import AnalyzerEngine
from pymongo import MongoClient

# Presidio's AnalyzerEngine loads the spaCy en_core_web_lg pipeline on construction
# — several seconds and a few hundred MB. The shipped code built one per call,
# which put that cost on every single query. One engine, built on first use.
_ANALYZER = None


def _get_analyzer() -> AnalyzerEngine:
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    return _ANALYZER


# Presidio confidence floor. Entities below this are treated as too speculative to
# act on; URL is excluded because permit documents cite public council URLs that
# are not personal data.
PII_CONFIDENCE_THRESHOLD = 0.85


class UnstructuredDataAgent:
    """
    MongoDB + Chroma + Azure OpenAI RAG pipeline

    Connects to exactly one data source: the MongoDB permit-document collection.
    Documents are embedded into a local Chroma index for retrieval, but the answer
    is always grounded in the full document fetched back from MongoDB by ObjectId —
    the source of truth never leaves its own store.
    """

    # ============================================
    # INIT
    # ============================================
    def __init__(
        self,
        mongo_uri: str,
        db_name: str,
        collection_name: str,
        chroma_path: str,
        azure_endpoint: str,
        azure_key: str,
        azure_api_version: str,
        embedding_deployment: str,
        chat_deployment: str,
        collection_label: str = "rag_collection",
        temperature: float = 0.0,
    ):
        # Mongo
        self.mongo_client = MongoClient(mongo_uri)
        self.mongo_db = self.mongo_client[db_name]
        self.mongo_coll = self.mongo_db[collection_name]

        self.chroma_path = chroma_path
        self.collection_label = collection_label

        # Embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            azure_deployment=embedding_deployment,
            openai_api_version=azure_api_version,
            api_key=azure_key,
        )

        # Vector DB
        self.vector_db = Chroma(
            collection_name=collection_label,
            embedding_function=self.embeddings,
            persist_directory=chroma_path,
        )

        # Chat model
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            azure_deployment=chat_deployment,
            api_version=azure_api_version,
            temperature=temperature,
        )

        # Populated by the most recent __contains_pii run so ask() can redact
        # rather than merely warn.
        self.last_pii_findings = []

    # ============================================
    # BUILD VECTOR INDEX
    # ============================================
    def build_index(self):
        print("Loading documents from MongoDB...")

        docs = list(self.mongo_coll.find({}, {"_id": 1, "content": 1}))

        texts = []
        ids = []
        metadatas = []

        for d in docs:
            doc_id = str(d["_id"])
            text = d.get("content", "").strip()

            if not text:
                continue

            texts.append(text)
            ids.append(doc_id)
            metadatas.append({"mongo_id": doc_id})

        if not texts:
            print("No documents found to index.")
            return

        print(f"Embedding {len(texts)} documents...")

        # Explicit ids make this an upsert keyed on the Mongo ObjectId, so
        # re-running refreshes vectors instead of duplicating them.
        self.vector_db.add_texts(
            texts=texts,
            ids=ids,
            metadatas=metadatas,
        )

        print("Vector index built.")
        print("Vector DB size:", self.vector_db._collection.count())

    # ============================================
    # RAG QUERY
    # ============================================
    def ask(self, question: str, k: int = 3, run_pii_audit=True):
        results = self.vector_db.similarity_search(question, k=k)

        context_chunks = []
        source_docs = []

        for doc in results:
            mongo_id = doc.metadata.get("mongo_id")

            try:
                # Chroma stores only the id; the authoritative document is read
                # back from MongoDB so the answer is grounded in the source store.
                mongo_doc = self.mongo_coll.find_one({"_id": ObjectId(mongo_id)})
            except Exception:
                mongo_doc = None

            if mongo_doc and mongo_doc.get("content"):
                content = mongo_doc["content"]
                context_chunks.append(content)

                # store full document metadata for transparency
                source_docs.append({
                    "mongo_id": str(mongo_doc["_id"]),
                    "filename": mongo_doc.get("filename"),
                    "content": content
                })

        self.last_pii_findings = []

        if not context_chunks:
            message = "No relevant documents found."
            # "response" added alongside the shipped keys: chat.py reads
            # result.get("response", str(result)), so without it this branch dumps
            # a raw dict at the user instead of the sentence.
            return {
                "response": message,
                "answer": message,
                "sources": []
            }

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
Use the following documents as context:

{context_text}

Question: {question}

Answer clearly and concisely:
"""

        response = self.llm.invoke(prompt)
        answer = response.content

        if run_pii_audit:
            print("\n" + "=" * 80)
            print("Using Unstrucutred Data stored in MongoDB database to answer question")
            print("Seraching for PII data in source documents")
            print("=" * 80)
            self.__contains_pii(str(source_docs))

            # Actionable safeguard: detecting PII and then printing it back to the
            # user in the answer would defeat the point of detecting it.
            answer = self.__redact(answer, self.last_pii_findings)

        return {
            "response": answer,
            "answer": answer,
            "sources": [
                {"mongo_id": s["mongo_id"], "filename": s["filename"]}
                for s in source_docs
            ],
            "pii_findings": self.last_pii_findings,
        }

    # ============================================
    # OPTIONAL — RESET VECTOR DB
    # ============================================
    def reset_index(self):
        print("Clearing vector database...")
        # Delete by id rather than delete_collection(): the latter would leave
        # self.vector_db bound to a destroyed collection, so the next add_texts
        # would fail on an object that still looks alive.
        existing = self.vector_db.get(include=[])
        ids = existing.get("ids", [])
        if ids:
            self.vector_db.delete(ids=ids)
        print("Vector DB cleared.")

    # ============================================
    # PRIVATE METHODS
    # ============================================

    def __redact(self, text: str, findings) -> str:
        """Replace PII in an outgoing answer with type placeholders.

        Detection runs on the ANSWER, not only on the source documents. Detecting
        on the source and then string-matching into the answer looks equivalent but
        is not: Presidio's spans over PDF-extracted text routinely bleed into
        neighbouring layout, e.g. it returns

            PERSON: 'Lydia R. Montrose\\n• Company/Developer'

        so the exact-match replace never fires against the clean "Lydia R.
        Montrose" the model wrote. In an earlier run that leaked every permit
        owner's name while dutifully redacting the dates around it. Redact what
        you are about to emit, using spans found in that same text.
        """
        if not text:
            return text

        analyzer = _get_analyzer()
        results = [
            r for r in analyzer.analyze(text=text, language="en")
            if r.entity_type != "URL" and r.score >= PII_CONFIDENCE_THRESHOLD
        ]

        # Replace from the end so earlier offsets stay valid, and drop spans nested
        # inside one already replaced.
        last_start = len(text)
        for r in sorted(results, key=lambda r: (r.start, -r.end), reverse=True):
            if r.end > last_start:
                continue
            text = text[:r.start] + f"[REDACTED {r.entity_type}]" + text[r.end:]
            last_start = r.start

        # Second pass: any value detected in the source documents that survived
        # verbatim into the answer. Longest first so a full name is masked before
        # a substring of it.
        for finding in sorted(findings or [], key=lambda f: len(f["text"]), reverse=True):
            value = finding["text"].split("\n")[0].strip()
            if len(value) > 3 and value in text:
                text = text.replace(value, f"[REDACTED {finding['entity_type']}]")

        return text

    def __contains_pii(self, text: str) -> bool:
        analyzer = _get_analyzer()

        results = analyzer.analyze(
            text=text,
            language="en"
        )

        if results:
            seen = set()  # track unique PII
            print("\nWARNING: PII detected:")
            for r in results:
                # skip URLs
                if r.entity_type == "URL" or r.score < PII_CONFIDENCE_THRESHOLD:
                    continue
                pii_text = text[r.start:r.end]
                if pii_text not in seen:
                    seen.add(pii_text)
                    self.last_pii_findings.append({
                        "entity_type": r.entity_type,
                        "text": pii_text,
                        "score": float(r.score),
                    })
                    print(f"- {r.entity_type}: '{pii_text}' (confidence={r.score:.2f})")
            print("\n")
            return bool(self.last_pii_findings)

        print("\n No PII detected.\n")
        return False
