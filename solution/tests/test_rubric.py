"""Rubric conformance suite — one test per rubric line.

The point of this file is that "meets the rubric" is a command you run, not a
claim in a README. Assertions read live constructed objects wherever possible
rather than grepping source text, because source text can say anything.

Every rubric ID here (R1.1 … R6.4) matches a row in docs/RUBRIC.md.

No test in this file makes a network call.
"""

import ast
import re
from pathlib import Path
from unittest import mock

import pytest
from conftest import SECRET_NAMES

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SOLUTION = Path(__file__).resolve().parents[1]
STARTER = SOLUTION.parent / "starter"

# Real post-load column names for all four tables. Used to prove the demographic
# detector picks the two genuine sensitive features and none of the count columns.
REAL_COLUMNS = {
    "neighborhood_houses": [
        "id", "house_number", "neighborhood", "household_income", "house_size_sqf",
        "house_market_value", "family_size", "identified_race", "swimming_pool",
        "ownership",
    ],
    "education_survey": [
        "id", "neighborhood", "education_level", "employment_status", "age",
        "years_of_experience", "annual_income", "gender", "household_size",
    ],
    "neighborhood_schools": [
        "id", "school_name", "neighborhood", "school_level", "number_of_students",
        "male_students", "female_students", "white_students", "black_students",
        "asian_students", "hispanic_students", "other_students", "average_age",
        "number_of_teachers", "student_teacher_ratio",
    ],
    "neighborhood_hospitals": [
        "id", "hospital_name", "neighborhood", "hospital_type", "number_of_beds",
        "average_daily_patients", "emergency_capacity", "doctors", "nurses",
        "male_patients", "female_patients", "white_patients", "black_patients",
        "asian_patients", "hispanic_patients", "other_patients",
        "average_wait_time_min", "icu_beds", "annual_admissions",
    ],
}


def solution_python_files():
    return [
        p for p in SOLUTION.rglob("*.py")
        if ".venv" not in p.parts and "__pycache__" not in p.parts
    ]


# Built at runtime so this file's own prose about the marker does not match it.
TODO_MARKER = "<" + "TODO"


# ===========================================================================
# Section 1 — Architecture Design
# ===========================================================================

def test_R1_1_architecture_doc_covers_orchestrator_agents_and_safeguards():
    doc = SOLUTION / "architecture.md"
    assert doc.exists(), "architecture.md is a named rubric deliverable"
    text = doc.read_text()

    assert "```mermaid" in text, "architecture.md must contain a diagram"

    for orchestrator in ["AgentChatManager"]:
        assert orchestrator in text

    for agent in [
        "StructuredDataAgent",
        "UnstructuredDataAgent",
        "MultimodalDataAgent",
        "TimeSeriesDataAgent",
    ]:
        assert agent in text, f"{agent} missing from architecture.md"

    for source in ["PostgreSQL", "MongoDB", "ChromaDB", "Blob Storage", "Table Storage"]:
        assert source in text, f"data source {source} missing from architecture.md"

    for safeguard in ["Fairlearn", "Presidio", "Content Safety"]:
        assert safeguard in text, f"ethical check {safeguard} missing from architecture.md"


def test_R1_1_architecture_doc_describes_each_agent_role():
    text = (SOLUTION / "architecture.md").read_text()
    # Each agent needs prose, not just a mention in a diagram.
    for agent in [
        "StructuredDataAgent",
        "UnstructuredDataAgent",
        "MultimodalDataAgent",
        "TimeSeriesDataAgent",
    ]:
        section = re.search(rf"###.*{agent}.*?\n(.+?)(?=\n###|\Z)", text, re.DOTALL)
        assert section, f"no role description section for {agent}"
        assert len(section.group(1).strip()) > 150, f"role description for {agent} is too thin"


# ===========================================================================
# Section 2 — Chat Orchestrator and Query Routing
# ===========================================================================

def test_R2_1_manager_exposes_required_methods(chat_module):
    manager_cls = chat_module.AgentChatManager
    for method in [
        "_load_agents",
        "_route_query",
        "_run_structured",
        "_run_unstructured",
        "_run_multimodal",
        "_run_timeseries",
        "chat",
    ]:
        assert callable(getattr(manager_cls, method, None)), f"missing {method}"


def test_R2_1_load_agents_constructs_all_four(manager, chat_module):
    fakes = chat_module._fakes
    fakes.structured.assert_called_once()
    fakes.unstructured.assert_called_once()
    fakes.multimodal.assert_called_once()
    fakes.timeseries.assert_called_once()

    assert manager.structured is fakes.structured.return_value
    assert manager.unstructured is fakes.unstructured.return_value
    assert manager.multimodal is fakes.multimodal.return_value
    assert manager.timeseries is fakes.timeseries.return_value

    # The RAG index has to be built at load time or the first query finds nothing.
    fakes.unstructured.return_value.build_index.assert_called_once()


def test_R2_1_agent_kwargs_match_the_documented_contract(manager, chat_module):
    fakes = chat_module._fakes

    s_kwargs = fakes.structured.call_args.kwargs
    assert s_kwargs["deployment"] == "gpt-4.1-mini"
    assert s_kwargs["db_config"]["port"] == 5432
    assert s_kwargs["db_config"]["dbname"] == "neighborhoods"

    u_kwargs = fakes.unstructured.call_args.kwargs
    assert u_kwargs["chroma_path"] == "./data/chroma_db_storage"
    assert u_kwargs["embedding_deployment"] == "text-embedding-ada-002"
    assert u_kwargs["chat_deployment"] == "gpt-4.1-mini"

    m_kwargs = fakes.multimodal.call_args.kwargs
    assert m_kwargs["container_name"] == "houses"

    t_kwargs = fakes.timeseries.call_args.kwargs
    assert t_kwargs["table_name"] == "SensorReadings"


ROUTING_CASES = [
    # The six evaluation prompts the project brief names explicitly.
    ("Which demographic group has the most expensive homes?", "structured"),
    ("How many homes are owned by hispanics in Ashford neighborhood?", "structured"),
    ("Was a permit approved for a restaurant in any of the neighborhoods?", "unstructured"),
    ("Show me all the permits for pools in each neighborhood", "unstructured"),
    ("Find me a house like mine", "multimodal"),
    # This one routed to the WRONG agent with the keyword list as shipped: it
    # matches no multimodal keyword ("similar house" does not occur in "houses
    # similar to") and fell through the default to unstructured.
    ("Show me houses similar to my house", "multimodal"),
    # Fourth agent.
    ("Are there any unusual air quality readings in Maplewood?", "timeseries"),
    ("What is the average noise level per neighborhood?", "timeseries"),
    # Priority ordering: permit wins over neighborhood.
    ("What is the average income in Rosedale?", "structured"),
    ("Show me a picture of a house", "multimodal"),
]


@pytest.mark.parametrize("prompt,expected", ROUTING_CASES)
def test_R2_2_routing_classifies_each_query(manager, prompt, expected):
    assert manager._route_query(prompt) == expected, (
        f"{prompt!r} should route to {expected}"
    )


def test_R2_2_chat_dispatches_to_the_matching_wrapper(manager):
    for prompt, expected in [
        ("Which demographic group has the most expensive homes?", "_run_structured"),
        ("Was a permit approved for a restaurant?", "_run_unstructured"),
        ("Find me a house like mine", "_run_multimodal"),
        ("Are there unusual sensor readings?", "_run_timeseries"),
    ]:
        with mock.patch.object(manager, expected, return_value="ok") as wrapper:
            manager.chat(prompt)
        wrapper.assert_called_once_with(prompt)


def test_R2_3_no_todo_markers_remain_in_solution():
    offenders = []
    for path in solution_python_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if TODO_MARKER in text:
            offenders.append(path.relative_to(SOLUTION))
    assert not offenders, f"unfinished TODO markers in {offenders}"


def test_R2_3_starter_is_left_untouched():
    """The starter must stay pristine — the solution lives in solution/."""
    # 60 angle-bracket placeholders across the four agent/orchestrator files.
    # (The often-quoted "62" also counts two plain "# TODO:" comments in the
    # load scripts, which are prose rather than syntax errors.)
    starter_todos = sum(
        p.read_text(encoding="utf-8", errors="replace").count(TODO_MARKER)
        for p in STARTER.rglob("*.py")
    )
    assert starter_todos == 60, (
        f"starter/ should still contain its 60 placeholder markers, found {starter_todos}"
    )


def test_R2_3_every_solution_module_parses():
    for path in solution_python_files():
        source = path.read_text(encoding="utf-8", errors="replace")
        ast.parse(source, filename=str(path))


def test_R2_3_repl_is_guarded_by_main(chat_module):
    """Importing chat must not start the interactive loop."""
    tree = ast.parse((SOLUTION / "chat.py").read_text())
    module_level_loops = [n for n in tree.body if isinstance(n, ast.While)]
    assert not module_level_loops, "the REPL loop must sit under if __name__ == '__main__'"


def test_R2_4_manager_answers_each_agent_type(manager):
    """One query per agent type returns a non-empty string."""
    manager.structured.ask.return_value = {"response": "structured answer"}
    manager.unstructured.ask.return_value = {"response": "unstructured answer"}
    manager.timeseries.ask.return_value = {"response": "timeseries answer"}
    manager.multimodal.find_similar.return_value = (
        [(0.9, "a.jpg", "a")], object()
    )
    manager.multimodal.last_safety_findings = []
    manager.multimodal.blocked_blobs = []

    for prompt in [
        "Which demographic group has the most expensive homes?",
        "Was a permit approved for a restaurant?",
        "Are there unusual sensor readings?",
    ]:
        answer = manager.chat(prompt)
        assert isinstance(answer, str) and answer.strip()
        assert not answer.startswith("An error occurred")


# ===========================================================================
# Section 3 — Agent Implementation
# ===========================================================================

@pytest.fixture
def structured_agent():
    """A real StructuredDataAgent with only the Azure/DB boundaries faked."""
    import agents.structured_data_agent as mod

    with mock.patch.object(mod, "AzureChatOpenAI") as llm, \
         mock.patch.object(mod, "SQLDatabase") as sqldb, \
         mock.patch.object(mod, "SQLDatabaseToolkit") as toolkit, \
         mock.patch.object(mod, "create_react_agent") as react:
        agent = mod.StructuredDataAgent(
            azure_endpoint="https://example.openai.azure.com/",
            api_key="k",
            deployment="gpt-4.1-mini",
            db_config={
                "host": "h.postgres.database.azure.com",
                "dbname": "neighborhoods",
                "user": "nbhdadmin",
                "password": "p@ss/word+1",
                "port": 5432,
            },
        )
        agent._probe = mock.MagicMock(llm=llm, sqldb=sqldb, toolkit=toolkit, react=react)
        yield agent


def test_R3_1_structured_builds_psycopg3_uri_with_tls(structured_agent):
    uri = structured_agent.db_uri
    # psycopg[binary] is v3. A bare postgresql:// scheme makes SQLAlchemy reach
    # for psycopg2, which is not installed.
    assert uri.startswith("postgresql+psycopg://"), uri
    # Azure Flexible Server runs require_secure_transport=ON.
    assert "sslmode=require" in uri, uri
    # The password must be percent-encoded, not passed through raw.
    assert "p%40ss%2Fword%2B1" in uri, uri
    assert "p@ss/word+1" not in uri
    structured_agent._probe.sqldb.from_uri.assert_called_once_with(uri)


def test_R3_1_structured_builds_react_agent_over_sql_toolkit(structured_agent):
    structured_agent._probe.toolkit.assert_called_once()
    kwargs = structured_agent._probe.toolkit.call_args.kwargs
    assert kwargs["db"] is structured_agent.db
    assert kwargs["llm"] is structured_agent.model
    structured_agent._probe.react.assert_called_once()


def test_R3_4_structured_ask_returns_response_dict(structured_agent):
    chunk = {"messages": [mock.MagicMock(content="the answer", type="ai")]}
    structured_agent.agent_executor.stream.return_value = [chunk]
    with mock.patch.object(structured_agent, "_StructuredDataAgent__auto_bias_check"):
        result = structured_agent.ask("q", run_bias_audit=True)

    assert isinstance(result, dict)
    # chat.py reads result.get("response"); a miss would dump a raw dict at the user.
    assert result["response"] == "the answer"
    assert result["question"] == "q"


@pytest.fixture
def unstructured_agent():
    import agents.unstructured_data_agent as mod

    with mock.patch.object(mod, "MongoClient") as mongo, \
         mock.patch.object(mod, "Chroma") as chroma, \
         mock.patch.object(mod, "AzureOpenAIEmbeddings") as emb, \
         mock.patch.object(mod, "AzureChatOpenAI") as llm:
        agent = mod.UnstructuredDataAgent(
            mongo_uri="mongodb://x",
            db_name="permits",
            collection_name="permit_documents",
            chroma_path="./data/chroma_db_storage",
            azure_endpoint="https://example.openai.azure.com/",
            azure_key="k",
            azure_api_version="2023-06-01-preview",
            embedding_deployment="text-embedding-ada-002",
            chat_deployment="gpt-4.1-mini",
        )
        agent._probe = mock.MagicMock(mongo=mongo, chroma=chroma, emb=emb, llm=llm)
        yield agent


def test_R3_2_unstructured_wires_mongo_chroma_and_embeddings(unstructured_agent):
    a = unstructured_agent
    a._probe.emb.assert_called_once()
    assert a._probe.emb.call_args.kwargs["azure_deployment"] == "text-embedding-ada-002"

    a._probe.chroma.assert_called_once()
    chroma_kwargs = a._probe.chroma.call_args.kwargs
    assert chroma_kwargs["persist_directory"] == "./data/chroma_db_storage"
    assert chroma_kwargs["embedding_function"] is a.embeddings

    # The attribute name mongo_coll is load-bearing: build_index and ask both use it.
    assert a.mongo_coll is a._probe.mongo.return_value["permits"]["permit_documents"]


def test_R3_2_build_index_embeds_and_upserts_by_mongo_id(unstructured_agent):
    from bson import ObjectId

    oid1, oid2, oid3 = ObjectId(), ObjectId(), ObjectId()
    unstructured_agent.mongo_coll.find.return_value = [
        {"_id": oid1, "content": "permit one"},
        {"_id": oid2, "content": "   "},        # blank, must be skipped
        {"_id": oid3, "content": "permit three"},
    ]
    unstructured_agent.build_index()

    kwargs = unstructured_agent.vector_db.add_texts.call_args.kwargs
    assert kwargs["texts"] == ["permit one", "permit three"]
    # Explicit ids make re-indexing an upsert rather than a duplication.
    assert kwargs["ids"] == [str(oid1), str(oid3)]


def test_R3_2_ask_runs_similarity_search_then_rag(unstructured_agent):
    from bson import ObjectId

    oid = ObjectId()
    unstructured_agent.vector_db.similarity_search.return_value = [
        mock.MagicMock(metadata={"mongo_id": str(oid)})
    ]
    unstructured_agent.mongo_coll.find_one.return_value = {
        "_id": oid, "content": "Permit approved for a restaurant.", "filename": "x.pdf",
    }
    unstructured_agent.llm.invoke.return_value = mock.MagicMock(content="Yes, approved.")

    with mock.patch.object(unstructured_agent, "_UnstructuredDataAgent__contains_pii"):
        result = unstructured_agent.ask("restaurant permit?", run_pii_audit=True)

    unstructured_agent.vector_db.similarity_search.assert_called_once()
    # The answer must be grounded in the document fetched back from Mongo.
    prompt = unstructured_agent.llm.invoke.call_args.args[0]
    assert "Permit approved for a restaurant." in prompt
    assert "restaurant permit?" in prompt
    assert result["response"] == "Yes, approved."


def test_R3_4_ask_with_no_hits_still_returns_a_response_key(unstructured_agent):
    unstructured_agent.vector_db.similarity_search.return_value = []
    result = unstructured_agent.ask("nothing matches this")
    # The shipped early return used only "answer"/"sources", so chat.py would have
    # rendered the raw dict instead of the sentence.
    assert result["response"] == "No relevant documents found."


@pytest.fixture
def multimodal_agent():
    import agents.multimodal_data_agent as mod

    with mock.patch.object(mod, "CLIPModel"), \
         mock.patch.object(mod, "CLIPProcessor"), \
         mock.patch.object(mod, "BlobServiceClient"), \
         mock.patch.object(mod, "ContentSafetyClient"):
        agent = mod.MultimodalDataAgent(
            azure_conn_str="conn",
            container_name="houses",
            content_safety_endpoint="https://cs.cognitiveservices.azure.com/",
            content_safety_key="k",
        )
        yield agent


def test_R3_3_find_similar_returns_sorted_top_k_and_query_image(multimodal_agent):
    blobs = [mock.MagicMock() for _ in range(5)]
    names = ["a.jpg", "b.jpg", "notes.txt", "c.jpg", "d.jpg"]
    for b, n in zip(blobs, names):
        b.name = n
    multimodal_agent.container_client.list_blobs.return_value = blobs

    scores = {"a.jpg": 0.5, "b.jpg": 0.9, "c.jpg": 0.7, "d.jpg": 0.1}
    sentinel = object()

    with mock.patch.object(multimodal_agent, "check_query_image", return_value=(False, {})), \
         mock.patch.object(
             multimodal_agent, "_MultimodalDataAgent__compute_similarity_blob",
             side_effect=lambda q, n: scores[n]), \
         mock.patch.object(
             multimodal_agent, "_MultimodalDataAgent__load_image", return_value=sentinel):
        matches, query_img = multimodal_agent.find_similar("query-house-2.jpg", top_k=3)

    assert [m[1] for m in matches] == ["b.jpg", "c.jpg", "a.jpg"], "must be sorted desc"
    assert len(matches) == 3
    assert all(not m[1].endswith(".txt") for m in matches), "non-images must be skipped"
    assert query_img is sentinel


def test_R3_3_address_is_derived_from_the_blob_name(multimodal_agent):
    blob = mock.MagicMock()
    blob.name = "126_Briarwood_Drive_Ashford.jpg"
    multimodal_agent.container_client.list_blobs.return_value = [blob]

    with mock.patch.object(multimodal_agent, "check_query_image", return_value=(False, {})), \
         mock.patch.object(
             multimodal_agent, "_MultimodalDataAgent__compute_similarity_blob",
             return_value=0.5), \
         mock.patch.object(multimodal_agent, "_MultimodalDataAgent__load_image"):
        matches, _ = multimodal_agent.find_similar("q.jpg")

    assert matches[0][2] == "126 Briarwood Drive Ashford"


# ===========================================================================
# Section 4 — Ethical Safeguards and Federated Data Management
# ===========================================================================

def test_R4_1_bias_audit_runs_during_query_processing(structured_agent):
    chunk = {"messages": [mock.MagicMock(content="answer", type="ai")]}
    structured_agent.agent_executor.stream.return_value = [chunk]

    with mock.patch.object(structured_agent, "_StructuredDataAgent__auto_bias_check") as audit:
        structured_agent.ask("q", run_bias_audit=True)
    audit.assert_called_once_with(structured_agent.db_uri)

    with mock.patch.object(structured_agent, "_StructuredDataAgent__auto_bias_check") as audit:
        structured_agent.ask("q", run_bias_audit=False)
    audit.assert_not_called()


def test_R4_1_fairlearn_metrics_computed_on_a_real_frame(structured_agent, capsys):
    import pandas as pd

    df = pd.DataFrame({
        "identified_race": ["White"] * 10 + ["Black"] * 10,
        # 90% vs 20% selection: a parity difference well past the 0.10 threshold.
        "swimming_pool": [True] * 9 + [False] + [True] * 2 + [False] * 8,
    })
    structured_agent._StructuredDataAgent__run_fairness_analysis(
        df, "neighborhood_houses", "identified_race"
    )
    out = capsys.readouterr().out
    assert "Selection Rate by Group" in out
    assert "Demographic Parity Difference" in out
    assert "Potential bias detected" in out
    assert structured_agent.last_bias_findings, "a flagged disparity must be recorded"


def test_R4_1_text_binary_is_encoded_before_selection_rate(structured_agent, capsys):
    """A TEXT binary must be measured, not silently reported as 0.0000."""
    import pandas as pd

    df = pd.DataFrame({
        "identified_race": ["White"] * 10 + ["Black"] * 10,
        "ownership": ["Own"] * 9 + ["Rent"] + ["Own"] * 2 + ["Rent"] * 8,
    })
    structured_agent._StructuredDataAgent__run_fairness_analysis(
        df, "neighborhood_houses", "identified_race"
    )
    out = capsys.readouterr().out
    assert "Positive label: 'Own'" in out
    match = re.search(r"Demographic Parity Difference: ([\d.]+)", out)
    assert match, out
    # With selection_rate's default pos_label=1 against strings this was 0.0000.
    assert float(match.group(1)) > 0.5, "the TEXT binary was not actually measured"


def test_R4_1_demographic_detection_finds_labels_not_counts(structured_agent):
    """The single highest-risk line in the project: keyword precision."""
    found = set()
    for columns in REAL_COLUMNS.values():
        found.update(
            structured_agent._StructuredDataAgent__find_demographic_columns(columns)
        )
    # identified_race and gender are the only true per-row group labels. Columns
    # like male_students and white_patients are aggregate counts; grouping on them
    # would emit statistically meaningless output under a fairness heading.
    assert found == {"identified_race", "gender"}, found


def test_R4_2_pii_audit_runs_during_query_processing(unstructured_agent):
    from bson import ObjectId

    oid = ObjectId()
    unstructured_agent.vector_db.similarity_search.return_value = [
        mock.MagicMock(metadata={"mongo_id": str(oid)})
    ]
    unstructured_agent.mongo_coll.find_one.return_value = {
        "_id": oid, "content": "text", "filename": "f.pdf"
    }
    unstructured_agent.llm.invoke.return_value = mock.MagicMock(content="answer")

    with mock.patch.object(unstructured_agent, "_UnstructuredDataAgent__contains_pii") as pii:
        unstructured_agent.ask("q", run_pii_audit=True)
    pii.assert_called_once()

    with mock.patch.object(unstructured_agent, "_UnstructuredDataAgent__contains_pii") as pii:
        unstructured_agent.ask("q", run_pii_audit=False)
    pii.assert_not_called()


def test_R4_2_pii_filter_reports_confidence_and_dedupes(unstructured_agent, capsys):
    text = "Lydia R. Montrose lmontrose@mcc-devgroup.com Lydia R. Montrose http://x.io"

    def result(entity, start, end, score):
        r = mock.MagicMock()
        r.entity_type, r.start, r.end, r.score = entity, start, end, score
        return r

    findings = [
        result("PERSON", 0, 17, 0.85),                 # boundary: kept (< is strict)
        result("EMAIL_ADDRESS", 18, 44, 1.00),         # kept
        result("PERSON", 45, 62, 0.85),                # duplicate text: deduped
        result("URL", 63, 74, 1.00),                   # URL: always dropped
        result("DATE_TIME", 0, 5, 0.40),               # below threshold: dropped
    ]
    analyzer = mock.MagicMock()
    analyzer.analyze.return_value = findings

    with mock.patch("agents.unstructured_data_agent._get_analyzer", return_value=analyzer):
        unstructured_agent.last_pii_findings = []
        unstructured_agent._UnstructuredDataAgent__contains_pii(text)

    out = capsys.readouterr().out
    assert "confidence=0.85" in out and "confidence=1.00" in out
    assert "http://x.io" not in out, "URLs must be filtered"
    assert out.count("Lydia R. Montrose") == 1, "duplicates must be deduped"

    types = [f["entity_type"] for f in unstructured_agent.last_pii_findings]
    assert types == ["PERSON", "EMAIL_ADDRESS"]


def test_R4_2_detected_pii_is_redacted_from_the_answer(unstructured_agent):
    """Actionable safeguard: detecting PII then echoing it would defeat the check."""
    from bson import ObjectId

    oid = ObjectId()
    unstructured_agent.vector_db.similarity_search.return_value = [
        mock.MagicMock(metadata={"mongo_id": str(oid)})
    ]
    unstructured_agent.mongo_coll.find_one.return_value = {
        "_id": oid, "content": "Applicant: Lydia R. Montrose", "filename": "f.pdf"
    }
    unstructured_agent.llm.invoke.return_value = mock.MagicMock(
        content="The applicant is Lydia R. Montrose."
    )

    answer = "The applicant is Lydia R. Montrose."

    def analyze(text, language):
        # Presidio is asked about whatever text it is handed. Return the span of
        # the name within THAT text, which is the behaviour the redactor relies on.
        start = text.find("Lydia R. Montrose")
        if start == -1:
            return []
        r = mock.MagicMock()
        r.entity_type, r.start, r.end, r.score = (
            "PERSON", start, start + len("Lydia R. Montrose"), 0.95,
        )
        return [r]

    analyzer = mock.MagicMock()
    analyzer.analyze.side_effect = analyze
    unstructured_agent.llm.invoke.return_value = mock.MagicMock(content=answer)

    with mock.patch("agents.unstructured_data_agent._get_analyzer", return_value=analyzer):
        result = unstructured_agent.ask("who applied?", run_pii_audit=True)

    assert "Lydia R. Montrose" not in result["response"]
    assert "[REDACTED PERSON]" in result["response"]


def test_R4_2_redaction_survives_ragged_source_spans(unstructured_agent):
    """Spans detected in the source bleed into layout; the answer must still be clean.

    Presidio over PDF text returns things like
    'Lydia R. Montrose\n• Company/Developer'. Matching that string into a clean
    answer fails, which once leaked every permit owner's name.
    """
    findings = [{
        "entity_type": "PERSON",
        "text": "Morgan Ellis\n• Company/Developer",
        "score": 0.85,
    }]
    analyzer = mock.MagicMock()
    analyzer.analyze.return_value = []   # nothing found in the answer itself

    with mock.patch("agents.unstructured_data_agent._get_analyzer", return_value=analyzer):
        cleaned = unstructured_agent._UnstructuredDataAgent__redact(
            "Owner: Morgan Ellis", findings
        )

    assert "Morgan Ellis" not in cleaned
    assert "[REDACTED PERSON]" in cleaned


def test_R4_3_content_safety_runs_on_every_image(multimodal_agent, capsys):
    category = lambda name, sev: mock.MagicMock(category=name, severity=sev)
    response = mock.MagicMock()
    response.categories_analysis = [
        category("Hate", 0), category("SelfHarm", 0),
        category("Sexual", 0), category("Violence", 0),
    ]
    multimodal_agent.content_safety_client.analyze_image.return_value = response

    blob_client = mock.MagicMock()
    blob_client.download_blob.return_value.readall.return_value = _tiny_jpeg()
    multimodal_agent.container_client.get_blob_client.return_value = blob_client

    multimodal_agent._MultimodalDataAgent__load_image_from_blob("a.jpg")

    multimodal_agent.content_safety_client.analyze_image.assert_called_once()
    out = capsys.readouterr().out
    for name in ["Hate", "SelfHarm", "Sexual", "Violence"]:
        assert name in out, f"category {name} severity must be printed"
    assert "No harmful content detected" in out


def test_R4_3_flagged_images_are_excluded_from_results(multimodal_agent):
    blobs = []
    for name in ["safe.jpg", "unsafe.jpg"]:
        b = mock.MagicMock()
        b.name = name
        blobs.append(b)
    multimodal_agent.container_client.list_blobs.return_value = blobs

    def fake_similarity(query, name):
        multimodal_agent._safety_cache[name] = (name == "unsafe.jpg")
        return 0.9 if name == "unsafe.jpg" else 0.5

    with mock.patch.object(multimodal_agent, "check_query_image", return_value=(False, {})), \
         mock.patch.object(
             multimodal_agent, "_MultimodalDataAgent__compute_similarity_blob",
             side_effect=fake_similarity), \
         mock.patch.object(multimodal_agent, "_MultimodalDataAgent__load_image"):
        matches, _ = multimodal_agent.find_similar("q.jpg")

    names = [m[1] for m in matches]
    assert "unsafe.jpg" not in names, "flagged image must not be returned"
    assert multimodal_agent.blocked_blobs == ["unsafe.jpg"]


def test_R4_3_query_image_is_screened_before_the_search(multimodal_agent):
    """The user's own image is otherwise the only unscanned input in the system."""
    multimodal_agent.container_client.list_blobs.return_value = []
    with mock.patch.object(
        multimodal_agent, "check_query_image", return_value=(True, {"Violence": 4})
    ) as check, pytest.raises(ValueError, match="flagged by"):
        multimodal_agent.find_similar("query-house-2.jpg")
    check.assert_called_once()


def test_R4_3_oversized_image_is_downscaled_for_content_safety(multimodal_agent):
    """274_Maplewood_Crescent.jpg is 2268x1676; the service limit is 2048."""
    from io import BytesIO

    from PIL import Image

    import agents.multimodal_data_agent as mod

    buf = BytesIO()
    Image.new("RGB", (2268, 1676), "white").save(buf, format="JPEG")
    result = multimodal_agent._MultimodalDataAgent__downscale_for_safety(buf.getvalue())

    assert max(Image.open(BytesIO(result)).size) <= mod.CONTENT_SAFETY_MAX_DIM


def test_R4_3_imagedata_serializes_raw_bytes_as_base64():
    """Guards the azure-ai-contentsafety 1.0.0 vs 1.0.0b1 encoding difference."""
    import base64

    from azure.ai.contentsafety.models import ImageData

    raw = b"\x89PNG\r\n\x1a\n" + b"payload"
    serialized = ImageData(content=raw).as_dict()["content"]
    assert isinstance(serialized, str), "1.0.0b1 would require manual base64 encoding"
    assert base64.b64decode(serialized) == raw


def test_R4_4_all_sixteen_secrets_come_from_key_vault(chat_module):
    kv = chat_module._fakes.kv
    assert kv.requested == SECRET_NAMES, (
        "chat.py must request exactly these secrets, in this order"
    )
    assert kv.vault_url == f"https://{chat_module.keyVaultName}.vault.azure.net/"


def test_R4_4_credential_is_device_code(chat_module):
    """DeviceCodeCredential is the documented path and must be constructed.

    It sits behind AzureCliCredential in a chain so non-interactive runs reuse an
    existing `az login` instead of blocking on a prompt nobody can answer, but the
    device-code flow is still what an interactive user without a CLI session gets.
    """
    chat_module._fakes.credential.assert_called_once_with()

    source = (SOLUTION / "chat.py").read_text()
    assert "ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())" in source


def test_R4_4_loader_secret_names_are_a_subset_of_the_vault():
    """A typo in a loader's secret name is unrecoverable without re-provisioning."""
    requested = set()
    for name in [
        "structured_load_azure.py",
        "unstructured_load_azure.py",
        "timeseries_load_azure.py",
    ]:
        tree = ast.parse((SOLUTION / "data" / name).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_secret"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                requested.add(node.args[0].value)

    unknown = requested - set(SECRET_NAMES)
    assert not unknown, f"loaders request secrets that provisioning never creates: {unknown}"


SECRET_PATTERNS = [
    (re.compile(r"AccountKey=[A-Za-z0-9+/]{20,}"), "storage account key"),
    (re.compile(r"mongodb://[^\"'\s]*:[^\"'\s@]{8,}@"), "mongo connection string with password"),
    (re.compile(r"""password\s*=\s*["'][^"'{}$]{8,}["']"""), "hardcoded password"),
    (re.compile(r"""api[_-]?key\s*=\s*["'][A-Za-z0-9]{20,}["']"""), "hardcoded api key"),
]


def test_R4_4_no_hardcoded_secrets_anywhere_in_the_solution():
    offenders = []
    for path in SOLUTION.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".md", ".txt", ".ini"}:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                snippet = match.group(0)
                # Documented placeholders and test fixtures are not secrets.
                if any(tok in snippet for tok in ("<", "fake-", "example", "AccountKey=aaaa", "AccountKey=bbbb", "acct:key")):
                    continue
                offenders.append(f"{path.relative_to(SOLUTION)}: {label}")
    assert not offenders, f"possible hardcoded secrets: {offenders}"


def test_R4_5_each_agent_module_imports_only_its_own_data_source():
    """Source isolation asserted from live module namespaces, not from grep."""
    import agents.multimodal_data_agent as multimodal
    import agents.structured_data_agent as structured
    import agents.timeseries_data_agent as timeseries
    import agents.unstructured_data_agent as unstructured

    forbidden = {
        structured: ["MongoClient", "BlobServiceClient", "ContentSafetyClient", "TableServiceClient"],
        unstructured: ["create_engine", "BlobServiceClient", "ContentSafetyClient", "TableServiceClient"],
        multimodal: ["MongoClient", "create_engine", "TableServiceClient"],
        timeseries: ["MongoClient", "create_engine", "BlobServiceClient"],
    }
    required = {
        structured: ["create_engine", "SQLDatabase"],
        unstructured: ["MongoClient", "Chroma"],
        multimodal: ["BlobServiceClient", "ContentSafetyClient", "CLIPModel"],
        timeseries: ["TableServiceClient"],
    }

    for module, names in forbidden.items():
        for name in names:
            assert not hasattr(module, name), (
                f"{module.__name__} must not have access to {name}"
            )
    for module, names in required.items():
        for name in names:
            assert hasattr(module, name), f"{module.__name__} is missing {name}"


def test_R4_5_each_agent_receives_only_its_own_credentials(manager, chat_module):
    fakes = chat_module._fakes

    structured_values = str(fakes.structured.call_args)
    assert "mongodb://" not in structured_values
    assert "AccountKey" not in structured_values

    unstructured_values = str(fakes.unstructured.call_args)
    assert "postgres" not in unstructured_values.lower()
    assert "AccountKey" not in unstructured_values

    multimodal_values = str(fakes.multimodal.call_args)
    assert "mongodb://" not in multimodal_values
    assert "postgres" not in multimodal_values.lower()

    # The two storage-backed agents must not share a storage account, or the
    # isolation is nominal rather than real.
    timeseries_values = str(fakes.timeseries.call_args)
    assert "stnbhdimg4hbr" not in timeseries_values
    assert "stnbhdsensors4hbr" not in multimodal_values


def test_R4_6_no_agent_writes_to_another_agents_store():
    """Data stays in situ: agents read their own source and never copy across."""
    write_calls = {"insert_one", "insert_many", "upload_blob", "upsert_entity", "to_sql"}
    for name in [
        "structured_data_agent.py",
        "unstructured_data_agent.py",
        "multimodal_data_agent.py",
        "timeseries_data_agent.py",
    ]:
        tree = ast.parse((SOLUTION / "agents" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in write_calls, (
                    f"{name} calls {node.func.attr}; agents must not write to data stores"
                )


def test_R4_6_provided_data_folders_are_unchanged():
    """data/ must remain exactly as the course shipped it."""
    import hashlib

    for folder in ["structured", "unstructured", "multimodal"]:
        for starter_file in sorted((STARTER / "data" / folder).rglob("*")):
            if not starter_file.is_file() or starter_file.name == ".DS_Store":
                continue
            mirror = SOLUTION / "data" / folder / starter_file.relative_to(
                STARTER / "data" / folder
            )
            assert mirror.exists(), f"missing {mirror}"
            assert (
                hashlib.sha256(mirror.read_bytes()).hexdigest()
                == hashlib.sha256(starter_file.read_bytes()).hexdigest()
            ), f"{mirror} differs from the course-provided file"


# ===========================================================================
# Section 5 — Evaluation and Reflection
# ===========================================================================

EVIDENCE = SOLUTION / "docs" / "evaluation"


def _transcripts():
    return sorted(EVIDENCE.glob("*.log")) if EVIDENCE.exists() else []


def test_R5_1_at_least_two_prompts_per_agent_type():
    from evaluate import PROMPTS

    by_route = {}
    for prompt, route in PROMPTS:
        by_route.setdefault(route, []).append(prompt)

    for route in ["structured", "unstructured", "multimodal", "timeseries"]:
        assert len(by_route.get(route, [])) >= 2, (
            f"the rubric requires >=2 prompts for {route}, found {len(by_route.get(route, []))}"
        )
    assert len(PROMPTS) >= 6


def test_R5_1_every_evaluation_prompt_routes_where_intended(manager):
    """Proves the evidence set actually exercises each agent."""
    from evaluate import PROMPTS

    for prompt, expected in PROMPTS:
        assert manager._route_query(prompt) == expected, (
            f"evaluation prompt {prompt!r} routes to "
            f"{manager._route_query(prompt)}, not {expected}"
        )


@pytest.mark.skipif(not _transcripts(), reason="no evaluation transcripts yet")
def test_R5_2_structured_transcripts_show_fairlearn_output():
    text = "\n".join(p.read_text() for p in EVIDENCE.glob("structured-*.log"))
    assert "Selection Rate by Group" in text
    assert "Demographic Parity Difference" in text


@pytest.mark.skipif(not _transcripts(), reason="no evaluation transcripts yet")
def test_R5_3_unstructured_transcripts_show_pii_with_confidence():
    text = "\n".join(p.read_text() for p in EVIDENCE.glob("unstructured-*.log"))
    assert "PII detected" in text
    assert re.search(r"confidence=\d\.\d\d", text), "PII entities need confidence scores"


@pytest.mark.skipif(not _transcripts(), reason="no evaluation transcripts yet")
def test_R5_4_multimodal_transcripts_show_category_severities():
    text = "\n".join(p.read_text() for p in EVIDENCE.glob("multimodal-*.log"))
    for category in ["Hate", "SelfHarm", "Sexual", "Violence"]:
        assert category in text, f"missing Content Safety category {category}"


@pytest.mark.skipif(not _transcripts(), reason="no evaluation transcripts yet")
def test_R5_4_multimodal_figures_were_captured():
    assert list(EVIDENCE.glob("*.png")), "multimodal PNG evidence missing"


def test_R5_5_written_observations_exist_for_every_agent():
    doc = SOLUTION / "docs" / "EVALUATION.md"
    assert doc.exists(), "EVALUATION.md carries the required written observations"
    text = doc.read_text()

    # Split on headings rather than using a DOTALL regex: `##.*Structured` with
    # re.DOTALL is greedy across newlines and spans from the first heading in the
    # file all the way to the LAST occurrence of the agent name.
    sections, current = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    for agent in ["Structured", "Unstructured", "Multimodal", "Time-Series"]:
        heading = next(
            (h for h in sections if re.search(rf"\b{agent}\b", h, re.IGNORECASE)), None
        )
        assert heading, f"no observation section for the {agent} agent"
        body = "\n".join(sections[heading])
        assert len(body.strip()) > 200, f"{agent} observation is too thin"
        assert re.search(r"detect", body, re.IGNORECASE), f"{agent}: what was detected?"
        assert re.search(r"handl|mitigat|redact|exclud|flag|withh", body, re.IGNORECASE), (
            f"{agent}: how was it handled?"
        )


# ===========================================================================
# Section 6 — Submission checklist
# ===========================================================================

def test_R6_1_all_four_agent_modules_import_cleanly():
    import agents.multimodal_data_agent
    import agents.structured_data_agent
    import agents.timeseries_data_agent
    import agents.unstructured_data_agent  # noqa: F401


def test_R6_2_required_deliverables_exist():
    for relative in [
        "architecture.md",
        "chat.py",
        "agents/structured_data_agent.py",
        "agents/unstructured_data_agent.py",
        "agents/multimodal_data_agent.py",
        "docs/AZURE_SETUP.md",
        "docs/RUBRIC.md",
        "docs/DEVIATIONS.md",
        "docs/EVALUATION.md",
        "README.md",
    ]:
        path = SOLUTION / relative
        assert path.exists(), f"missing deliverable {relative}"
        assert path.stat().st_size > 200, f"{relative} is a stub"


def test_R6_3_audit_log_records_agent_and_check_outcome(tmp_path):
    from audit import AuditLog, summarize_bias

    log = AuditLog(tmp_path / "audit.jsonl")
    log.record(
        query="q",
        route="structured",
        router_method="llm",
        checks=[summarize_bias([{
            "table": "t", "column": "c", "demographic_column": "d",
            "metric": "m", "value": 0.5, "detail": "x",
        }])],
        response_modified=True,
    )
    entries = log.read()
    assert len(entries) == 1
    assert entries[0]["route"] == "structured"
    assert entries[0]["checks"][0]["check"] == "fairlearn_bias_audit"
    assert entries[0]["response_modified"] is True


def _tiny_jpeg() -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="JPEG")
    return buf.getvalue()
