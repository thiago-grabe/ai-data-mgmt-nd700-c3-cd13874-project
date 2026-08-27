# Rubric traceability matrix

Every rubric line maps to an implementation, a mechanical check, and an evidence
artifact. "Meets the rubric" is a command:

```bash
cd solution
.venv/bin/python -m pytest tests/          # offline conformance suite, no Azure calls
.venv/bin/python verify_deliverables.py    # artifact + evidence acceptance gate
.venv/bin/python scripts/verify_azure.py   # live Azure probes
```

The IDs below (R1.1 … R6.4) are the test names in `tests/test_rubric.py`.

---

## Section 1 — Architecture Design

> Design and document a multi-agent system architecture with a central
> orchestrator, specialized agents, and ethical safeguards.

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R1.1 | Architecture documented with diagram and agent roles | `architecture.md` | `test_R1_1_architecture_doc_covers_orchestrator_agents_and_safeguards` — asserts a Mermaid diagram plus every agent, data source, and safeguard by name; `test_R1_1_architecture_doc_describes_each_agent_role` requires substantive prose per agent |

---

## Section 2 — Chat Orchestrator and Query Routing

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R2.1 | `AgentChatManager` with `_load_agents()`, `_route_query()`, wrapper methods | `chat.py` | `test_R2_1_manager_exposes_required_methods`, `test_R2_1_load_agents_constructs_all_four`, `test_R2_1_agent_kwargs_match_the_documented_contract` |
| R2.2 | Routing classifies and dispatches to each agent | `chat.py:_route_query`, `router_llm.py` | `test_R2_2_routing_classifies_each_query` (10 cases incl. every example query from the brief), `test_R2_2_chat_dispatches_to_the_matching_wrapper` |
| R2.3 | `python chat.py` starts without import errors or crashes | `chat.py` | `test_R2_3_no_todo_markers_remain_in_solution`, `test_R2_3_every_solution_module_parses`, `test_R2_3_repl_is_guarded_by_main`, `test_R2_3_starter_is_left_untouched` |
| R2.4 | Responds to at least one query per agent type | `chat.py:chat` | `test_R2_4_manager_answers_each_agent_type`; **live evidence**: `docs/evaluation/*.log` |

---

## Section 3 — Agent Implementation

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R3.1 | Structured agent connects to PostgreSQL and runs NL→SQL | `agents/structured_data_agent.py` | `test_R3_1_structured_builds_psycopg3_uri_with_tls` (asserts `postgresql+psycopg://`, `sslmode=require`, and password percent-encoding), `test_R3_1_structured_builds_react_agent_over_sql_toolkit` |
| R3.2 | Unstructured agent: MongoDB → ChromaDB via `AzureOpenAIEmbeddings` → similarity search → RAG via `AzureChatOpenAI` | `agents/unstructured_data_agent.py` | `test_R3_2_unstructured_wires_mongo_chroma_and_embeddings`, `test_R3_2_build_index_embeds_and_upserts_by_mongo_id`, `test_R3_2_ask_runs_similarity_search_then_rag` |
| R3.3 | Multimodal agent: Blob images, CLIP similarity, top-k | `agents/multimodal_data_agent.py` | `test_R3_3_find_similar_returns_sorted_top_k_and_query_image`, `test_R3_3_address_is_derived_from_the_blob_name` |
| R3.4 | Each agent queries its source and returns a response | all four agents | `test_R3_4_structured_ask_returns_response_dict`, `test_R3_4_ask_with_no_hits_still_returns_a_response_key` |

---

## Section 4 — Ethical Safeguards and Federated Data Management

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R4.1 | Fairlearn bias audit during query processing | `structured_data_agent.py:__auto_bias_check` | `test_R4_1_bias_audit_runs_during_query_processing`, `test_R4_1_fairlearn_metrics_computed_on_a_real_frame`, `test_R4_1_text_binary_is_encoded_before_selection_rate`, `test_R4_1_demographic_detection_finds_labels_not_counts` |
| R4.2 | Presidio PII detection during query processing | `unstructured_data_agent.py:__contains_pii` | `test_R4_2_pii_audit_runs_during_query_processing`, `test_R4_2_pii_filter_reports_confidence_and_dedupes`, `test_R4_2_detected_pii_is_redacted_from_the_answer` |
| R4.3 | Azure Content Safety during query processing | `multimodal_data_agent.py:__analyze_bytes` | `test_R4_3_content_safety_runs_on_every_image`, `test_R4_3_flagged_images_are_excluded_from_results`, `test_R4_3_query_image_is_screened_before_the_search`, `test_R4_3_oversized_image_is_downscaled_for_content_safety`, `test_R4_3_imagedata_serializes_raw_bytes_as_base64` |
| R4.4 | All credentials from Key Vault at runtime; no hardcoded secrets | `chat.py` lines 40–80 | `test_R4_4_all_sixteen_secrets_come_from_key_vault` (exact names, exact order), `test_R4_4_credential_is_device_code`, `test_R4_4_loader_secret_names_are_a_subset_of_the_vault`, `test_R4_4_no_hardcoded_secrets_anywhere_in_the_solution` |
| R4.5 | Each agent connects exclusively to its own data source | agent constructors | `test_R4_5_each_agent_module_imports_only_its_own_data_source` (live module namespaces, not grep), `test_R4_5_each_agent_receives_only_its_own_credentials`; **live**: probe A9 |
| R4.6 | Data not centralized or copied between sources | all four agents | `test_R4_6_no_agent_writes_to_another_agents_store` (AST scan for write calls), `test_R4_6_provided_data_folders_are_unchanged` (SHA-256 vs `starter/`) |

### Evidence: per-agent Key Vault configuration

Each agent receives only its own connection configuration. From `chat.py:_load_agents`:

```python
self.structured = StructuredDataAgent(
    azure_endpoint=structuredazureendpoint,   # Azure OpenAI key1
    api_key=structuredazureapikey,
    deployment="gpt-4.1-mini",
    db_config={"host": structuredpostgresqlhost, ...},   # PostgreSQL only
)
self.unstructured = UnstructuredDataAgent(
    mongo_uri=unstructuredmongourl,           # MongoDB only
    azure_key=unstructuredazurekey,           # Azure OpenAI key2 — a different key
    ...
)
self.multimodal = MultimodalDataAgent(
    azure_conn_str=multimodalazureconnstring,  # storage account stnbhdimg4hbr
    content_safety_endpoint=multimodalazurecontentsafetyendpoint,
    ...
)
self.timeseries = TimeSeriesDataAgent(
    connection_string=timeseriesazureconnstring,  # storage account stnbhdsensors4hbr
    ...
)
```

The two storage-backed agents use **separate storage accounts** and the two Azure
OpenAI consumers use **different keys on the same account**, so every agent's
credential is independently rotatable and a leak is contained to one source.

---

## Section 5 — Evaluation and Reflection

| ID | Requirement | Implementation | Verification |
|---|---|---|---|
| R5.1 | ≥2 prompts per agent type (≥6 total) | `evaluate.py:PROMPTS` — 8 prompts, 2 per agent | `test_R5_1_at_least_two_prompts_per_agent_type`, `test_R5_1_every_evaluation_prompt_routes_where_intended` |
| R5.2 | Structured output shows Fairlearn results | `docs/evaluation/structured-*.log` | `test_R5_2_structured_transcripts_show_fairlearn_output` |
| R5.3 | Unstructured output shows Presidio entities with confidence scores | `docs/evaluation/unstructured-*.log` | `test_R5_3_unstructured_transcripts_show_pii_with_confidence` |
| R5.4 | Multimodal output shows Content Safety category severities | `docs/evaluation/multimodal-*.log`, `*.png` | `test_R5_4_multimodal_transcripts_show_category_severities`, `test_R5_4_multimodal_figures_were_captured` |
| R5.5 | Written observation per agent | `docs/EVALUATION.md` | `test_R5_5_written_observations_exist_for_every_agent` |

---

## Section 6 — Submission checklist

| ID | Requirement | Verification |
|---|---|---|
| R6.1 | All TODOs completed in `chat.py` and the three agent files | `test_R2_3_no_todo_markers_remain_in_solution`, `test_R6_1_all_four_agent_modules_import_cleanly` |
| R6.2 | `architecture.md` created; all deliverables present | `test_R6_2_required_deliverables_exist` |
| R6.3 | Runs end-to-end without errors | `test_R6_3_audit_log_records_agent_and_check_outcome`; **live**: `evaluate.py` exits 0 |
| R6.4 | `data/` unchanged | `test_R4_6_provided_data_folders_are_unchanged` |

---

## Stand-out enhancements

Beyond the required rubric, all five suggested enhancements are implemented:

| Suggestion | Where |
|---|---|
| A fourth agent for a new data type | `agents/timeseries_data_agent.py` — Azure Table Storage sensor readings with z-score anomaly detection |
| LLM-based query routing instead of keyword matching | `router_llm.py` — `gpt-4.1-mini` classifier with the keyword router as deterministic fallback |
| Make ethical checks actionable | PII redacted from responses, flagged images excluded from results, bias findings prepended as a caveat, unreliable sensor windows withheld |
| Audit log per query | `audit.py` → `logs/audit_log.jsonl`: agent invoked, routing method, check run, outcome, whether the response was modified |
| Content-safety the user's query image | `multimodal_data_agent.py:check_query_image`, called before the similarity sweep |

## Notes for the reviewer

- **`keyVaultName` is a literal (env-overridable), and that is deliberate.** A
  vault name is a resource identifier, not a secret: it grants nothing without an
  Entra ID token, and the bootstrap has to start somewhere. Real secrets are never
  in source, and `test_R4_4_no_hardcoded_secrets_anywhere_in_the_solution` scans
  every committed file to prove it.
- **`starter/` is untouched.** All work is in `solution/`, which began as a
  byte-for-byte copy. Every subsequent change to already-written starter code is
  listed with its reasoning in `docs/DEVIATIONS.md`.
