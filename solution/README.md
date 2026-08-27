# Ethical Multi-Agent Data Orchestrator

A command-line assistant that answers natural-language questions about five
neighborhoods — Ashford, Huntington, Kingsley, Maplewood, Rosedale — by routing
each query to a specialist agent. Every agent owns exactly one data source, holds
only that source's credentials, and runs its own ethical check while answering.

```
You: Which demographic group has the most expensive homes?
[Router] Selected agent: structured (via llm)
================================================================================
Using Strucutred Data stored in PostgresQL database to answer question
Performing a Fairness audit before providing answer
================================================================================
Selection Rate by Group:
                 selection_rate
identified_race
Asian                  0.196794
Black                  0.213611
...
Demographic Parity Difference: 0.0294
```

| Agent | Data source | Ethical check |
|---|---|---|
| Structured | PostgreSQL — demographics & housing | Fairlearn bias audit |
| Unstructured | MongoDB + ChromaDB — permit documents | Presidio PII detection |
| Multimodal | Azure Blob Storage — house photographs | Azure AI Content Safety |
| Time-series | Azure Table Storage — sensor readings | Anomaly / reliability guardrail |

See [`architecture.md`](architecture.md) for the design and
[`docs/EVALUATION.md`](docs/EVALUATION.md) for what the safeguards actually caught.

---

## Getting started

### Dependencies

Python 3.12 and the Azure CLI. **Use 3.12, not 3.14** — the stack pulls three
C-extension chains (torch, the spaCy family, chromadb's core) that must all have
wheels for the interpreter simultaneously.

### Installation

```bash
cd solution
uv venv --python 3.12
uv pip install -r requirements.txt -c constraints.txt
uv pip install -r requirements-dev.txt          # pytest, ruff
```

`constraints.txt` pins four packages, each for a stated reason — read it before
loosening anything. `requirements.txt` adds three entries the course list omits:
`pypdf` (the shipped loader imports a PDF reader that was never listed), the spaCy
`en_core_web_lg` model pinned by URL (Presidio raises `OSError [E050]` without it),
and `azure-data-tables` for the fourth agent.

### Azure setup

```bash
az login --use-device-code
./scripts/provision_azure.sh                  # resources + 16 Key Vault secrets
./scripts/seed_data.sh                        # load all four data stores
.venv/bin/python scripts/verify_azure.py      # 9 probes, all must pass
```

Both scripts are idempotent. [`docs/AZURE_SETUP.md`](docs/AZURE_SETUP.md) explains
every resource, why it was chosen, and a troubleshooting table for the failures
worth anticipating.

> **Re-running the loaders duplicates data silently.** The course loaders use
> `CREATE TABLE IF NOT EXISTS` plus an unconditional `INSERT`. `seed_data.sh`
> guards on row counts and asserts them afterwards; use `FORCE_RELOAD=1` to drop
> and reload. Forcing a MongoDB reload also deletes `data/chroma_db_storage/`,
> because new ObjectIds would leave the vector index pointing at documents that no
> longer exist.

### Running

```bash
cd solution
python chat.py
```

Authenticate with the device code when prompted (an existing `az login` session is
reused automatically), then ask away:

- *Which demographic group has the most expensive homes?*
- *Was a permit approved for a restaurant in any of the neighborhoods?*
- *Find me a house like mine*
- *Are there any unusual air quality readings in Maplewood?*

First run downloads the CLIP weights (~600 MB); it is not hung.

---

## Testing

```bash
.venv/bin/python -m pytest tests/          # rubric conformance — no Azure calls
.venv/bin/ruff check .                     # lint
.venv/bin/python verify_deliverables.py    # artifact + evidence acceptance gate
.venv/bin/python scripts/verify_azure.py   # live Azure probes
```

### What each suite does

**`tests/test_rubric.py`** is one test per rubric line, named `R1.1` … `R6.4` to
match [`docs/RUBRIC.md`](docs/RUBRIC.md). Assertions read live constructed objects
rather than grepping source, because source text can say anything. Every Azure
boundary is patched, so the suite runs offline in about 15 seconds.

The tests worth knowing about:

- `test_R4_1_demographic_detection_finds_labels_not_counts` — runs the detector
  over all 62 real column names and asserts it returns exactly `identified_race`
  and `gender`. The schools and hospitals tables carry `male_students` and
  `white_patients`, which are aggregate counts; grouping on them would emit
  meaningless output under a fairness heading.
- `test_R4_1_text_binary_is_encoded_before_selection_rate` — asserts a TEXT binary
  produces a non-zero parity difference. With `selection_rate`'s default
  `pos_label=1` it silently scored 0.0000 and reported "within threshold" for a
  column never measured.
- `test_R4_5_each_agent_module_imports_only_its_own_data_source` — source isolation
  proved from live module namespaces, not grep.
- `test_R4_4_loader_secret_names_are_a_subset_of_the_vault` — AST-parses the load
  scripts and asserts every `get_secret` name is one provisioning creates.

**`verify_deliverables.py`** checks artifacts rather than behaviour, including the
one thing a working-tree test cannot see: whether the evidence is actually
**tracked by git**. The repo root `.gitignore` has a blanket `*.log` that would
otherwise silently drop every evaluation transcript.

**`scripts/verify_azure.py`** runs nine live probes using the same SDK calls the
agents use, so a pass means the agents will connect.

---

## Evaluation

```bash
.venv/bin/python evaluate.py
```

Runs eight prompts — two per agent — against live Azure, writes a full transcript
per prompt to `docs/evaluation/`, and saves the similarity figures as PNGs. Output
is passed through a redactor first, because a database driver traceback will
happily print a connection string into a file you are about to commit.

Findings are written up in [`docs/EVALUATION.md`](docs/EVALUATION.md). The short
version: Content Safety flagged a house photograph as Violence severity 2 (almost
certainly a false positive, and the system excluded it); Presidio found 60 PII
instances including applicant emails at 1.00 confidence; the anomaly detector
recovered all three injected sensor spikes; and the bias audit found no disparity
above threshold in this dataset.

---

## Project layout

```
solution/
├── chat.py                    # AgentChatManager: secrets, routing, dispatch
├── agents/                    # one module per data source
├── audit.py                   # per-query JSONL audit trail
├── router_llm.py              # LLM classifier + keyword fallback
├── tls_trust.py               # OS trust store, for TLS-intercepting networks
├── evaluate.py                # evaluation harness
├── verify_deliverables.py     # acceptance gate
├── architecture.md            # design + diagram
├── data/                      # course data (unchanged) + timeseries/ (new)
├── docs/                      # AZURE_SETUP, RUBRIC, DEVIATIONS, EVALUATION
├── scripts/                   # provision, seed, verify, teardown
└── tests/                     # rubric conformance suite
```

`../starter/` is untouched and still carries its 60 placeholders.
[`docs/DEVIATIONS.md`](docs/DEVIATIONS.md) lists every change made to code the
course shipped as already-written, with the reasoning for each.

---

## Teardown

The lab subscription bills continuously for the PostgreSQL server.

```bash
./scripts/teardown_azure.sh
```

## Built With

* [LangChain + LangGraph](https://python.langchain.com) — SQL toolkit and the ReAct agent
* [Azure OpenAI](https://azure.microsoft.com/products/ai-services/openai-service) — `gpt-4.1-mini`, `text-embedding-ada-002`
* [Fairlearn](https://fairlearn.org) — demographic parity and selection-rate metrics
* [Presidio](https://microsoft.github.io/presidio/) — PII detection
* [Azure AI Content Safety](https://azure.microsoft.com/products/ai-services/ai-content-safety) — image moderation
* [ChromaDB](https://www.trychroma.com) — vector index
* [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) — image embeddings

## License

[License](../LICENSE.txt)
