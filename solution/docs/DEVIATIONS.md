# Deviations from the starter code

`solution/` began as a byte-for-byte copy of `starter/`. `starter/` is untouched
and still carries its 60 `<TODO …>` placeholders — `test_R2_3_starter_is_left_untouched`
asserts that.

Beyond filling in the placeholders, the changes below alter code that the course
shipped as already-written. Each is listed with what was wrong and why the change
was made, so the difference from the starter is auditable rather than silent.

---

## Defects fixed in shipped code

### 1. The keyword router misroutes one of the project's own example queries

**`chat.py`, `_route_query_keywords`**

Priority runs multimodal → unstructured → structured → default `"unstructured"`.
The prompt **"Show me houses similar to my house"** — a required evaluation query
from the project brief — matches nothing:

- `"similar house"` is in the multimodal list, but the sentence reads
  `"houses similar to"`, so the substring never occurs
- no unstructured or structured keyword matches either

It therefore fell through to the default and reached the **unstructured** agent,
answering a question about photographs from permit PDFs.

**Fix:** added `"houses similar"`, `"similar to my house"`, `"similar to mine"`,
and `"like my house"` to the multimodal list, and put an LLM classifier in front
of the keyword router. `test_R2_2_routing_classifies_each_query` pins every
example query from the brief.

Two related weaknesses in the shipped lists are documented rather than changed,
because they need intent rather than substrings to fix — which is exactly what the
LLM router provides: `"report"` is an unstructured keyword that would hijack
"report the average income", and `"text"` matches inside "context".

### 2. Fairlearn silently fails to measure TEXT binary columns

**`agents/structured_data_agent.py`, `__run_fairness_analysis`**

`selection_rate` defaults to `pos_label=1`. The `ownership` column is TEXT
(`'Own'`/`'Rent'`), so the comparison `'Own' == 1` is never true: every group
scored 0.0000, the demographic parity difference came back 0.0000, and the audit
printed **"Within fairness threshold"** for a column it had never actually
measured. A fairness metric's silence is not evidence of fairness.

**Fix:** `__to_binary_indicator` maps two-valued columns to 0/1 and prints which
value it treated as positive, so the assumption is visible in the transcript.
`test_R4_1_text_binary_is_encoded_before_selection_rate` asserts the parity
difference is now non-zero for exactly this case.

### 3. `show_results` blocks, double-bills, and breaks for any `top_k != 3`

**`agents/multimodal_data_agent.py`**

Three separate problems in one shipped method:

- it ends in `plt.show()`, which blocks on a GUI window — fatal for any
  automated run
- it re-called `__load_image_from_blob` for each top match, re-downloading images
  and re-running (and re-billing) Content Safety on images analysed seconds earlier
- `plt.subplot(1, 4, i)` hard-codes a four-panel grid, so it raises for any
  `top_k` other than 3

**Fix:** optional `save_path` writes a PNG instead of blocking; a per-query image
cache means each blob is downloaded and screened exactly once; the grid is
computed from `len(matches)`.

### 4. `_run_multimodal` ignores the user's prompt

**`chat.py`**

It hard-coded `query-house-2.jpg` regardless of what was asked — its own docstring
said so ("your original code ignored the prompt … Keeping that behavior here for
parity"). Two different multimodal questions produced byte-identical answers.

**Fix:** the query image is selected from the prompt when one is named; the
default is unchanged.

### 5. `ask()` return shapes contradict their caller

**`agents/unstructured_data_agent.py`**

`chat.py` reads `result.get("response", str(result))`, but the early return for a
no-hits query used only `{"answer", "sources"}`. That branch would have printed a
raw Python dict to the user instead of the sentence.

**Fix:** `"response"` added alongside the shipped keys — nothing removed. All four
agents now return a dict containing `"response"`.

### 6. Content Safety rejects one of the twelve house images

**`agents/multimodal_data_agent.py`**

`274_Maplewood_Crescent.jpg` is 2268×1676; the service limit is 2048px. The
rejection surfaced inside `find_similar`'s `try/except` as a confusing
`Skipping: …` line, silently dropping one of the twelve houses from the candidate
set on every query.

**Fix:** `__downscale_for_safety` resizes only for the safety call, so CLIP still
ranks on full-resolution pixels and `data/multimodal/` stays byte-identical to the
course original.

### 7. `AnalyzerEngine` was constructed on every query

**`agents/unstructured_data_agent.py`, `__contains_pii`**

Presidio loads the spaCy `en_core_web_lg` pipeline on construction — several
seconds and a few hundred MB, paid on every single query.

**Fix:** a module-level lazily-initialised engine.

---

## Dependency changes

### 8. `PyPDF2` → `pypdf`, and the missing PDF library

`data/unstructured_load_azure.py` imports a PDF reader, but the course
`requirements.txt` lists **no PDF library at all** — the loader failed on import as
delivered. `pypdf` is the maintained successor to `PyPDF2` with the same
`PdfReader` API.

### 9. spaCy model pinned by URL

`presidio-analyzer` requires `en_core_web_lg`, which is published as a GitHub
release rather than on PyPI. Without it `AnalyzerEngine()` raises `OSError [E050]`
on the first PII audit — at demo time, not install time. Pinned by wheel URL so a
clean-room install is reproducible.

### 10. Version pins in `constraints.txt`

`requirements.txt` pins nothing. Four constraints carry real risk; see that file
for the reasoning on each. The load-bearing one is
`azure-ai-contentsafety==1.0.0`: in `1.0.0b1`, `ImageData.content` was typed `str`
and required manual base64 encoding, so the shipped
`ImageData(content=image_bytes)` would fail at runtime.

---

## Environment and authentication

### 11. `DeviceCodeCredential` behind `AzureCliCredential`

`chat.py`'s TODO comment said "Interactive Browser Credential", but line 5 already
imported `DeviceCodeCredential`, both load scripts use it, and the official project
instructions say to use it. The import and the instructions win.

It is wrapped as `ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())`
so that a non-interactive run (the evaluation harness, CI) reuses an existing
`az login` instead of blocking forever on a device-code prompt nobody can answer.
An interactive user with no CLI session still gets the documented device-code flow.

### 12. `keyVaultName` reads from the environment

`keyVaultName = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")`. The vault name is
a resource identifier, not a secret — it grants nothing on its own and every read
is gated by Entra ID — but sourcing it from the environment lets the same code run
against a differently-named vault without an edit.

### 13. PostgreSQL URI: driver and TLS

The URI is `postgresql+psycopg://…?sslmode=require`. Both parts are mandatory and
neither is obvious: `requirements.txt` ships `psycopg[binary]` (v3), so a bare
`postgresql://` scheme makes SQLAlchemy reach for psycopg2 and raise
`ModuleNotFoundError`; and Azure Flexible Server runs with
`require_secure_transport=ON`, so a connection without `sslmode=require` is refused.

---

## Additions (nothing replaced)

- **`agents/timeseries_data_agent.py`** — fourth agent over Azure Table Storage,
  with anomaly detection as its reliability guardrail.
- **`audit.py`** — per-query JSONL audit trail.
- **`router_llm.py`** — LLM query classifier with the keyword router as fallback.
- **`check_query_image`** — screens the user's own image before the similarity
  sweep. It was previously the only input in the system never checked for safety.
- **Actionable safeguards** — PII redacted from responses, flagged images excluded
  from results, bias findings prepended as a caveat, unreliable sensor windows
  withheld. The shipped checks printed warnings and returned the answer anyway.
- **`data/timeseries/`** — new folder for the fourth agent's synthetic data. The
  three course-provided folders are unchanged, verified by SHA-256 in
  `test_R4_6_provided_data_folders_are_unchanged`.
- **`scripts/`, `tests/`, `evaluate.py`, `verify_deliverables.py`, `docs/`.**

## Not changed, deliberately

- The `var = retrieved_secret = client.get_secret(...)` idiom in `chat.py`. It is
  odd but it is given code, and rewriting it would add diff noise for no gain.
- The shipped banner, print statements, and their typos ("Strucutred",
  "contente"). They are cosmetic and changing them would obscure the real diff.
- The bias audit reading every table on every query. Inefficient at scale, correct
  here, and the loop is shipped code.
