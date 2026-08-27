# Azure setup

Everything here is automated by `scripts/provision_azure.sh`. This document
explains what each step creates and **why**, so you can reproduce it by hand in
the portal, adapt it to a different subscription, or debug it when a step fails.

```bash
az login --use-device-code        # once, interactively
./scripts/provision_azure.sh      # create resources + write 16 secrets
./scripts/seed_data.sh            # load all four data stores
.venv/bin/python scripts/verify_azure.py   # 9 probes, all must pass
```

The script is idempotent: existing resources are reused, never recreated, so it is
safe to re-run after a partial failure.

---

## What gets created

| Resource | Name | SKU | Why this one |
|---|---|---|---|
| AI Foundry (existing) | `aifoundry-agentic-cd14705` | S0 | Reused. Deployments are per-account, so adding two models costs nothing extra. |
| ├ chat deployment | `gpt-4.1-mini` | GlobalStandard, 100K TPM | Orchestration, RAG summarisation, LLM routing |
| └ embedding deployment | `text-embedding-ada-002` | GlobalStandard, 120K TPM | 1536-dim vectors for the ChromaDB index |
| Content Safety | `cs-nbhd-4hbr` | F0 (free) | Image moderation. F0 allows 5 RPS / 5,000 images per month — ample here. |
| PostgreSQL Flexible Server | `pg-nbhd-4hbr` | Burstable B1ms, 32 GB, PG 16 | Structured demographics. Smallest tier that runs the workload. |
| Cosmos DB for MongoDB | `cosmos-nbhd-4hbr` | Serverless, Mongo 4.2 | Permit documents. Serverless bills per request — right for 21 documents. |
| Storage (images) | `stnbhdimg4hbr` | Standard_LRS | Blob container `houses` |
| Storage (sensors) | `stnbhdsensors4hbr` | Standard_LRS | Table `SensorReadings` |
| Key Vault | `kv-nbhd-4hbr` | RBAC mode | All 16 secrets |

**Two storage accounts, deliberately.** The multimodal and time-series agents
could share one account, but then they would share a credential and the
per-source isolation the rubric checks would be nominal rather than real. Two
accounts cost the same and make the property verifiable — probe A9 asserts a
credential for one cannot read the other.

**Two Azure OpenAI keys, deliberately.** The structured agent gets `key1`, the
unstructured agent `key2`, on the same account. Free, and it makes "each agent
holds its own independently rotatable credential" demonstrable.

---

## Step by step

### 1. Sign in

```bash
az login --use-device-code
```

Device code rather than browser: the lab account enforces MFA via a Temporary
Access Pass, so username/password sign-in is refused outright.

### 2. Model deployments

```bash
az cognitiveservices account deployment create \
  -n aifoundry-agentic-cd14705 -g Regroup_4hbr \
  --deployment-name gpt-4.1-mini \
  --model-name gpt-4.1-mini --model-version 2025-04-14 --model-format OpenAI \
  --sku-name GlobalStandard --sku-capacity 100
```

**The deployment name must equal the model name.** `chat.py` hardcodes
`deployment="gpt-4.1-mini"` and `embedding_deployment="text-embedding-ada-002"`.

**On quota:** the Cognitive Services usages API is blocked for this lab's custom
role — it returns `BadGateway: Response from service 'Microsoft.CognitiveServices'
does not contain sufficient information to enforce access control policy`. Quota
therefore cannot be read in advance. The script instead walks a descending
capacity ladder (100 → 50 → 30 → 10) and takes the highest the service accepts;
the attempt *is* the probe. Capacity matters here: a ReAct SQL agent resends its
whole message history every turn, so token use compounds and 10K TPM invites 429s.

### 3. Content Safety

```bash
az cognitiveservices account create -n cs-nbhd-4hbr -g Regroup_4hbr -l westus \
  --kind ContentSafety --sku F0 --custom-domain cs-nbhd-4hbr --yes
```

`--custom-domain` guarantees the endpoint is
`https://cs-nbhd-4hbr.cognitiveservices.azure.com/`, which is the form
`ContentSafetyClient(endpoint=…)` expects. F0 is limited to one instance per
subscription; the script falls back to S0 automatically.

### 4. PostgreSQL

```bash
MY_IP=$(curl -4 -sS https://ifconfig.me/ip)
az postgres flexible-server create -n pg-nbhd-4hbr -g Regroup_4hbr -l westus \
  --admin-user nbhdadmin --admin-password "$PGPASS" \
  --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 \
  --public-access "$MY_IP" --yes
az postgres flexible-server db create -g Regroup_4hbr -s pg-nbhd-4hbr -n neighborhoods
```

Four things worth knowing, each of which cost time to discover:

- **`--database-name` is rejected** on a non-elastic server ("can only be used
  when --node-count is present"). Create the database separately, and the flag
  there is `-n`, not `-d`.
- **The firewall flags are `-s` (server) and `-n` (rule name).** Passing the
  server name to `-n` fails with "unrecognized arguments", which a `|| true` will
  happily hide.
- **Different IP-echo services can disagree.** On a proxied network,
  `api.ipify.org` reported `177.39.124.219` while traffic to Azure actually
  arrived from `147.161.129.5`. The firewall rule looked correct and `psql` timed
  out anyway. The script now allows every address the echo services report and
  probes port 5432 afterwards to prove the rule works.
- **TLS is mandatory.** Flexible Server runs with `require_secure_transport=ON`,
  so the connection URI needs `?sslmode=require`.

The generated admin password is written to `solution/.pg_password` (git-ignored)
because it is otherwise unrecoverable; it is also stored in Key Vault, which is
what the application reads.

### 5. Cosmos DB for MongoDB

```bash
az cosmosdb create -n cosmos-nbhd-4hbr -g Regroup_4hbr \
  --kind MongoDB --server-version 4.2 --capabilities EnableServerless \
  --locations regionName=westus failoverPriority=0
az cosmosdb mongodb database create -g Regroup_4hbr -a cosmos-nbhd-4hbr -n permits
```

**`retrywrites=false` is required.** pymongo 4.x sends `retryWrites=true` by
default and Cosmos DB's RU-based Mongo API rejects it with "Retryable writes are
not supported", which breaks the PDF loader's `insert_one`. Azure normally
includes the parameter in the connection string it hands you; the script asserts
it rather than assuming.

**The collection is created by the loader, not by ARM.** An ARM-created Cosmos
Mongo collection requires a shard key; letting `db.create_collection()` make it
yields an unsharded fixed collection, which is correct at this size.

### 6. Storage and Key Vault

```bash
az storage account create -n stnbhdimg4hbr -g Regroup_4hbr -l westus \
  --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2
az storage container create --name houses --connection-string "$CONN"

az keyvault create -n kv-nbhd-4hbr -g Regroup_4hbr -l westus \
  --enable-rbac-authorization true --retention-days 7
az role assignment create --role "Key Vault Secrets Officer" \
  --assignee-object-id "$USER_OBJECT_ID" --assignee-principal-type User \
  --scope "$VAULT_ID"
```

**Do not pass `--enable-purge-protection false`.** Azure rejects it outright
("cannot be set to false … irreversible action"). Leaving it unset is already off,
which is what teardown needs.

**RBAC propagation is eventually consistent.** A secret write immediately after
the role assignment reliably 403s. The script polls with a probe secret until the
vault accepts a write (typically ~10s). If the role assignment itself is denied,
recreate the vault with `--enable-rbac-authorization false`, which grants the
creating principal full secret permissions synchronously.

### 7. The 16 secrets

| Secret | Value | Consumer |
|---|---|---|
| `structuredazureendpoint` | AI Foundry endpoint | structured |
| `structuredazureapikey` | AI Foundry **key1** | structured |
| `structuredpostgresqlhost` | PG FQDN | structured |
| `structuredpostgresqldbname` | `neighborhoods` | structured |
| `structuredpostgresqluser` | `nbhdadmin` | structured |
| `structuredpostgresqlpassword` | generated, 30 chars | structured |
| `unstructuredmongourl` | Cosmos Mongo connection string | unstructured |
| `unstructureddbname` | `permits` | unstructured |
| `unstructuredcollectionname` | `permit_documents` | unstructured |
| `unstructuredazureendpoint` | AI Foundry endpoint | unstructured |
| `unstructuredazurekey` | AI Foundry **key2** | unstructured |
| `multimodalazureconnstring` | `stnbhdimg4hbr` connection string | multimodal |
| `multimodalazurecontentsafetyendpoint` | Content Safety endpoint | multimodal |
| `multimodalazurecontentsafetykey` | Content Safety key1 | multimodal |
| `timeseriesazureconnstring` | `stnbhdsensors4hbr` connection string | time-series |
| `timeseriestablename` | `SensorReadings` | time-series |

The first seven names are **contracts**: they are hardcoded in the course's own
load scripts, which cannot be renamed without editing given code.
`test_R4_4_loader_secret_names_are_a_subset_of_the_vault` asserts the loaders only
ask for names that provisioning actually creates.

### 8. Seed the data

```bash
./scripts/seed_data.sh              # skips stores that already hold data
FORCE_RELOAD=1 ./scripts/seed_data.sh   # drop and reload everything
```

**The guards are not optional.** The course loaders use
`CREATE TABLE IF NOT EXISTS` followed by an unconditional `INSERT`, and the Mongo
loader inserts every PDF unconditionally. Running either twice duplicates every
row — with no error — and silently doubles every group in the Fairlearn audit. The
wrapper checks row counts before loading and asserts them afterwards, because the
loader exits 0 even when its inserts never landed.

`FORCE_RELOAD` on MongoDB also deletes `data/chroma_db_storage/`: reloading mints
new ObjectIds, and Chroma upserts by id without removing stale entries, so old
vectors would point at documents that no longer exist and every RAG query would
answer "No relevant documents found."

---

## Verification

`scripts/verify_azure.py` runs nine probes using the same SDK calls, with the same
arguments, that the agents use — so a pass means the agents will connect, not just
that the resources exist.

| Probe | Asserts |
|---|---|
| A1 | `gpt-4.1-mini` answers a chat completion |
| A2 | `text-embedding-ada-002` returns a 1536-dim vector **at api-version `2023-06-01-preview`** |
| A3 | All 16 secrets readable |
| A4 | PostgreSQL: 5540 / 2580 / 10 / 6 rows — exact counts catch a double-load |
| A5 | MongoDB: exactly 21 documents |
| A6 | Blob: exactly 12 images |
| A7 | Content Safety analyses `274_Maplewood_Crescent.jpg` — the one over the size limit |
| A8 | Table Storage: 1680 readings across 5 neighborhoods |
| A9 | Isolation: the sensor credential cannot read the image container |

A2 deserves a note. `chat.py` hardcodes `azure_api_version="2023-06-01-preview"`
for the unstructured agent, and Azure has been retiring pre-2024 preview
data-plane versions. If that version were gone, both embeddings and the RAG chat
would be dead. The probe checks it explicitly rather than letting it surface
mid-demo. **As of this run it returns HTTP 200**, so no code change was needed.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `psycopg.OperationalError: connection timed out` | Firewall allows the wrong IP — echo services disagree behind a proxy | `curl -4 https://ifconfig.me/ip`, add that address as a firewall rule |
| `ModuleNotFoundError: psycopg2` | URI used `postgresql://` instead of `postgresql+psycopg://` | `requirements.txt` installs psycopg **3** |
| `connection is insecure` from PostgreSQL | Missing `?sslmode=require` | Flexible Server sets `require_secure_transport=ON` |
| `Retryable writes are not supported` | pymongo default vs Cosmos RU | Ensure `retrywrites=false` in the connection string |
| `OSError [E050] Can't find model 'en_core_web_lg'` | spaCy model missing | It is pinned by URL in `requirements.txt`; re-run the install |
| `CERTIFICATE_VERIFY_FAILED` downloading CLIP | TLS-intercepting proxy; its CA is in the OS keychain but not in certifi | `tls_trust.py` routes verification to the OS trust store; it is called from `chat.py` |
| `InvalidImage` from Content Safety | Image over 2048px or 4 MB | The agent downscales for the safety call only |
| `403` writing a Key Vault secret | RBAC not yet propagated | Wait ~30s; the script polls automatically |
| Deployment refused with a quota error | Capacity unavailable at that tier | The ladder retries lower; or delete the unused `gpt-4.1` deployment to free TPM |

---

## Teardown

The lab subscription accrues cost — PostgreSQL B1ms runs continuously.

```bash
./scripts/teardown_azure.sh
```

This deletes the PostgreSQL server, Cosmos account, both storage accounts, the
Content Safety resource, and the Key Vault (then purges it so the name is
reusable). It leaves `aifoundry-agentic-cd14705` alone, since it predates this
project.
