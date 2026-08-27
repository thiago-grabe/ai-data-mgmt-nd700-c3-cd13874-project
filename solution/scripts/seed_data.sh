#!/usr/bin/env bash
# Seed all four data stores. Run AFTER scripts/provision_azure.sh.
#
#     ./scripts/seed_data.sh              # skip stores that already hold data
#     FORCE_RELOAD=1 ./scripts/seed_data.sh   # drop and reload everything
#
# Why the guards matter: the course loaders use CREATE TABLE IF NOT EXISTS
# followed by an unconditional INSERT, and the Mongo loader inserts every PDF
# unconditionally. Running either twice silently duplicates every row — which
# would skew the Fairlearn audit with no error and no visible symptom.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$SOLUTION_DIR/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
FORCE_RELOAD="${FORCE_RELOAD:-0}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m    OK  %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

cd "$SOLUTION_DIR/data"

# ------------------------------------------------------------------ sensors CSV
step "Generating the synthetic sensor dataset"
"$PY" timeseries/generate_sensors.py

# --------------------------------------------------------------- blob images
step "Uploading house images to Blob Storage"
"$PY" - <<'PY'
import os
from pathlib import Path

from azure.identity import AzureCliCredential, ChainedTokenCredential, DeviceCodeCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient

vault = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
kv = SecretClient(
    vault_url=f"https://{vault}.vault.azure.net/",
    credential=ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential()),
)
conn = kv.get_secret("multimodalazureconnstring").value

container = BlobServiceClient.from_connection_string(conn).get_container_client("houses")

# The originals are uploaded unmodified: data/multimodal/ is course-provided and
# stays byte-identical. The one image that exceeds Azure Content Safety's 2048px
# limit is downscaled inside the agent, for the safety call only, so CLIP still
# compares full-resolution pixels.
local = sorted(Path("multimodal").glob("*.jpg"))
existing = {b.name for b in container.list_blobs()}

for path in local:
    if path.name in existing and os.environ.get("FORCE_RELOAD", "0") != "1":
        continue
    with path.open("rb") as fh:
        container.upload_blob(name=path.name, data=fh, overwrite=True)
    print(f"  uploaded {path.name}")

count = len(list(container.list_blobs()))
print(f"Container 'houses' now holds {count} blobs (expected 12).")
assert count == 12, f"expected 12 blobs, found {count}"
PY
ok "12 images in blob storage"

# ----------------------------------------------------------------- postgresql
step "Loading demographics into PostgreSQL"
EXISTING_ROWS=$("$PY" - <<'PY'
import os
import psycopg
from azure.identity import AzureCliCredential, ChainedTokenCredential, DeviceCodeCredential
from azure.keyvault.secrets import SecretClient

vault = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
kv = SecretClient(
    vault_url=f"https://{vault}.vault.azure.net/",
    credential=ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential()),
)
cfg = {
    "host": kv.get_secret("structuredpostgresqlhost").value,
    "dbname": kv.get_secret("structuredpostgresqldbname").value,
    "user": kv.get_secret("structuredpostgresqluser").value,
    "password": kv.get_secret("structuredpostgresqlpassword").value,
    "port": 5432,
    "sslmode": "require",
}
try:
    with psycopg.connect(**cfg) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM neighborhood_houses")
        print(cur.fetchone()[0])
except Exception:
    print(0)
PY
)
info "neighborhood_houses currently has $EXISTING_ROWS rows"

if [[ "$EXISTING_ROWS" -gt 0 && "$FORCE_RELOAD" != "1" ]]; then
  ok "already loaded; skipping (FORCE_RELOAD=1 to drop and reload)"
else
  if [[ "$FORCE_RELOAD" == "1" ]]; then
    info "dropping existing tables"
    "$PY" - <<'PY'
import os
import psycopg
from azure.identity import AzureCliCredential, ChainedTokenCredential, DeviceCodeCredential
from azure.keyvault.secrets import SecretClient

vault = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
kv = SecretClient(
    vault_url=f"https://{vault}.vault.azure.net/",
    credential=ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential()),
)
cfg = {
    "host": kv.get_secret("structuredpostgresqlhost").value,
    "dbname": kv.get_secret("structuredpostgresqldbname").value,
    "user": kv.get_secret("structuredpostgresqluser").value,
    "password": kv.get_secret("structuredpostgresqlpassword").value,
    "port": 5432,
    "sslmode": "require",
}
with psycopg.connect(**cfg) as conn, conn.cursor() as cur:
    cur.execute(
        "DROP TABLE IF EXISTS neighborhood_houses, education_survey, "
        "neighborhood_hospitals, neighborhood_schools"
    )
    conn.commit()
print("dropped")
PY
  fi
  "$PY" structured_load_azure.py
  # The loader exits 0 even when its inserts never landed — a transient connection
  # failure once left this step "successful" with zero tables created, and the
  # problem only surfaced two stores later. Assert the end state.
  "$PY" - <<'PY'
import os
import sys

import psycopg
from azure.identity import AzureCliCredential, ChainedTokenCredential, DeviceCodeCredential
from azure.keyvault.secrets import SecretClient

vault = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
kv = SecretClient(
    vault_url=f"https://{vault}.vault.azure.net/",
    credential=ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential()),
)
cfg = {
    "host": kv.get_secret("structuredpostgresqlhost").value,
    "dbname": kv.get_secret("structuredpostgresqldbname").value,
    "user": kv.get_secret("structuredpostgresqluser").value,
    "password": kv.get_secret("structuredpostgresqlpassword").value,
    "port": 5432,
    "sslmode": "require",
}
expected = {
    "neighborhood_houses": 5540,
    "education_survey": 2580,
    "neighborhood_schools": 10,
    "neighborhood_hospitals": 6,
}
with psycopg.connect(**cfg) as conn, conn.cursor() as cur:
    bad = {}
    for table, want in expected.items():
        cur.execute(f"SELECT count(*) FROM {table}")
        got = cur.fetchone()[0]
        if got != want:
            bad[table] = (got, want)
if bad:
    print(f"row count mismatch (got, expected): {bad}", file=sys.stderr)
    sys.exit(1)
print("verified: " + ", ".join(f"{t}={c}" for t, c in expected.items()))
PY
  ok "demographics loaded and row counts verified"
fi

# -------------------------------------------------------------------- mongodb
step "Loading permit PDFs into MongoDB"
MONGO_COUNT=$("$PY" - <<'PY'
import os
from azure.identity import AzureCliCredential, ChainedTokenCredential, DeviceCodeCredential
from azure.keyvault.secrets import SecretClient
from pymongo import MongoClient

vault = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
kv = SecretClient(
    vault_url=f"https://{vault}.vault.azure.net/",
    credential=ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential()),
)
try:
    client = MongoClient(kv.get_secret("unstructuredmongourl").value)
    coll = client[kv.get_secret("unstructureddbname").value][
        kv.get_secret("unstructuredcollectionname").value
    ]
    print(coll.count_documents({}))
except Exception:
    print(0)
PY
)
info "collection currently holds $MONGO_COUNT documents"

if [[ "$MONGO_COUNT" -gt 0 && "$FORCE_RELOAD" != "1" ]]; then
  ok "already loaded; skipping"
else
  if [[ "$FORCE_RELOAD" == "1" ]]; then
    info "clearing collection"
    "$PY" - <<'PY'
import os
from azure.identity import AzureCliCredential, ChainedTokenCredential, DeviceCodeCredential
from azure.keyvault.secrets import SecretClient
from pymongo import MongoClient

vault = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
kv = SecretClient(
    vault_url=f"https://{vault}.vault.azure.net/",
    credential=ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential()),
)
client = MongoClient(kv.get_secret("unstructuredmongourl").value)
coll = client[kv.get_secret("unstructureddbname").value][
    kv.get_secret("unstructuredcollectionname").value
]
coll.delete_many({})
print("cleared")
PY
    # Reloading Mongo mints new ObjectIds. Chroma upserts by id and never removes,
    # so stale vectors would keep pointing at documents that no longer exist and
    # every RAG query would silently answer "No relevant documents found."
    info "removing the stale Chroma index (ObjectIds have changed)"
    rm -rf "$SOLUTION_DIR/data/chroma_db_storage"
  fi
  "$PY" unstructured_load_azure.py
  ok "permit documents loaded"
fi

# --------------------------------------------------------------- table storage
step "Loading sensor readings into Table Storage"
"$PY" timeseries_load_azure.py
ok "sensor readings loaded"

step "Seeding complete"
echo "Next: python scripts/verify_azure.py"
