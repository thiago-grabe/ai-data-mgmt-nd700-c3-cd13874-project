#!/usr/bin/env bash
# Provision every Azure resource the Ethical Multi-Agent Data Orchestrator needs,
# then write all 16 secrets into Key Vault.
#
# Prerequisite (interactive, do this first):
#     az login --use-device-code
#
# Usage:
#     ./scripts/provision_azure.sh
#
# Idempotent: existing resources are reused, never recreated. Safe to re-run after
# a partial failure. See docs/AZURE_SETUP.md for what each step does and why.
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------- configuration
RG="${RESOURCE_GROUP:-Regroup_4hbr}"
LOCATION="${LOCATION:-westus}"

# Reuse the AI Foundry account that already serves gpt-4.1 from the previous
# project. Deployments are per-account, so adding two more costs nothing extra.
FOUNDRY="${FOUNDRY_NAME:-aifoundry-agentic-cd14705}"
CHAT_DEPLOYMENT="gpt-4.1-mini"
CHAT_MODEL="gpt-4.1-mini"
EMBED_DEPLOYMENT="text-embedding-ada-002"
EMBED_MODEL="text-embedding-ada-002"
# Quota is unreadable for this role (the usages API 502s), so capacity is
# discovered by attempting the deployment. Descending order: take the most
# capacity the service will give us. A ReAct SQL agent resends its whole message
# history every turn, so tokens compound fast — the previous project in this
# course needed 100K TPM after hitting a 429 at 10K.
CHAT_CAPACITY_LADDER=(100 50 30 10)
EMBED_CAPACITY_LADDER=(120 50 30 10)

KEYVAULT="${KEYVAULT_NAME:-kv-nbhd-4hbr}"

PG_SERVER="${PG_SERVER_NAME:-pg-nbhd-4hbr}"
PG_ADMIN="nbhdadmin"
PG_DB="neighborhoods"

COSMOS="${COSMOS_NAME:-cosmos-nbhd-4hbr}"
MONGO_DB="permits"
MONGO_COLLECTION="permit_documents"

ST_IMAGES="${ST_IMAGES_NAME:-stnbhdimg4hbr}"
IMAGES_CONTAINER="houses"

ST_SENSORS="${ST_SENSORS_NAME:-stnbhdsensors4hbr}"
SENSOR_TABLE="SensorReadings"

CONTENT_SAFETY="${CONTENT_SAFETY_NAME:-cs-nbhd-4hbr}"

# ---------------------------------------------------------------------- helpers
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '\033[32m    OK  %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    !!  %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ preflight
step "Verifying sign-in"
az account show -o none 2>/dev/null || die "Not signed in. Run: az login --use-device-code"
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
USER_OBJECT_ID=$(az ad signed-in-user show --query id -o tsv)
az account show --query "{subscription:name, user:user.name}" -o table
info "objectId: $USER_OBJECT_ID"

az group show -n "$RG" -o none 2>/dev/null \
  || die "Resource group $RG not found. Lab subscriptions restrict creating new ones."
ok "resource group $RG"

step "Checking resource providers"
# Provider registration is a SUBSCRIPTION-scope action. The lab's custom role is
# scoped to the resource group, so `az provider register` returns
# AuthorizationFailed here — there is nothing to wait for. Microsoft.Network is
# NotRegistered on this subscription; PostgreSQL Flexible Server in public-access
# mode does not create a VNet or private DNS zone, so it should not need it. If
# creation later fails citing Microsoft.Network, the lab operator must register
# it at subscription scope.
for p in Microsoft.DBforPostgreSQL Microsoft.DocumentDB Microsoft.Storage \
         Microsoft.KeyVault Microsoft.CognitiveServices; do
  state=$(az provider show -n "$p" --query registrationState -o tsv 2>/dev/null || echo Unknown)
  if [[ "$state" == "Registered" ]]; then
    info "$p: Registered"
  else
    warn "$p: $state — attempting registration"
    az provider register -n "$p" -o none 2>/dev/null \
      || warn "cannot register $p (subscription-scope action denied for this role)"
  fi
done
net_state=$(az provider show -n Microsoft.Network --query registrationState -o tsv 2>/dev/null || echo Unknown)
[[ "$net_state" == "Registered" ]] \
  && info "Microsoft.Network: Registered" \
  || warn "Microsoft.Network: $net_state — expected on this lab subscription; public-access PostgreSQL does not require it"

# ------------------------------------------------------------- model deployments
step "Deploying models on $FOUNDRY"
az cognitiveservices account show -n "$FOUNDRY" -g "$RG" -o none 2>/dev/null \
  || die "AI Foundry account $FOUNDRY not found in $RG."

deploy_model() {
  local dep="$1" model="$2"; shift 2
  local ladder=("$@")

  if az cognitiveservices account deployment show -n "$FOUNDRY" -g "$RG" \
       --deployment-name "$dep" -o none 2>/dev/null; then
    ok "deployment $dep already exists"
    return 0
  fi

  local version
  version=$(az cognitiveservices model list -l "$LOCATION" \
    --query "[?model.name=='${model}'] | [0].model.version" -o tsv 2>/dev/null || true)
  [[ -z "$version" || "$version" == "None" ]] \
    && die "$model is not offered in $LOCATION."
  info "$model version $version"

  # The quota API returns 502 for this role, so capacity cannot be read up front.
  # Climb the ladder until the service stops rejecting us.
  for cap in "${ladder[@]}"; do
    info "trying capacity ${cap}K TPM"
    if az cognitiveservices account deployment create \
         -n "$FOUNDRY" -g "$RG" --deployment-name "$dep" \
         --model-name "$model" --model-version "$version" --model-format OpenAI \
         --sku-name GlobalStandard --sku-capacity "$cap" -o none 2>/tmp/deploy_err.txt; then
      ok "deployed $dep at ${cap}K TPM"
      return 0
    fi
    if grep -qiE "quota|capacity|exceed" /tmp/deploy_err.txt; then
      warn "capacity ${cap}K refused, trying higher"
      continue
    fi
    cat /tmp/deploy_err.txt >&2
    die "deployment of $dep failed for a reason other than quota"
  done
  die "no capacity worked for $dep"
}

deploy_model "$CHAT_DEPLOYMENT"  "$CHAT_MODEL"  "${CHAT_CAPACITY_LADDER[@]}"
deploy_model "$EMBED_DEPLOYMENT" "$EMBED_MODEL" "${EMBED_CAPACITY_LADDER[@]}"

FOUNDRY_ENDPOINT=$(az cognitiveservices account show -n "$FOUNDRY" -g "$RG" \
  --query properties.endpoint -o tsv)
# Two keys on the same account: the structured agent gets key1, the unstructured
# agent key2. Costs nothing and makes "each agent holds its own independently
# rotatable credential" a demonstrable property rather than a claim.
FOUNDRY_KEY=$(az cognitiveservices account keys list -n "$FOUNDRY" -g "$RG" --query key1 -o tsv)
FOUNDRY_KEY2=$(az cognitiveservices account keys list -n "$FOUNDRY" -g "$RG" --query key2 -o tsv)
ok "endpoint $FOUNDRY_ENDPOINT"

# ---------------------------------------------------------------- content safety
step "Creating Content Safety resource $CONTENT_SAFETY"
if az cognitiveservices account show -n "$CONTENT_SAFETY" -g "$RG" -o none 2>/dev/null; then
  ok "already exists"
else
  # F0 is free but limited to one per subscription; fall back to pay-as-you-go S0.
  if az cognitiveservices account create -n "$CONTENT_SAFETY" -g "$RG" -l "$LOCATION" \
       --kind ContentSafety --sku F0 --custom-domain "$CONTENT_SAFETY" --yes -o none 2>/dev/null; then
    ok "created (F0 free tier)"
  else
    warn "F0 unavailable, falling back to S0"
    az cognitiveservices account create -n "$CONTENT_SAFETY" -g "$RG" -l "$LOCATION" \
      --kind ContentSafety --sku S0 --custom-domain "$CONTENT_SAFETY" --yes -o none \
      || die "could not create Content Safety resource"
    ok "created (S0)"
  fi
fi
CS_ENDPOINT=$(az cognitiveservices account show -n "$CONTENT_SAFETY" -g "$RG" \
  --query properties.endpoint -o tsv)
CS_KEY=$(az cognitiveservices account keys list -n "$CONTENT_SAFETY" -g "$RG" \
  --query key1 -o tsv)

# -------------------------------------------------------------------- postgresql
step "Creating PostgreSQL Flexible Server $PG_SERVER"
PG_PASSWORD_FILE="$SOLUTION_DIR/.pg_password"
if az postgres flexible-server show -n "$PG_SERVER" -g "$RG" -o none 2>/dev/null; then
  ok "already exists"
  [[ -f "$PG_PASSWORD_FILE" ]] \
    || die "server exists but $PG_PASSWORD_FILE is missing; cannot recover the admin password. Reset it with 'az postgres flexible-server update --admin-password'."
  PG_PASSWORD=$(cat "$PG_PASSWORD_FILE")
else
  # Alphanumeric only: the password travels through a SQLAlchemy URI and a
  # connection string, and quote_plus handling differs between the two.
  # Generated in Python rather than `tr </dev/urandom | head -c`: that idiom
  # SIGPIPEs `tr` when `head` exits, which under `set -o pipefail` returns 141
  # and kills the script.
  PG_PASSWORD="Nb$(python3 -c 'import secrets,string; print("".join(secrets.choice(string.ascii_letters+string.digits) for _ in range(28)))')"
  umask 077
  printf '%s' "$PG_PASSWORD" > "$PG_PASSWORD_FILE"
  MY_IP=$(curl -sS https://api.ipify.org)
  # --database-name is rejected on non-elastic servers ("can only be used when
  # --node-count is present"), so the database is created separately below.
  # --public-access <ip> both enables public networking and seeds the first
  # firewall rule in one call.
  az postgres flexible-server create \
    --name "$PG_SERVER" --resource-group "$RG" --location "$LOCATION" \
    --admin-user "$PG_ADMIN" --admin-password "$PG_PASSWORD" \
    --tier Burstable --sku-name Standard_B1ms --storage-size 32 \
    --version 16 \
    --public-access "$MY_IP" --yes -o none \
    || die "PostgreSQL creation failed"
  ok "created"
fi

info "ensuring database $PG_DB exists"
# The flag is -n/--name, not -d. An earlier version used -d and the `|| info`
# fallback swallowed the resulting argument error, reporting "already exists" for
# a database that was never created — the seed step then failed much later with a
# confusing connection error. Verify the end state instead of trusting the call.
if ! az postgres flexible-server db list -g "$RG" -s "$PG_SERVER" --query "[].name" -o tsv | grep -qx "$PG_DB"; then
  az postgres flexible-server db create -g "$RG" -s "$PG_SERVER" -n "$PG_DB" -o none \
    || die "could not create database $PG_DB"
fi
az postgres flexible-server db list -g "$RG" -s "$PG_SERVER" --query "[].name" -o tsv | grep -qx "$PG_DB" \
  || die "database $PG_DB still missing after create"
ok "database $PG_DB exists"

PG_HOST=$(az postgres flexible-server show -n "$PG_SERVER" -g "$RG" \
  --query fullyQualifiedDomainName -o tsv)

info "configuring firewall"
# Different IP-echo services can report different addresses when the network
# egresses through a proxy: api.ipify.org reported 177.39.124.219 here while
# traffic to Azure actually arrived from 147.161.129.5, so the rule allowed an
# address that never connects and psql timed out with the rule apparently in
# place. Allow every distinct address the echo services report.
CLIENT_IPS=$(
  for svc in https://ifconfig.me/ip https://checkip.amazonaws.com https://api.ipify.org; do
    curl -4 -sS --max-time 10 "$svc" 2>/dev/null | tr -d '[:space:]'
    echo
  done | grep -E '^[0-9]+(\.[0-9]+){3}$' | sort -u
)
[[ -n "$CLIENT_IPS" ]] || die "could not determine this machine's public IP"

# The flags are -s (server) and -n (RULE name). Passing the server to -n makes the
# call fail with "unrecognized arguments", which a `|| true` would hide.
idx=0
while read -r ip; do
  [[ -z "$ip" ]] && continue
  idx=$((idx + 1))
  az postgres flexible-server firewall-rule create -g "$RG" -s "$PG_SERVER" \
    -n "allow-client-$idx" --start-ip-address "$ip" --end-ip-address "$ip" -o none \
    || warn "could not add firewall rule for $ip"
  info "allowed $ip"
done <<< "$CLIENT_IPS"

az postgres flexible-server firewall-rule create -g "$RG" -s "$PG_SERVER" \
  -n allow-azure --start-ip-address 0.0.0.0 --end-ip-address 0.0.0.0 -o none \
  || warn "could not add the Azure-services firewall rule"

# Prove the rules actually work rather than assuming they do.
if command -v nc >/dev/null 2>&1; then
  nc -z -w 15 "$PG_HOST" 5432 2>/dev/null \
    && ok "port 5432 reachable" \
    || warn "port 5432 still unreachable — check egress IP or corporate firewall"
fi
ok "firewall configured"


# ----------------------------------------------------------------------- cosmos
step "Creating Cosmos DB for MongoDB $COSMOS"
if az cosmosdb show -n "$COSMOS" -g "$RG" -o none 2>/dev/null; then
  ok "already exists"
else
  az cosmosdb create --name "$COSMOS" --resource-group "$RG" \
    --kind MongoDB --server-version 4.2 \
    --capabilities EnableServerless \
    --locations regionName="$LOCATION" failoverPriority=0 isZoneRedundant=False \
    -o none || die "Cosmos DB creation failed"
  ok "created (serverless, Mongo 4.2)"
fi

# Create the database, but NOT the collection: an ARM-created Cosmos Mongo
# collection requires a shard key. Letting the loader's db.create_collection()
# make it yields an unsharded fixed collection, which is right for 21 documents.
az cosmosdb mongodb database create -g "$RG" -a "$COSMOS" -n "$MONGO_DB" -o none 2>/dev/null \
  || info "database $MONGO_DB already exists"

MONGO_URI=$(az cosmosdb keys list --name "$COSMOS" --resource-group "$RG" \
  --type connection-strings --query "connectionStrings[0].connectionString" -o tsv)
# pymongo 4.x sends retryWrites=true by default and Cosmos RU rejects it with
# "Retryable writes are not supported", which would break the PDF loader's
# insert_one. Azure normally includes retrywrites=false; assert rather than hope.
if ! grep -qi 'retrywrites=false' <<<"$MONGO_URI"; then
  warn "connection string lacked retrywrites=false; appending it"
  MONGO_URI="${MONGO_URI}&retrywrites=false"
fi
ok "mongo connection string has retrywrites=false"

# ------------------------------------------------------------- storage accounts
create_storage() {
  local name="$1" label="$2"
  if az storage account show -n "$name" -g "$RG" -o none 2>/dev/null; then
    ok "$label storage $name already exists"
  else
    az storage account create -n "$name" -g "$RG" -l "$LOCATION" \
      --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 -o none \
      || die "could not create storage account $name"
    ok "$label storage $name created"
  fi
}

step "Creating storage accounts"
# Two separate accounts, deliberately: the multimodal and time-series agents must
# not share a credential, or the per-source isolation the rubric checks collapses.
create_storage "$ST_IMAGES"  "images"
create_storage "$ST_SENSORS" "sensors"

ST_IMAGES_CONN=$(az storage account show-connection-string -n "$ST_IMAGES" -g "$RG" -o tsv)
ST_SENSORS_CONN=$(az storage account show-connection-string -n "$ST_SENSORS" -g "$RG" -o tsv)

az storage container create --name "$IMAGES_CONTAINER" \
  --connection-string "$ST_IMAGES_CONN" -o none
ok "container $IMAGES_CONTAINER ready"

az storage table create --name "$SENSOR_TABLE" \
  --connection-string "$ST_SENSORS_CONN" -o none
ok "table $SENSOR_TABLE ready"

# -------------------------------------------------------------------- key vault
step "Creating Key Vault $KEYVAULT"
if az keyvault show -n "$KEYVAULT" -g "$RG" -o none 2>/dev/null; then
  ok "already exists"
else
  # Purge protection is left unset rather than passed as false: Azure rejects
  # "--enable-purge-protection false" outright ("cannot be set to false ...
  # irreversible action"). Unset is already off, which is what teardown needs so
  # the vault can be purged and the name reused.
  az keyvault create --name "$KEYVAULT" --resource-group "$RG" --location "$LOCATION" \
    --enable-rbac-authorization true --retention-days 7 -o none \
    || die "Key Vault creation failed"
  ok "created (RBAC authorization mode)"
fi
VAULT_ID=$(az keyvault show -n "$KEYVAULT" -g "$RG" --query id -o tsv)

info "granting Key Vault Secrets Officer to the signed-in user"
az role assignment create --role "Key Vault Secrets Officer" \
  --assignee-object-id "$USER_OBJECT_ID" --assignee-principal-type User \
  --scope "$VAULT_ID" -o none 2>/dev/null || info "role assignment already present"

# RBAC propagation is eventually consistent; a write immediately after the grant
# reliably 403s. Poll until a probe secret sticks.
info "waiting for RBAC propagation"
for i in $(seq 1 30); do
  if az keyvault secret set --vault-name "$KEYVAULT" --name provisioningprobe \
       --value ok -o none 2>/dev/null; then
    ok "vault writable after $((i * 10))s"
    break
  fi
  [[ $i -eq 30 ]] && die "vault still not writable after 5 minutes; check the role assignment"
  sleep 10
done

# ----------------------------------------------------------------------- secrets
step "Writing 16 secrets"
set_secret() {
  az keyvault secret set --vault-name "$KEYVAULT" --name "$1" --value "$2" -o none \
    || die "could not write secret $1"
  info "$1"
}

# Structured agent -> PostgreSQL + Azure OpenAI
set_secret structuredazureendpoint            "$FOUNDRY_ENDPOINT"
set_secret structuredazureapikey              "$FOUNDRY_KEY"
set_secret structuredpostgresqlhost           "$PG_HOST"
set_secret structuredpostgresqldbname         "$PG_DB"
set_secret structuredpostgresqluser           "$PG_ADMIN"
set_secret structuredpostgresqlpassword       "$PG_PASSWORD"

# Unstructured agent -> MongoDB + Azure OpenAI
set_secret unstructuredmongourl               "$MONGO_URI"
set_secret unstructureddbname                 "$MONGO_DB"
set_secret unstructuredcollectionname         "$MONGO_COLLECTION"
set_secret unstructuredazureendpoint          "$FOUNDRY_ENDPOINT"
set_secret unstructuredazurekey               "$FOUNDRY_KEY2"

# Multimodal agent -> Blob Storage + Content Safety
set_secret multimodalazureconnstring          "$ST_IMAGES_CONN"
set_secret multimodalazurecontentsafetyendpoint "$CS_ENDPOINT"
set_secret multimodalazurecontentsafetykey    "$CS_KEY"

# Time-series agent -> Table Storage
set_secret timeseriesazureconnstring          "$ST_SENSORS_CONN"
set_secret timeseriestablename                "$SENSOR_TABLE"

az keyvault secret delete --vault-name "$KEYVAULT" --name provisioningprobe -o none 2>/dev/null || true

SECRET_COUNT=$(az keyvault secret list --vault-name "$KEYVAULT" --query "length(@)" -o tsv)
[[ "$SECRET_COUNT" == "16" ]] || die "expected 16 secrets in $KEYVAULT, found $SECRET_COUNT"
ok "16 secrets present"

# ------------------------------------------------------------- endpoint probes
step "Probing model endpoints"
# chat.py hardcodes azure_api_version="2023-06-01-preview" for the unstructured
# agent. Azure has been retiring pre-2024 preview data-plane versions, and if that
# one is gone both the embeddings and the RAG chat die. Find out now, with curl,
# rather than three hours from now in the middle of a live evaluation run.
probe() {
  curl -sS -o /tmp/probe_body.json -w '%{http_code}' -X POST \
    -H "api-key: $FOUNDRY_KEY" -H 'Content-Type: application/json' \
    -d "$2" "$1"
}

CHAT_BODY='{"messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
EMBED_BODY='{"input":"ping"}'
EP="${FOUNDRY_ENDPOINT%/}"

for API_VERSION in 2024-10-21 2023-06-01-preview; do
  code=$(probe "$EP/openai/deployments/$CHAT_DEPLOYMENT/chat/completions?api-version=$API_VERSION" "$CHAT_BODY")
  if [[ "$code" == "200" ]]; then
    ok "chat  @ $API_VERSION -> 200"
  else
    warn "chat  @ $API_VERSION -> $code"
    head -c 300 /tmp/probe_body.json; echo
    [[ "$API_VERSION" == "2023-06-01-preview" ]] \
      && warn "chat.py hardcodes this version for the unstructured agent — change that literal and record it in docs/DEVIATIONS.md"
  fi
done

for API_VERSION in 2024-10-21 2023-06-01-preview; do
  code=$(probe "$EP/openai/deployments/$EMBED_DEPLOYMENT/embeddings?api-version=$API_VERSION" "$EMBED_BODY")
  if [[ "$code" == "200" ]]; then
    dims=$(python3 -c "import json;print(len(json.load(open('/tmp/probe_body.json'))['data'][0]['embedding']))" 2>/dev/null || echo '?')
    ok "embed @ $API_VERSION -> 200 (dim $dims)"
  else
    warn "embed @ $API_VERSION -> $code"
    head -c 300 /tmp/probe_body.json; echo
  fi
done
rm -f /tmp/probe_body.json

# -------------------------------------------------------------------- summary
step "Provisioned"
cat <<EOF
Resource group   : $RG ($LOCATION)
AI Foundry       : $FOUNDRY
                   chat      -> $CHAT_DEPLOYMENT
                   embedding -> $EMBED_DEPLOYMENT
Content Safety   : $CONTENT_SAFETY
PostgreSQL       : $PG_HOST (db: $PG_DB, user: $PG_ADMIN)
Cosmos (Mongo)   : $COSMOS (db: $MONGO_DB, collection: $MONGO_COLLECTION)
Blob storage     : $ST_IMAGES / container $IMAGES_CONTAINER
Table storage    : $ST_SENSORS / table $SENSOR_TABLE
Key Vault        : $KEYVAULT (16 secrets)

Next:
  1. Seed the data:   ./scripts/seed_data.sh
  2. Verify it all:   python scripts/verify_azure.py
EOF
