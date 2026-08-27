#!/usr/bin/env bash
# Delete everything scripts/provision_azure.sh created. The lab subscription bills
# continuously for the PostgreSQL server, so tear down when you are finished.
#
# Leaves aifoundry-agentic-cd14705 alone: it predates this project. Pass
# DELETE_DEPLOYMENTS=1 to also remove the two model deployments added to it.
set -euo pipefail

RG="${RESOURCE_GROUP:-Regroup_4hbr}"
LOCATION="${LOCATION:-westus}"
KEYVAULT="${KEYVAULT_NAME:-kv-nbhd-4hbr}"
PG_SERVER="${PG_SERVER_NAME:-pg-nbhd-4hbr}"
COSMOS="${COSMOS_NAME:-cosmos-nbhd-4hbr}"
ST_IMAGES="${ST_IMAGES_NAME:-stnbhdimg4hbr}"
ST_SENSORS="${ST_SENSORS_NAME:-stnbhdsensors4hbr}"
CONTENT_SAFETY="${CONTENT_SAFETY_NAME:-cs-nbhd-4hbr}"
FOUNDRY="${FOUNDRY_NAME:-aifoundry-agentic-cd14705}"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m    OK  %s\033[0m\n' "$*"; }

read -r -p "Delete all project resources in $RG? [y/N] " reply
[[ "$reply" == "y" || "$reply" == "Y" ]] || { echo "aborted"; exit 0; }

step "PostgreSQL $PG_SERVER"
az postgres flexible-server delete -g "$RG" -n "$PG_SERVER" --yes -o none 2>/dev/null && ok deleted || ok "absent"

step "Cosmos DB $COSMOS"
az cosmosdb delete -g "$RG" -n "$COSMOS" --yes -o none 2>/dev/null && ok deleted || ok "absent"

step "Storage accounts"
for sa in "$ST_IMAGES" "$ST_SENSORS"; do
  az storage account delete -g "$RG" -n "$sa" --yes -o none 2>/dev/null && ok "$sa deleted" || ok "$sa absent"
done

step "Content Safety $CONTENT_SAFETY"
az cognitiveservices account delete -g "$RG" -n "$CONTENT_SAFETY" -o none 2>/dev/null && ok deleted || ok "absent"
az cognitiveservices account purge -g "$RG" -n "$CONTENT_SAFETY" -l "$LOCATION" -o none 2>/dev/null || true

step "Key Vault $KEYVAULT"
az keyvault delete -g "$RG" -n "$KEYVAULT" -o none 2>/dev/null && ok deleted || ok "absent"
# Purge so the name is immediately reusable; soft-delete would otherwise block a re-run.
az keyvault purge -n "$KEYVAULT" -l "$LOCATION" -o none 2>/dev/null && ok purged || true

if [[ "${DELETE_DEPLOYMENTS:-0}" == "1" ]]; then
  step "Model deployments on $FOUNDRY"
  for dep in gpt-4.1-mini text-embedding-ada-002; do
    az cognitiveservices account deployment delete -g "$RG" -n "$FOUNDRY" \
      --deployment-name "$dep" -o none 2>/dev/null && ok "$dep deleted" || ok "$dep absent"
  done
fi

step "Local state"
rm -f "$(dirname "${BASH_SOURCE[0]}")/../.pg_password"
ok "removed .pg_password"

step "Done"
echo "The AI Foundry account $FOUNDRY was left in place."
