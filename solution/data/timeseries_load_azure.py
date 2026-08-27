"""Load the synthetic environmental-sensor readings into Azure Table Storage.

Companion to structured_load_azure.py and unstructured_load_azure.py, following the
same shape: pull credentials from Key Vault, then write to exactly one data store.

This backs the fourth (time-series) agent. It uses its OWN storage account, not the
one holding the house images, so no agent ever holds another agent's credential.

Run from the data/ directory:
    cd solution/data && python timeseries_load_azure.py

Idempotent: rows are upserted on (PartitionKey, RowKey), so re-running refreshes
rather than duplicating.
"""

import csv
import os
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.data.tables import TableServiceClient
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DeviceCodeCredential,
)
from azure.keyvault.secrets import SecretClient

# ---------- CONFIG ----------

keyVaultName = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
KVUri = f"https://{keyVaultName}.vault.azure.net/"

print("Connecting to Azure for authentication.")
# DeviceCodeCredential is the documented path for this project. AzureCliCredential
# is tried first so an existing `az login` session is reused and repeated runs do
# not demand a fresh device code.
credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())
kv_client = SecretClient(vault_url=KVUri, credential=credential)

CONNECTION_STRING = kv_client.get_secret("timeseriesazureconnstring").value
TABLE_NAME = kv_client.get_secret("timeseriestablename").value

DATA_FILE = Path("./timeseries/neighborhood_sensors.csv")
BATCH_SIZE = 100


def load_sensors_to_table(csv_path: Path = DATA_FILE, table_name: str = TABLE_NAME) -> int:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Generate it first: "
            "cd timeseries && python generate_sensors.py"
        )

    service = TableServiceClient.from_connection_string(CONNECTION_STRING)
    try:
        service.create_table(table_name)
        print(f"Created table {table_name}.")
    except ResourceExistsError:
        print(f"Table {table_name} already exists; upserting into it.")

    table = service.get_table_client(table_name)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    written = 0
    for row in rows:
        entity = {
            # Partition by neighborhood so per-neighborhood queries stay a single
            # partition scan; timestamp as row key keeps each partition ordered.
            "PartitionKey": row["neighborhood"],
            "RowKey": row["timestamp"],
            "sensor_id": row["sensor_id"],
            "pm25": float(row["pm25"]),
            "noise_db": float(row["noise_db"]),
            "water_gal": float(row["water_gal"]),
        }
        table.upsert_entity(entity)
        written += 1
        if written % BATCH_SIZE == 0:
            print(f"  {written}/{len(rows)} readings written")

    print(f"Loaded {written} sensor readings into {table_name}.")
    return written


if __name__ == "__main__":
    load_sensors_to_table()
