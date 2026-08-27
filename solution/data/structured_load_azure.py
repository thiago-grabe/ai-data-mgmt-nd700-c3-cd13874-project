# brew install postgresql@15
# brew services start postgresql@15
# brew services list
# brew link postgresql@15 --force
# psql --version
# postgresql cli: psql postgres 
# brew services stop postgresql@15

# change user name to admin
# psql -U postgres -d postgres
# \dt

import csv
import os
import re
from pathlib import Path

import psycopg
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DeviceCodeCredential,
)
from azure.keyvault.secrets import SecretClient

# pip install "psycopg[binary]"

# ---------- CONFIG ----------

# Key Vault name. A resource identifier, not a secret: it grants nothing on its
# own and every read is gated by Entra ID. Overridable by environment.
keyVaultName = os.environ.get("KEYVAULT_NAME", "kv-nbhd-4hbr")
KVUri = f"https://{keyVaultName}.vault.azure.net/"

print("Connecting to Azure for authentication.")
# DeviceCodeCredential is the documented path for this project. AzureCliCredential
# is tried first so an existing `az login` session is reused instead of demanding a
# fresh device code on every run.
credential = ChainedTokenCredential(AzureCliCredential(), DeviceCodeCredential())
kv_client = SecretClient(vault_url=KVUri, credential=credential)

DB_CONFIG = {
    "host": kv_client.get_secret("structuredpostgresqlhost").value,
    "dbname": kv_client.get_secret("structuredpostgresqldbname").value,
    "user": kv_client.get_secret("structuredpostgresqluser").value,
    "password": kv_client.get_secret("structuredpostgresqlpassword").value,
    "port": 5432,
}

DATA_FOLDER = Path("./structured")  # folder containing your CSV datasets


# ---------- HELPERS ----------

def clean_identifier(name: str) -> str:
    """Convert column/table names to safe SQL identifiers."""
    name = name.lower()
    name = re.sub(r"\W+", "_", name)
    return name.strip("_")


def infer_type(value: str):
    """Simple type inference."""
    if value == "":
        return "TEXT"

    try:
        int(value)
        return "INTEGER"
    except:
        pass

    try:
        float(value)
        return "FLOAT"
    except:
        pass

    if value.lower() in {"true", "false", "yes", "no"}:
        return "BOOLEAN"

    return "TEXT"


def infer_schema(csv_path: Path):
    """Infer schema from first non-empty row."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)

    schema = {}

    for col, value in first_row.items():
        schema[clean_identifier(col)] = infer_type(value)

    return schema


# ---------- DATABASE FUNCTIONS ----------

def create_table(conn, table_name, schema):
    columns_sql = ",\n".join(
        f"{col} {dtype}" for col, dtype in schema.items()
    )

    create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {columns_sql}
        );
    """

    with conn.cursor() as cur:
        cur.execute(create_sql)

    conn.commit()
    print(f"✔ Table created: {table_name}")


def ingest_csv(conn, csv_path, table_name, schema):
    columns = list(schema.keys())

    insert_sql = f"""
        INSERT INTO {table_name} ({",".join(columns)})
        VALUES ({",".join(["%s"] * len(columns))})
    """

    rows = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cleaned = []

            for col in row.values():
                if col == "":
                    cleaned.append(None)
                else:
                    cleaned.append(col)

            rows.append(tuple(cleaned))

    with conn.cursor() as cur:
        cur.executemany(insert_sql, rows)

    conn.commit()

    print(f"✔ Loaded {len(rows)} rows into {table_name}")


# ---------- PIPELINE ----------

def process_csv(conn, csv_path):
    table_name = clean_identifier(csv_path.stem)

    print(f"\nProcessing: {csv_path.name}")

    schema = infer_schema(csv_path)

    create_table(conn, table_name, schema)
    ingest_csv(conn, csv_path, table_name, schema)


def main():
    try:
        with psycopg.connect(**DB_CONFIG) as conn:

            csv_files = list(DATA_FOLDER.glob("*.csv"))

            if not csv_files:
                print("No CSV files found.")
                return

            for csv_file in csv_files:
                process_csv(conn, csv_file)

        print("\n✅ All datasets ingested successfully!")

    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    main()
