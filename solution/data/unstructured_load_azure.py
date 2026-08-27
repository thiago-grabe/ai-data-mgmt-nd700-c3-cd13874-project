import os

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    DeviceCodeCredential,
)
from azure.keyvault.secrets import SecretClient
from pymongo import MongoClient

# pypdf is the maintained successor to PyPDF2 (same codebase, same PdfReader
# API). requirements.txt shipped without ANY pdf library, so this import failed
# outright as delivered.
from pypdf import PdfReader

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

mongo_uri = kv_client.get_secret("unstructuredmongourl").value
db_name = kv_client.get_secret("unstructureddbname").value
collection_name = kv_client.get_secret("unstructuredcollectionname").value


def load_pdfs_to_mongo(folder_path="unstructured"):
    # Connect to MongoDB
    client = MongoClient(mongo_uri)

    # Create or access the database and collection
    db = client[db_name]

    # Explicitly create collection if it doesn't exist
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

    collection = db[collection_name]

    # Ensure folder exists
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Loop through PDFs
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing: {filename}")

            with open(file_path, "rb") as f:
                reader = PdfReader(f)
                text = ""

                for page in reader.pages:
                    text += page.extract_text() or ""

            # Insert into MongoDB
            doc = {
                "filename": filename,
                "content": text
            }

            collection.insert_one(doc)

    print(f"All PDFs loaded into MongoDB ({db_name}.{collection_name}).")


if __name__ == "__main__":
    load_pdfs_to_mongo()

