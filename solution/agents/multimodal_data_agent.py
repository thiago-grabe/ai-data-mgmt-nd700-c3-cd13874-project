import warnings
from io import BytesIO

from azure.storage.blob import BlobServiceClient

warnings.filterwarnings("ignore", message="Field.*conflict with protected namespace")
import os

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


import matplotlib.pyplot as plt
import torch
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeImageOptions, ImageData
from azure.core.credentials import AzureKeyCredential
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Azure AI Content Safety rejects images larger than 2048x2048 or 4 MB.
# 274_Maplewood_Crescent.jpg is 2268x1676 and would be refused, which inside
# find_similar's try/except silently drops one of the twelve houses from the
# candidate set while printing a confusing "Skipping:" line.
CONTENT_SAFETY_MAX_DIM = 2048


class MultimodalDataAgent:
    """CLIP image similarity over Azure Blob Storage, screened by Content Safety.

    Connects to exactly one data source: the Blob Storage container of house
    photographs. It holds no PostgreSQL or MongoDB credential.
    """

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        azure_conn_str = None,
        container_name = None,
        content_safety_endpoint = None,
        content_safety_key = None
    ):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Azure Blob Storage
        self.azure_conn_str = azure_conn_str
        self.container_name = container_name

        if azure_conn_str and container_name:
            self.blob_service = BlobServiceClient.from_connection_string(azure_conn_str)
            self.container_client = self.blob_service.get_container_client(container_name)
        else:
            self.container_client = None

        # Azure Content Safety
        self.content_safety_client = ContentSafetyClient(endpoint=content_safety_endpoint, credential=AzureKeyCredential(content_safety_key))

        # Per-query caches. The shipped flow analysed each blob once during the
        # similarity sweep and then again when rendering the top matches, paying
        # for and printing every Content Safety verdict twice.
        self._image_cache = {}
        self._safety_cache = {}
        self.last_safety_findings = []
        self.blocked_blobs = []

    # ============================================
    # PRIVATE METHODS
    # ============================================

    def __downscale_for_safety(self, image_bytes):
        """Return bytes within the Content Safety size limit, re-encoding only if needed."""
        image = Image.open(BytesIO(image_bytes))
        if max(image.size) <= CONTENT_SAFETY_MAX_DIM:
            return image_bytes

        resized = image.convert("RGB")
        resized.thumbnail((CONTENT_SAFETY_MAX_DIM, CONTENT_SAFETY_MAX_DIM), Image.LANCZOS)
        buffer = BytesIO()
        resized.save(buffer, format="JPEG", quality=92)
        print(
            f"  (downscaled {image.size[0]}x{image.size[1]} -> "
            f"{resized.size[0]}x{resized.size[1]} for content safety)"
        )
        return buffer.getvalue()

    def __analyze_bytes(self, image_bytes, label):
        """Run Content Safety over raw image bytes and print the category severities."""
        # ImageData.content is declared rest_field(format="base64") in
        # azure-ai-contentsafety 1.0.0, so the SDK base64-encodes these bytes for
        # us. Pinned to ==1.0.0 because 1.0.0b1 typed it as str and required
        # manual encoding.
        request = AnalyzeImageOptions(image=ImageData(content=self.__downscale_for_safety(image_bytes)))
        response = self.content_safety_client.analyze_image(request)

        print(f"Analyzing Blob Image: {label}")
        inappropriate_content = False
        severities = {}
        for c in response.categories_analysis:
            print(c.category, c.severity)
            severities[str(c.category)] = int(c.severity)
            if c.severity != 0:
                inappropriate_content = True

        if inappropriate_content:
            print(f"Warning: Harmful content detected in blob image: {label}\n")
        else:
            print(f"No harmful content detected in blob image: {label}\n")

        self.last_safety_findings.append({
            "image": label,
            "severities": severities,
            "flagged": inappropriate_content,
        })
        return inappropriate_content, severities

    def __load_image_from_blob(self, blob_name):

        if blob_name in self._image_cache:
            return self._image_cache[blob_name]

        print("\n" + "=" * 80)
        print(f"Analysing image: {blob_name}")

        blob = self.container_client.get_blob_client(blob_name)
        image_bytes = blob.download_blob().readall()

        # --- RUN CONTENT SAFETY ---
        flagged, _ = self.__analyze_bytes(image_bytes, blob_name)
        self._safety_cache[blob_name] = flagged

        # Return the image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        self._image_cache[blob_name] = image
        return image

    def __load_image(self, path):
        return Image.open(path).convert("RGB")

    def __compute_similarity_blob(self, query_image_path, blob_name):
        img1 = self.__load_image(query_image_path)
        img2 = self.__load_image_from_blob(blob_name)

        inputs = self.processor(images=[img1, img2], return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        if not isinstance(outputs, torch.Tensor):
            image_embeds = outputs.pooler_output
        else:
            image_embeds = outputs

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        similarity = torch.dot(image_embeds[0], image_embeds[1]).item()
        return similarity

    # ============================================
    # PUBLIC METHOD — SCREEN THE QUERY IMAGE
    # ============================================

    def check_query_image(self, query_image_path):
        """Screen the user's own image before it is embedded or compared.

        Without this the user-supplied image is the only input in the whole system
        that never passes a safety check.
        """
        print("\n" + "=" * 80)
        print(f"Screening query image before search: {query_image_path}")
        print("=" * 80)
        with open(query_image_path, "rb") as fh:
            flagged, severities = self.__analyze_bytes(fh.read(), os.path.basename(query_image_path))
        return flagged, severities

    # ============================================
    # PUBLIC METHOD — FIND SIMILAR
    # ============================================

    def find_similar(self, query_image_path, prefix="", top_k=3):
        if not self.container_client:
            raise ValueError("Azure Blob Storage is not configured.")

        self._image_cache = {}
        self._safety_cache = {}
        self.last_safety_findings = []
        self.blocked_blobs = []

        results = []

        # Screen the query image first: if the user's own input is unsafe there is
        # no reason to embed it or to scan twelve blobs against it.
        query_flagged, _ = self.check_query_image(query_image_path)
        if query_flagged:
            raise ValueError(
                f"Query image {os.path.basename(query_image_path)} was flagged by "
                "Azure AI Content Safety; similarity search aborted."
            )

        print("\n" + "=" * 80)
        print("Using Multimodal (Image files) Data stored in Azure Data Blob to answer question")
        print("Scan images for harmful contente")
        print("=" * 80)

        blob_list = list(self.container_client.list_blobs(name_starts_with=prefix))
        print("Blob count:", len(blob_list))

        for blob in blob_list:
            fname = blob.name
            print(fname)

            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                similarity = self.__compute_similarity_blob(query_image_path, fname)

                # Actionable safeguard: an image the safety service flagged is
                # excluded from the results rather than merely logged.
                if self._safety_cache.get(fname):
                    self.blocked_blobs.append(fname)
                    print(f"Excluding {fname} from results: flagged by content safety.")
                    continue

                address = os.path.splitext(os.path.basename(fname))[0].replace("_", " ")
                results.append((similarity, fname, address))

            except Exception as e:
                print("Skipping:", fname, e)

        results.sort(reverse=True, key=lambda x: x[0])

        query_img = self.__load_image(query_image_path)
        return results[:top_k], query_img

    # ============================================
    # PUBLIC METHOD — SHOW RESULTS
    # ============================================

    def show_results(self, query_img, matches, save_path=None):
        # Grid sized from the actual match count. The shipped subplot(1, 4, i)
        # hard-coded room for exactly three matches and raised for any other top_k.
        columns = len(matches) + 1
        plt.figure(figsize=(4 * columns, 4))

        plt.subplot(1, columns, 1)
        plt.imshow(query_img)
        plt.title("Query")
        plt.axis("off")

        for i, (score, blob_name, address) in enumerate(matches, start=2):
            # Served from the per-query cache, so this neither re-downloads the
            # image nor bills a second Content Safety call.
            img = self.__load_image_from_blob(blob_name)

            plt.subplot(1, columns, i)
            plt.imshow(img)
            plt.title(f"{address}\n{score:.3f}")
            plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            plt.close("all")
            print(f"[evidence] figure saved: {save_path}")
        else:
            plt.show()
