from __future__ import annotations

import os

# You can use google-cloud-aiplatform (Python SDK) or direct REST.
# We'll implement via REST to avoid heavy deps; supply PROJECT & LOCATION.
# Docs: Vertex AI Text Embeddings API (Gemini).
import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from kb_vectorizer.embedding.interfaces import BaseEmbedder, EmbedResponse


class VertexAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-004",  # current Vertex text embedding model
        project_id: str | None = None,
        location: str = "us-central1",
        access_token: str | None = None,    # OAuth2 bearer token
        max_batch_size: int = 256,
    ):
        self.model_name = model
        self.project_id = project_id or os.getenv("VERTEX_PROJECT")
        self.location = location or os.getenv("VERTEX_LOCATION", "us-central1")
        self.token = access_token or os.getenv("VERTEX_ACCESS_TOKEN")
        if not (self.project_id and self.token):
            raise RuntimeError("VertexAIEmbedder requires PROJECT and ACCESS TOKEN (use gcloud auth print-access-token).")
        self.max_batch_size = max_batch_size
        self.dimension = None

    @retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=0.25, max=8.0), reraise=True)
    def embed(self, texts: list[str]) -> EmbedResponse:
        url = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/google/models/{self.model_name}:embedText"
        )
        # API accepts instances list: {"content": "..."}; returns embeddings[].values
        # Ref: Get text embeddings docs.
        body = {"instances": [{"content": t} for t in texts]}
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
        vecs: list[list[float]] = [inst["embeddings"]["values"] for inst in data.get("predictions", [])]
        if self.dimension is None and vecs:
            self.dimension = len(vecs[0])
        return EmbedResponse(vectors=vecs, model=self.model_name, dimension=self.dimension or (len(vecs[0]) if vecs else 0))
