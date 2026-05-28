from __future__ import annotations

import os

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from kb_vectorizer.embedding.interfaces import BaseEmbedder, EmbedResponse


class CloudEmbedder(BaseEmbedder):
    """A generic cloud embedder that interfaces with OpenAI-compatible embedding APIs.
    
    This can be used to connect to OpenAI, LiteLLM proxy, vLLM, or any other endpoint 
    that supports the standard `/v1/embeddings` JSON format.
    """

    def __init__(
        self,
        model_id: str = "text-embedding-3-small",
        api_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 512,
    ):
        """Initialize the cloud embedder.

        Args:
            model_id: The ID of the model to use on the remote server.
            api_url: The base URL of the API. If not provided, defaults to OpenAI's public API or `OPENAI_API_BASE`.
                     It should point to the root or the embeddings endpoint directly.
            api_key: The authentication key for the API. Defaults to `OPENAI_API_KEY` from the environment.
            batch_size: The batch size for each request.

        """
        self.model_name = model_id
        
        # Determine API base URL (fallback to standard OpenAI URL if not given)
        base_url = api_url or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        self.api_url = f"{base_url.rstrip('/')}/embeddings"
        
        # Determine API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("CloudEmbedder requires an API key (passed directly or via OPENAI_API_KEY).")
        
        self.max_batch_size = batch_size
        self.dimension = None

    @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(initial=0.5, max=10.0), reraise=True)
    def embed(self, texts: list[str]) -> EmbedResponse:
        """Embeds a list of texts by sending them to the cloud API.
        
        Args:
            texts: A list of strings to embed.
            
        Returns:
            An EmbedResponse containing the computed vectors, model name, and dimension.

        """
        if not texts:
            return EmbedResponse(vectors=[], model=self.model_name, dimension=self.dimension or 0)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        data = response.json()
        
        # The standard OpenAI format has an "data" array with "embedding" arrays inside
        vectors = []
        for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
            vectors.append(item["embedding"])
            
        if self.dimension is None and vectors:
            self.dimension = len(vectors[0])
            
        return EmbedResponse(
            vectors=vectors,
            model=self.model_name,
            dimension=self.dimension or (len(vectors[0]) if vectors else 0)
        )