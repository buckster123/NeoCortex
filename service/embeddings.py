"""
Embedding functions for Neo-Cortex.

Adapted from Knowledge Base service with cloud-compatible additions.

Priority order:
1. sentence-transformers (primary, CPU-based, works everywhere)
2. Ollama (alternative, if running locally)
3. OpenAI/Voyage (for cloud deployments)
4. Fallback hash (last resort, no semantic meaning)
"""

import hashlib
from typing import List, Optional, Protocol
import logging

from .config import (
    SBERT_MODEL,
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def name(self) -> str: ...
    def embed(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...
    def __call__(self, input: List[str]) -> List[List[float]]: ...


class SentenceTransformerEmbeddings:
    """Primary embedding function using sentence-transformers."""

    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self._model = None
        self._name = f"sbert:{model_name}"

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def name(self) -> str:
        """Return embedding function name for ChromaDB."""
        return self._name

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return EMBEDDING_DIM

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts (ChromaDB interface)."""
        return self.embed(input)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str = None, *, input: str = None) -> List[float]:
        """Embed a single query text."""
        query = text if text is not None else input
        if query is None:
            raise ValueError("Must provide text or input")
        model = self._load_model()
        embedding = model.encode(query, convert_to_numpy=True)
        return embedding.tolist()


class OllamaEmbeddings:
    """Alternative embedding function using Ollama API."""

    def __init__(self, model: str = OLLAMA_EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._dimension = None
        self._name = f"ollama:{model}"

    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension or 768  # Default for nomic-embed-text

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed(input)

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx
        embeddings = []
        for text in texts:
            try:
                embedding = self._embed_single(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                if self._dimension:
                    embeddings.append([0.0] * self._dimension)
                else:
                    raise
        return embeddings

    def _embed_single(self, text: str) -> List[float]:
        import httpx
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            if self._dimension is None:
                self._dimension = len(embedding)
            return embedding

    def embed_query(self, text: str = None, *, input: str = None) -> List[float]:
        query = text if text is not None else input
        if query is None:
            raise ValueError("Must provide text or input")
        return self._embed_single(query)


class FallbackEmbeddings:
    """Hash-based embeddings when nothing else is available."""

    def __init__(self, dimension: int = EMBEDDING_DIM):
        self._dimension = dimension
        self._name = "fallback"
        logger.warning("Using fallback embeddings - no semantic search available")

    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed(input)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(text) for text in texts]

    def _hash_embed(self, text: str) -> List[float]:
        embeddings = []
        for i in range(self._dimension):
            hash_input = f"{text}:{i}".encode()
            hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
            normalized = ((hash_val % 10000) / 5000) - 1
            embeddings.append(normalized)
        return embeddings

    def embed_query(self, text: str = None, *, input: str = None) -> List[float]:
        query = text if text is not None else input
        if query is None:
            raise ValueError("Must provide text or input")
        return self._hash_embed(query)


# =============================================================================
# Availability Checks
# =============================================================================

def check_sentence_transformers_available() -> bool:
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


def check_ollama_available() -> bool:
    """Check if Ollama is running and the embedding model is available."""
    try:
        import httpx
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return OLLAMA_EMBEDDING_MODEL.split(":")[0] in model_names
    except Exception:
        return False


# =============================================================================
# Factory Function
# =============================================================================

def get_embedding_function(prefer: str = "auto") -> EmbeddingFunction:
    """
    Get the best available embedding function.

    Args:
        prefer: "auto", "sbert", "ollama", or "fallback"

    Returns:
        Embedding function instance
    """
    if prefer == "auto":
        if check_sentence_transformers_available():
            logger.info("Using sentence-transformers embeddings")
            return SentenceTransformerEmbeddings()
        if check_ollama_available():
            logger.info("Using Ollama embeddings")
            return OllamaEmbeddings()
        logger.warning("No embedding model available, using fallback")
        return FallbackEmbeddings()

    elif prefer == "sbert":
        if check_sentence_transformers_available():
            return SentenceTransformerEmbeddings()
        raise RuntimeError("sentence-transformers not available")

    elif prefer == "ollama":
        if check_ollama_available():
            return OllamaEmbeddings()
        raise RuntimeError("Ollama not available")

    elif prefer == "fallback":
        return FallbackEmbeddings()

    else:
        raise ValueError(f"Unknown embedding preference: {prefer}")
