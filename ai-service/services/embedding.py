import os
import hashlib
import logging
import numpy as np
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


def _get_client():
    return InferenceClient(token=os.getenv("HF_TOKEN"))


def _model_name():
    return os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def _pool(result):
    """Mean-pool token-level embeddings if needed."""
    arr = np.array(result)
    if arr.ndim == 2:
        return arr.mean(axis=0).tolist()
    return arr.tolist()


def _fallback_dim() -> int:
    try:
        return max(32, int(os.getenv("EMBEDDING_FALLBACK_DIM", "384")))
    except ValueError:
        return 384


def _embed_local(text: str) -> list:
    """
    Deterministic local fallback embedding when remote provider is unreachable.
    Uses hashed token frequencies projected onto a fixed-size vector.
    """
    dim = _fallback_dim()
    vector = np.zeros(dim, dtype=float)
    tokens = (text or "").lower().split()
    if not tokens:
        return vector.tolist()

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()


def embed(text: str) -> list:
    """
    Generate embedding for a single text using Hugging Face Inference API.
    Falls back to a deterministic local embedding if the API is unavailable.
    """
    try:
        result = _get_client().feature_extraction(text or "", model=_model_name())
        return _pool(result)
    except Exception as exc:
        logger.warning("Remote embedding unavailable, using local fallback: %s", exc)
        return _embed_local(text or "")


def embed_many(texts: list) -> list:
    """
    Generate embeddings for multiple texts using Hugging Face Inference API.
    Falls back to deterministic local embeddings if the API is unavailable.
    """
    try:
        client = _get_client()
        model = _model_name()
        # The inference API can handle lists, but let's ensure consistency with local pooling
        return [_pool(client.feature_extraction(t or "", model=model)) for t in texts]
    except Exception as exc:
        logger.warning("Remote batch embedding unavailable, using local fallback: %s", exc)
        return [_embed_local(t or "") for t in texts]
