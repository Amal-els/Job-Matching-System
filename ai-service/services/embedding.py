import os
import numpy as np
from huggingface_hub import InferenceClient


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


def embed(text: str) -> list:
    result = _get_client().feature_extraction(text or "", model=_model_name())
    return _pool(result)


def embed_many(texts: list) -> list:
    client = _get_client()
    model = _model_name()
    return [_pool(client.feature_extraction(t or "", model=model)) for t in texts]
