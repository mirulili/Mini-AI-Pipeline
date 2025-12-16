from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from . import config


def get_openai_client() -> OpenAI:
    """Initialize and return an OpenAI client."""
    return OpenAI(api_key=config.get_openai_api_key())


def openai_embed(texts: list[str], client: OpenAI | None = None) -> NDArray[np.float32]:
    """Convert a list of texts to L2-normalized OpenAI embeddings."""
    if client is None:
        client = get_openai_client()

    if not texts:
        return np.array([], dtype=np.float32)

    resp = client.embeddings.create(
        model=config.MODEL_EMBEDDING,
        input=texts,
    )

    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)

    # L2 normalization (for Cosine Similarity)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def map_label(val: object) -> str:
    """Normalize answer labels to standard format (A/B/C/D)."""
    s = str(val).strip().upper()
    mapping = {
        "1": "A",
        "2": "B",
        "3": "C",
        "4": "D",
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
    }
    if s not in mapping:
        raise ValueError(f"Invalid label: {val}")
    return mapping[s]
