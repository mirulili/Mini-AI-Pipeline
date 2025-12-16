from __future__ import annotations

import json
from typing import cast

import joblib
from sklearn.neighbors import NearestNeighbors

from . import config, utils

# Global cache
_CACHED_TEXTS: list[str] | None = None
_CACHED_NN: NearestNeighbors | None = None


def _load_resources() -> tuple[list[str], NearestNeighbors]:
    """
    Loads index resources (texts, NN) from disk and caches them.
    """
    global _CACHED_TEXTS, _CACHED_NN

    if _CACHED_TEXTS is None:
        if not config.INDEX_TXT.exists():
            raise FileNotFoundError(
                f"Missing index texts: {config.INDEX_TXT}. "
                "Please run 'make index' to generate it."
            )
        _CACHED_TEXTS = json.loads(config.INDEX_TXT.read_text(encoding="utf-8"))

    if _CACHED_NN is None:
        if not config.INDEX_NN.exists():
            raise FileNotFoundError(
                f"Missing NN index: {config.INDEX_NN}. "
                "Please run 'make index' to generate it."
            )
        _CACHED_NN = cast(NearestNeighbors, joblib.load(config.INDEX_NN))

    return cast(list[str], _CACHED_TEXTS), cast(NearestNeighbors, _CACHED_NN)


def query(text: str, top_k: int = config.TOP_K) -> list[tuple[str, float]]:
    """
    Retrieves the most similar documents for the input text.
    """
    texts, nn = _load_resources()

    # Generate embedding
    embs = utils.openai_embed([text])  # Shape: (1, D)

    # Search (kneighbors returns distances and indices)
    distances, indices = nn.kneighbors(embs, n_neighbors=min(top_k, len(texts)))

    # Convert numpy output to lists
    # indices[0] is the list of indices for the first (and only) query vector
    idxs = indices[0].tolist()
    dists = distances[0].tolist()

    # Map back to text
    docs = [texts[i] for i in idxs]

    return list(zip(docs, dists, strict=False))
