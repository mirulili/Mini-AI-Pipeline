from __future__ import annotations

from . import config
from .generator import call_llm
from .retriever import query as retrieve


def _shorten_context(docs: list[str]) -> str:
    """Limit retrieved context to max length to avoid token limits."""
    joined = "\n\n".join(docs)
    return joined[: config.MAX_CONTEXT_LEN]


def answer_question(raw_question: str) -> str:
    """RAG Pipeline: Retrieve + Generate."""
    if config.TOP_K > 0:
        hits = retrieve(raw_question, top_k=config.TOP_K)
        docs = [d for d, _ in hits]
    else:
        hits = []
        docs = []
    context = _shorten_context(docs)
    letter = call_llm(raw_question, context)
    return letter
