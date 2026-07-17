"""
RAG pipeline: embed the user's query, retrieve the most relevant document
chunks from the FAISS index, assemble a grounded prompt, and call the LLM
for a context-aware answer with source attribution.
"""

from typing import List, Optional

import faiss
import numpy as np

import config
from src.ingestion import DocumentChunk, get_embedding_model
from src.llm_client import generate_response
from src.utils import get_logger, load_pickle

logger = get_logger(__name__)

_index: Optional[faiss.Index] = None
_chunks: Optional[List[DocumentChunk]] = None


def _load_index():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(str(config.FAISS_INDEX_PATH))
        _chunks = load_pickle(config.CHUNK_STORE_PATH)
        logger.info("Loaded FAISS index with %s vectors", _index.ntotal)
    return _index, _chunks


def retrieve(query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[dict]:
    """Return the top_k most relevant chunks with similarity scores, filtered
    by config.MIN_SIMILARITY_SCORE."""
    index, chunks = _load_index()
    model = get_embedding_model()

    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < config.MIN_SIMILARITY_SCORE:
            continue
        chunk = chunks[idx]
        results.append({
            "text": chunk.text,
            "source": chunk.source,
            "score": round(float(score), 4),
        })
    return results


def build_context_block(retrieved: List[dict]) -> str:
    if not retrieved:
        return "No relevant documents were found in the knowledge base for this query."

    parts = []
    for r in retrieved:
        parts.append(f"[Source: {r['source']} | relevance: {r['score']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def answer_query(
    query: str,
    conversation_history: Optional[List[dict]] = None,
    portfolio_context: Optional[str] = None,
) -> dict:
    """
    Full RAG turn: retrieve context, optionally fold in live portfolio data,
    and generate a grounded answer.

    Returns a dict with the answer text and the sources used, so the API/UI
    layer can display citations alongside the response.
    """
    retrieved = retrieve(query)
    context_block = build_context_block(retrieved)

    prompt_parts = [f"Relevant knowledge base context:\n{context_block}"]
    if portfolio_context:
        prompt_parts.append(f"Client portfolio data:\n{portfolio_context}")
    prompt_parts.append(f"Client/advisor question:\n{query}")

    user_message = "\n\n".join(prompt_parts)

    answer = generate_response(
        system_prompt=config.SYSTEM_PROMPT,
        user_message=user_message,
        conversation_history=conversation_history,
    )

    return {
        "answer": answer,
        "sources": [{"source": r["source"], "score": r["score"]} for r in retrieved],
    }
