"""
Ingestion pipeline for the RAG knowledge base.

Loads financial documents (.txt, .md, .pdf) from data/documents/, splits them
into overlapping chunks, embeds them with a local sentence-transformers
model, and builds a FAISS index for fast similarity search at query time.

Run directly to (re)build the index from scratch:
    python -m src.ingestion
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config
from src.utils import get_logger, save_pickle

logger = get_logger(__name__)

_embedding_model: SentenceTransformer | None = None


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedding_model


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, source: str) -> List[DocumentChunk]:
    """Fixed-size sliding-window chunking with overlap, chunked on characters
    for simplicity and determinism (swap for a token-aware splitter if your
    documents mix languages/scripts heavily)."""
    chunks = []
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + config.CHUNK_SIZE].strip()
        if len(piece) < 40:  # skip near-empty tail fragments
            continue
        chunks.append(DocumentChunk(text=piece, source=source, chunk_id=i))
    return chunks


def load_and_chunk_documents(documents_dir: Path = config.DOCUMENTS_DIR) -> List[DocumentChunk]:
    supported = {".txt", ".md", ".pdf"}
    all_chunks: List[DocumentChunk] = []

    files = [p for p in documents_dir.rglob("*") if p.suffix.lower() in supported]
    if not files:
        logger.warning(
            "No documents found in %s. Add .txt/.md/.pdf files (fund fact sheets, "
            "market commentary, client statements) before ingesting.",
            documents_dir,
        )

    for path in files:
        text = read_file(path)
        chunks = chunk_text(text, source=path.name)
        all_chunks.extend(chunks)
        logger.info("Chunked %s -> %s chunks", path.name, len(chunks))

    return all_chunks


def build_index(chunks: List[DocumentChunk]) -> None:
    if not chunks:
        logger.warning("No chunks to index; skipping FAISS build.")
        return

    model = get_embedding_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
    index.add(embeddings)

    faiss.write_index(index, str(config.FAISS_INDEX_PATH))
    save_pickle(chunks, config.CHUNK_STORE_PATH)

    logger.info(
        "Built FAISS index with %s vectors (dim=%s) -> %s",
        index.ntotal, dim, config.FAISS_INDEX_PATH,
    )


def main():
    chunks = load_and_chunk_documents()
    build_index(chunks)


if __name__ == "__main__":
    main()
