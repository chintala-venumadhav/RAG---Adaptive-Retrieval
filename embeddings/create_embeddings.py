"""
create_embeddings.py – Orchestrate the Full Embedding Pipeline
===============================================================
Ties together:
1. PDF loading   (preprocessing.pdf_loader)
2. Text cleaning (preprocessing.cleaner)
3. Smart chunking(preprocessing.chunker)
4. Embedding     (embeddings.embedding_model)
5. Saving        (pickle + handoff to FAISS manager)

All artefacts are persisted to ``data/embeddings/`` as pickle files
so they can be reloaded without reprocessing.

Author  : B.Tech CSE-AI Student Project
"""

import pickle
import time
from typing import Any, Dict, List

# pyrefly: ignore [missing-import]
import numpy as np

from embeddings.embedding_model import EmbeddingModel
from preprocessing.chunker import chunk_documents
from preprocessing.cleaner import clean_documents
from preprocessing.pdf_loader import load_all_pdfs
from utils.config import CHUNKS_PKL_PATH, EMBEDDINGS_PKL_PATH
from utils.logger import logger
from vectordb.faiss_manager import FAISSManager


def save_pickle(data: Any, path: str) -> None:
    """
    Serialize any Python object to a pickle file.

    Parameters
    ----------
    data : Any
        The object to save.
    path : str
        Destination file path.
    """
    with open(path, "wb") as fh:
        pickle.dump(data, fh)
    logger.info("Saved pickle: %s", path)


def load_pickle(path: str) -> Any:
    """
    Deserialize a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickle file.

    Returns
    -------
    Any
        The deserialized Python object.
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    logger.info("Loaded pickle: %s", path)
    return data


def run_embedding_pipeline() -> Dict[str, Any]:
    """
    Execute the complete pipeline: PDFs → chunks → embeddings → FAISS.

    Returns
    -------
    dict
        Summary with keys:
        - ``total_pages``  : int
        - ``total_chunks`` : int
        - ``embedding_dim``: int
        - ``elapsed_secs`` : float
    """
    start_time: float = time.time()
    logger.info("=" * 60)
    logger.info("STARTING EMBEDDING PIPELINE")
    logger.info("=" * 60)

    # ── Phase 1: Load PDFs ───────────────────────────────────────────
    logger.info("Phase 1 ▸ Loading PDFs …")
    raw_pages: List[Dict[str, Any]] = load_all_pdfs()

    # ── Phase 2: Clean text ──────────────────────────────────────────
    logger.info("Phase 2 ▸ Cleaning text …")
    cleaned_pages: List[Dict[str, Any]] = clean_documents(raw_pages)

    # ── Phase 3: Chunk documents ─────────────────────────────────────
    logger.info("Phase 3 ▸ Chunking documents …")
    chunks: List[Dict[str, Any]] = chunk_documents(cleaned_pages)

    if not chunks:
        logger.error("No chunks were created. Aborting pipeline.")
        raise ValueError("Pipeline produced zero chunks – check your PDFs.")

    # Save chunk metadata for later inspection
    save_pickle(chunks, str(CHUNKS_PKL_PATH))

    # ── Phase 4: Generate embeddings ─────────────────────────────────
    logger.info("Phase 4 ▸ Generating embeddings …")
    texts: List[str] = [chunk["text"] for chunk in chunks]

    emb_model = EmbeddingModel()
    embeddings: np.ndarray = emb_model.encode(texts)

    # Save raw embeddings as pickle
    save_pickle(embeddings, str(EMBEDDINGS_PKL_PATH))

    # ── Phase 5: Build FAISS index ───────────────────────────────────
    logger.info("Phase 5 ▸ Building FAISS index …")
    faiss_mgr = FAISSManager(dimension=emb_model.dimension)
    faiss_mgr.build_index(embeddings, chunks)

    elapsed: float = round(time.time() - start_time, 2)
    summary: Dict[str, Any] = {
        "total_pages": len(raw_pages),
        "total_chunks": len(chunks),
        "embedding_dim": emb_model.dimension,
        "elapsed_secs": elapsed,
    }

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.2f seconds", elapsed)
    logger.info("Pages: %d  │  Chunks: %d  │  Dim: %d",
                summary["total_pages"],
                summary["total_chunks"],
                summary["embedding_dim"])
    logger.info("=" * 60)
    return summary


# Allow running this module directly: python -m embeddings.create_embeddings
if __name__ == "__main__":
    run_embedding_pipeline()
