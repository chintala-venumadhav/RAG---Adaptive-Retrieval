"""
faiss_manager.py – Phase 5: FAISS Vector Database Manager
==========================================================
Provides a manager class that can:

* **Create** a FAISS index from embeddings
* **Save** the index + metadata to disk
* **Load** a previously saved index
* **Search** for the ``top_k`` most similar vectors

The index uses **Inner Product** on L2-normalised vectors, which is
mathematically equivalent to **cosine similarity** and is the fastest
option in FAISS for CPU.

Author  : B.Tech CSE-AI Student Project
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from utils.config import FAISS_INDEX_PATH, FAISS_METADATA_PATH
from utils.logger import logger


class FAISSManager:
    """
    Manage a FAISS flat index backed by inner product (cosine similarity).

    Attributes
    ----------
    dimension : int
        Dimensionality of the embedding vectors.
    index : faiss.IndexFlatIP | None
        The in-memory FAISS index.
    metadata : list[dict]
        Parallel list of chunk metadata (one dict per vector).
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: Path = FAISS_INDEX_PATH,
        metadata_path: Path = FAISS_METADATA_PATH,
    ) -> None:
        """
        Initialise the manager.

        Parameters
        ----------
        dimension : int
            Length of each embedding vector (384 for all-MiniLM-L6-v2).
        index_path : Path
            Where to save / load the FAISS binary index.
        metadata_path : Path
            Where to save / load the metadata pickle.
        """
        self.dimension: int = dimension
        self.index_path: Path = index_path
        self.metadata_path: Path = metadata_path

        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    # ─────────────────────────── BUILD ───────────────────────────────

    def build_index(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Build a new FAISS index from scratch and save it to disk.

        The vectors are L2-normalised so that inner product == cosine
        similarity.

        Parameters
        ----------
        embeddings : np.ndarray
            Shape ``(n, dimension)`` float32 matrix.
        metadata : list[dict]
            One metadata dict per embedding (same order).
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"Mismatch: {embeddings.shape[0]} embeddings vs "
                f"{len(metadata)} metadata entries."
            )

        logger.info(
            "Building FAISS index: %d vectors of dim %d",
            embeddings.shape[0],
            embeddings.shape[1],
        )

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.metadata = metadata

        self.save_index()
        logger.info("FAISS index built and saved (%d vectors).", self.index.ntotal)

    # ─────────────────────────── SAVE ────────────────────────────────

    def save_index(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save.")
            return

        # Ensure parent directories exist
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as fh:
            pickle.dump(self.metadata, fh)

        logger.info("Index saved to %s", self.index_path)

    # ─────────────────────────── LOAD ────────────────────────────────

    def load_index(self) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Returns
        -------
        bool
            ``True`` if loaded successfully, ``False`` otherwise.
        """
        if not self.index_path.exists():
            logger.error("FAISS index file not found: %s", self.index_path)
            return False
        if not self.metadata_path.exists():
            logger.error("Metadata file not found: %s", self.metadata_path)
            return False

        try:
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as fh:
                self.metadata = pickle.load(fh)
            self.dimension = self.index.d
            logger.info(
                "FAISS index loaded: %d vectors, dim %d",
                self.index.ntotal,
                self.dimension,
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    # ─────────────────────────── SEARCH ──────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the ``top_k`` most similar vectors to the query.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape ``(1, dimension)`` float32 vector **before** normalisation
            (this method normalises it internally).
        top_k : int
            Number of nearest neighbours to return.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``score``    (float) – cosine similarity in [0, 1]
            - ``chunk_id`` (int)
            - ``book_name``(str)
            - ``page_number`` (int)
            - ``text``     (str)
        """
        if self.index is None:
            loaded = self.load_index()
            if not loaded:
                raise RuntimeError(
                    "No FAISS index available. Please build embeddings first."
                )

        # Normalise query for cosine similarity
        query = query_embedding.astype(np.float32).copy()
        faiss.normalize_L2(query)

        # Clamp top_k to the number of vectors in the index
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query, top_k)

        results: List[Dict[str, Any]] = []
        for rank in range(len(indices[0])):
            idx: int = int(indices[0][rank])
            if idx == -1:
                continue  # FAISS returns -1 when fewer than top_k results exist

            meta: Dict[str, Any] = self.metadata[idx]
            results.append({
                "score": float(distances[0][rank]),
                "chunk_id": meta.get("chunk_id", idx),
                "book_name": meta.get("book_name", "Unknown"),
                "page_number": meta.get("page_number", 0),
                "text": meta.get("text", ""),
            })

        return results

    # ─────────────────────────── STATUS ──────────────────────────────

    def is_ready(self) -> bool:
        """Return ``True`` if an index is loaded in memory."""
        return self.index is not None and self.index.ntotal > 0

    def index_exists_on_disk(self) -> bool:
        """Return ``True`` if saved index files exist."""
        return self.index_path.exists() and self.metadata_path.exists()
