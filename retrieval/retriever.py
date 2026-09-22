"""
retriever.py – Phase 6: Base Retriever
========================================
Converts the user's question into an embedding and performs a
straight FAISS similarity search.  This is the building block that
``adaptive_retriever.py`` wraps with adaptive logic.

Author  : B.Tech CSE-AI Student Project
"""

from typing import Any, Dict, List

import numpy as np

from embeddings.embedding_model import EmbeddingModel
from vectordb.faiss_manager import FAISSManager
from utils.logger import logger


class Retriever:
    """
    Base retriever: embed a question → search FAISS → return results.

    Attributes
    ----------
    embedding_model : EmbeddingModel
        Loaded sentence-transformer model.
    faiss_manager : FAISSManager
        Loaded (or loadable) FAISS index.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        faiss_manager: FAISSManager,
    ) -> None:
        """
        Parameters
        ----------
        embedding_model : EmbeddingModel
            An already-initialised EmbeddingModel instance.
        faiss_manager : FAISSManager
            A FAISSManager that either has an index loaded or can load one.
        """
        self.embedding_model: EmbeddingModel = embedding_model
        self.faiss_manager: FAISSManager = faiss_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Embed the *query* and return the ``top_k`` most similar chunks.

        Parameters
        ----------
        query : str
            User's natural-language question.
        top_k : int
            How many chunks to return.

        Returns
        -------
        list[dict]
            Each dict has: score, chunk_id, book_name, page_number, text.
        """
        logger.info("Retrieving top-%d chunks for: '%s'", top_k, query[:80])

        # Phase 6: Convert question → embedding
        query_vec: np.ndarray = self.embedding_model.encode(
            query,
            show_progress=False,
        )
        # Reshape to (1, dim)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # Search FAISS
        results: List[Dict[str, Any]] = self.faiss_manager.search(
            query_vec, top_k=top_k
        )

        logger.info(
            "Retrieved %d chunks (best score: %.4f).",
            len(results),
            results[0]["score"] if results else 0.0,
        )
        return results
