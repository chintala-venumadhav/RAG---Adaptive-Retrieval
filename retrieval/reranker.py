"""
reranker.py – Phase 10: Cross-Encoder Re-Ranking
==================================================
Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` to re-score each
(query, chunk) pair.  The cross-encoder reads both texts jointly,
so it is **much** more accurate than the bi-encoder similarity from
FAISS – at the cost of being slower (which is fine because we only
re-rank a small number of chunks).

Author  : B.Tech CSE-AI Student Project
"""

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

from utils.config import RERANKER_MODEL_NAME, RERANKER_TOP_K, RERANK_THRESHOLD
from utils.logger import logger


class Reranker:
    """
    Re-rank retrieved chunks using a Cross-Encoder model.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    model : CrossEncoder
        The loaded cross-encoder model.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_NAME) -> None:
        """
        Load the cross-encoder model into memory.

        Parameters
        ----------
        model_name : str
            HuggingFace model name or local path.
        """
        self.model_name: str = model_name
        logger.info("Loading re-ranker model: %s …", model_name)

        try:
            self.model: CrossEncoder = CrossEncoder(model_name)
        except Exception as exc:
            logger.error("Failed to load re-ranker: %s", exc)
            raise RuntimeError(
                f"Could not load re-ranker '{model_name}'."
            ) from exc

        logger.info("Re-ranker model loaded.")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = RERANKER_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank *chunks* by their relevance to *query*.

        Parameters
        ----------
        query : str
            The user's question.
        chunks : list[dict]
            Retrieved chunks – each dict must contain a ``"text"`` key.
        top_k : int
            Number of best chunks to keep after re-ranking.

        Returns
        -------
        list[dict]
            The top-k chunks sorted by re-rank score (highest first).
            Each dict gains an extra ``"rerank_score"`` key.
        """
        if not chunks:
            return []

        logger.info("Re-ranking %d chunks …", len(chunks))

        # Build (query, chunk_text) pairs as tuples, not lists
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs at once
        scores = self.model.predict(pairs) # type: ignore
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif isinstance(scores, (float, int)):
            scores = [scores]
        else:
            scores = [float(scores)]

        # Attach score to each chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score

        # Filter out chunks below the threshold
        filtered_chunks = [c for c in chunks if c["rerank_score"] >= RERANK_THRESHOLD]

        # Sort descending by re-rank score
        ranked: List[Dict[str, Any]] = sorted(
            filtered_chunks,
            key=lambda c: c["rerank_score"],
            reverse=True,
        )
        
        # Keep top_k, but ensure we don't return weak candidates
        final_ranked = ranked[:top_k]

        logger.info(
            "Re-ranking done. Best rerank score: %.4f",
            final_ranked[0]["rerank_score"] if final_ranked else 0.0,
        )
        return final_ranked
