"""
embedding_model.py – Phase 4: Embedding Model Wrapper
=======================================================
Wraps ``sentence-transformers/all-MiniLM-L6-v2`` behind a simple class
that can encode texts into dense vectors.  The model is lightweight
(≈80 MB) and runs comfortably on a CPU-only laptop.

Usage
-----
    from embeddings.embedding_model import EmbeddingModel

    model = EmbeddingModel()
    vectors = model.encode(["What is diabetes?", "Symptoms of flu"])

Author  : B.Tech CSE-AI Student Project
"""

from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import EMBEDDING_MODEL_NAME
from utils.logger import logger


class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier.
    model : SentenceTransformer
        The loaded model ready for inference.
    dimension : int
        Length of the output embedding vectors.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """
        Load the sentence-transformer model into memory.

        Parameters
        ----------
        model_name : str
            HuggingFace model name or local path.
        """
        self.model_name: str = model_name
        logger.info("Loading embedding model: %s …", model_name)

        try:
            self.model: SentenceTransformer = SentenceTransformer(model_name)
        except Exception as exc:
            logger.error("Failed to load embedding model: %s", exc)
            raise RuntimeError(
                f"Could not load embedding model '{model_name}'. "
                "Make sure you have an internet connection for the first download."
            ) from exc

        # Determine vector dimension from a dummy encoding
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded. Dimension: %d", self.dimension
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Convert text(s) into embedding vectors.

        Parameters
        ----------
        texts : str | list[str]
            Single string or list of strings to embed.
        batch_size : int
            Texts processed per batch (higher = faster but more RAM).
        show_progress : bool
            Whether to show a progress bar in the console.

        Returns
        -------
        np.ndarray
            2-D array of shape ``(n_texts, dimension)`` with dtype float32.
        """
        if isinstance(texts, str):
            texts = [texts]

        logger.info("Encoding %d text(s) …", len(texts))

        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Ensure float32 for FAISS compatibility
        embeddings = embeddings.astype(np.float32)
        logger.info("Encoding complete. Shape: %s", embeddings.shape)
        return embeddings
