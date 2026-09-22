"""
config.py – Centralized Configuration for the Adaptive RAG Project
====================================================================
This file stores every path, threshold, and model name used across the
project.  Changing a value here automatically propagates everywhere.

Author  : B.Tech CSE-AI Student Project
Purpose : Single source of truth for all settings
"""

import os
from pathlib import Path

# ─────────────────────────────── PATHS ───────────────────────────────

# Base directory of the project (parent of utils/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR: Path = BASE_DIR / "data"
BOOKS_DIR: Path = DATA_DIR / "books"            # Place your medical PDFs here
EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"  # Pickle files for embeddings
FAISS_DIR: Path = DATA_DIR / "faiss"            # FAISS index files

# Ensure all data directories exist on import
for _dir in [BOOKS_DIR, EMBEDDINGS_DIR, FAISS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# Specific file paths
FAISS_INDEX_PATH: Path = FAISS_DIR / "medical_index.faiss"
FAISS_METADATA_PATH: Path = FAISS_DIR / "medical_metadata.pkl"
EMBEDDINGS_PKL_PATH: Path = EMBEDDINGS_DIR / "chunk_embeddings.pkl"
CHUNKS_PKL_PATH: Path = EMBEDDINGS_DIR / "chunks_metadata.pkl"

# Log file
LOG_FILE: Path = BASE_DIR / "adaptive_rag.log"

# ──────────────────────────── CHUNKING ───────────────────────────────

CHUNK_SIZE: int = 1500       # Characters per chunk
CHUNK_OVERLAP: int = 300    # Overlapping characters between consecutive chunks

# ─────────────────────── EMBEDDING MODEL ─────────────────────────────
# Lightweight model that runs fast on CPU (≈80 MB)
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"

# ──────────────────────── RERANKER MODEL ─────────────────────────────
# Cross-encoder for re-ranking retrieved chunks (≈80 MB, CPU-friendly)
RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ────────────────────── RETRIEVAL SETTINGS ───────────────────────────

TOP_K_RETRIEVAL: int = 80    # Retrieve Top-50 candidate chunks from FAISS

# Cosine similarity thresholds (FAISS inner-product on L2-normalised vectors)
CONFIDENCE_HIGH: float = 0.70    # Above this → High confidence, stop expanding
CONFIDENCE_MEDIUM: float = 0.60  # Above this → Medium confidence
# Below CONFIDENCE_MEDIUM → Low confidence, keep expanding

# Number of best chunks to keep after re-ranking
RERANKER_TOP_K: int = 10
RERANK_THRESHOLD: float = -5.0    # Discard chunks with cross-encoder score below this

# ────────────────────── LLM (OLLAMA) SETTINGS ───────────────────────

OLLAMA_BASE_URL: str = "http://localhost:11434"
LLM_TEMPERATURE: float = 0.0             # Low temperature for factual answers
LLM_TOP_P: float = 0.50
LLM_TOP_K: int = 20
LLM_REPEAT_PENALTY: float = 1.1
LLM_NUM_PREDICT: int = 100
# Ollama's own default context window is only 2048 tokens if not told
# otherwise, which silently truncates/corrupts prompts built up to
# MAX_CONTEXT_LENGTH (25000 chars, ~8K context as noted below). This was
# the root cause of empty/garbled generations on longer contexts.
LLM_NUM_CTX: int = 8192                  # Matches llama3.2:3b's 8K context, per MAX_CONTEXT_LENGTH below
LLM_TIMEOUT: int = 900                   # Seconds to wait for Ollama response


# ────────────────────── STREAMLIT SETTINGS ───────────────────────────

APP_TITLE: str = "⚕️ Medical Adaptive RAG"
APP_ICON: str = "⚕️"

DEBUG_MODE: bool = False
MAX_CONTEXT_LENGTH: int = 25000   # Max prompt length in characters (safe limit for llama3.2:3b 8K context)
