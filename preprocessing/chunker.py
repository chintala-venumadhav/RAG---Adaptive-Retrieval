"""
chunker.py – Phase 3: Smart Chunking
======================================
Splits cleaned page text into smaller, overlapping chunks using
LangChain's ``RecursiveCharacterTextSplitter``.

Each chunk carries its own metadata:
* ``chunk_id``   – unique integer ID
* ``book_name``  – which PDF it came from
* ``page_number``– original page number

Settings (from config.py):
* Chunk size  : 500 characters
* Overlap     : 100 characters

Author  : B.Tech CSE-AI Student Project
"""

from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.logger import logger


def chunk_documents(
    cleaned_docs: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split every cleaned document page into smaller text chunks.

    Parameters
    ----------
    cleaned_docs : list[dict]
        Output of ``cleaner.clean_documents``.  Each dict must have
        ``book_name``, ``page_number``, and ``text`` keys.
    chunk_size : int
        Maximum characters per chunk (default from config).
    chunk_overlap : int
        Overlapping characters between consecutive chunks.

    Returns
    -------
    list[dict]
        Each dict contains:
        - ``chunk_id``    (int)  – globally unique ID
        - ``book_name``   (str)  – source PDF name
        - ``page_number`` (int)  – source page
        - ``text``        (str)  – the chunk text
    """
    logger.info(
        "Chunking %d cleaned pages (size=%d, overlap=%d)…",
        len(cleaned_docs),
        chunk_size,
        chunk_overlap,
    )

    # Initialise the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # try paragraph → sentence → word
        length_function=len,
    )

    all_chunks: List[Dict[str, Any]] = []
    chunk_id: int = 0

    for doc in cleaned_docs:
        text: str = doc["text"]
        if not text.strip():
            continue

        # Split the page text into fragments
        fragments: List[str] = splitter.split_text(text)

        for fragment in fragments:
            all_chunks.append({
                "chunk_id": chunk_id,
                "book_name": doc["book_name"],
                "page_number": doc["page_number"],
                "text": fragment,
            })
            chunk_id += 1

    logger.info("Chunking complete. Created %d chunks.", len(all_chunks))
    return all_chunks
