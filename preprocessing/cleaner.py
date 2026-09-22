"""
cleaner.py – Phase 2: Text Cleaning
=====================================
Cleans raw text extracted from PDF pages by removing:

* Extra / trailing spaces
* Repeated empty lines
* Common headers & footers (page numbers, chapter headings at top)
* Duplicate consecutive lines
* Non-printable characters

Author  : B.Tech CSE-AI Student Project
"""

import re
from typing import Any, Dict, List

from utils.logger import logger


def clean_text(raw_text: str) -> str:
    """
    Clean a single block of raw PDF text.

    Parameters
    ----------
    raw_text : str
        Raw text from one PDF page.

    Returns
    -------
    str
        Cleaned text with noise removed.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text: str = raw_text

    # 1. Remove non-printable / control characters (keep newlines & tabs)
    text = re.sub(r"[^\S\n\t]+", " ", text)

    # 2. Remove lines that are just page numbers (e.g. "  42  " or "Page 42")
    text = re.sub(r"(?m)^\s*(Page\s*)?\d{1,4}\s*$", "", text, flags=re.IGNORECASE)

    # 3. Remove common header / footer patterns
    #    e.g. "Chapter 5 – Cardiology" at the very top of a page
    text = re.sub(r"(?m)^(Chapter\s+\d+.*|CHAPTER\s+\d+.*)$", "", text)

    # 4. Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # 6. Remove duplicate consecutive lines
    lines: List[str] = text.split("\n")
    deduped_lines: List[str] = []
    prev_line: str = ""
    for line in lines:
        stripped = line.strip()
        if stripped != prev_line:
            deduped_lines.append(line)
        prev_line = stripped
    text = "\n".join(deduped_lines)

    # 7. Final strip
    text = text.strip()

    return text


def clean_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply ``clean_text`` to every document dict in the list.

    Parameters
    ----------
    documents : list[dict]
        Each dict must have a ``"text"`` key with the raw page text.

    Returns
    -------
    list[dict]
        Same dicts but with cleaned text.  Empty pages are dropped.
    """
    cleaned: List[Dict[str, Any]] = []

    for doc in documents:
        cleaned_text = clean_text(doc["text"])
        if cleaned_text:  # Drop pages that became empty after cleaning
            cleaned.append({
                "book_name": doc["book_name"],
                "page_number": doc["page_number"],
                "text": cleaned_text,
                "metadata": doc.get("metadata", {}),
            })

    removed_count: int = len(documents) - len(cleaned)
    if removed_count > 0:
        logger.info(
            "Cleaning: removed %d empty pages. %d pages remain.",
            removed_count,
            len(cleaned),
        )
    else:
        logger.info("Cleaning complete. %d pages retained.", len(cleaned))

    return cleaned
