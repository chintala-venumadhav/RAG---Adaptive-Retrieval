"""
pdf_loader.py – Phase 1: Read Medical PDFs
============================================
Uses PyMuPDF (imported as ``fitz``) to read every PDF inside
``data/books/``.  For each page it extracts:

* Book Name  – derived from the file name
* Page Number
* Raw Text
* Metadata   – author / title from PDF info dict, if available

Author  : B.Tech CSE-AI Student Project
"""

from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF

from utils.config import BOOKS_DIR
from utils.logger import logger


def load_single_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Read one PDF and return a list of page-level dictionaries.

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the PDF file.

    Returns
    -------
    list[dict]
        Each dict has keys: book_name, page_number, text, metadata.

    Raises
    ------
    RuntimeError
        If the PDF cannot be opened (e.g. corrupted).
    """
    pages: List[Dict[str, Any]] = []
    book_name: str = pdf_path.stem  # File name without extension

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.error("Failed to open PDF '%s': %s", pdf_path.name, exc)
        raise RuntimeError(f"Corrupted or unreadable PDF: {pdf_path.name}") from exc

    # Extract document-level metadata (author, title, etc.)
    pdf_metadata: Dict[str, str] = {}
    try:
        info = doc.metadata
        if info:
            pdf_metadata = {
                "title": info.get("title", ""),
                "author": info.get("author", ""),
                "subject": info.get("subject", ""),
            }
    except Exception:
        pass  # Metadata is optional; proceed even if it fails

    # Iterate through every page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        raw_text: str = page.get_text("text")  # plain-text extraction

        pages.append({
            "book_name": book_name,
            "page_number": page_num + 1,   # 1-indexed for readability
            "text": raw_text,
            "metadata": pdf_metadata,
        })

    doc.close()
    logger.info(
        "Loaded '%s': %d pages extracted.",
        pdf_path.name,
        len(pages),
    )
    return pages


def load_all_pdfs(books_dir: Path = BOOKS_DIR) -> List[Dict[str, Any]]:
    """
    Scan ``data/books/`` and load every PDF found.

    Parameters
    ----------
    books_dir : Path
        Directory to scan (defaults to BOOKS_DIR from config).

    Returns
    -------
    list[dict]
        Combined list of page-level dicts from all PDFs.

    Raises
    ------
    FileNotFoundError
        If the books directory does not exist.
    ValueError
        If no PDFs are found inside the directory.
    """
    if not books_dir.exists():
        raise FileNotFoundError(f"Books directory not found: {books_dir}")

    pdf_files: List[Path] = sorted(books_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", books_dir)
        raise ValueError(
            f"No PDF files in {books_dir}. "
            "Please place your medical textbooks there."
        )

    logger.info("Found %d PDF(s) in '%s'.", len(pdf_files), books_dir)

    all_pages: List[Dict[str, Any]] = []
    for pdf_path in pdf_files:
        try:
            pages = load_single_pdf(pdf_path)
            all_pages.extend(pages)
        except RuntimeError as exc:
            # Log and skip corrupted PDFs instead of crashing
            logger.error("Skipping corrupted PDF: %s", exc)

    logger.info("Total pages extracted across all books: %d", len(all_pages))
    return all_pages
