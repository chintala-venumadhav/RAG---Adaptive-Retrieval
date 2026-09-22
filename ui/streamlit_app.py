"""
streamlit_app.py – Phase 13: Professional Streamlit Interface
==============================================================================
A dark-themed, feature-rich Streamlit UI for the Adaptive RAG system.

Features
--------
* Upload PDFs & save to ``data/books/``
* Generate embeddings with a single button click
* Select generation model from sidebar
* View retrieved chunks, similarity, re-rank scores, and confidence badges
* Full offline functionality

Run with
--------
    streamlit run ui/streamlit_app.py

Author  : B.Tech CSE-AI Student Project
"""

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# pyrefly: ignore [missing-import]
import streamlit as st

# ── Fix Python path so imports work when launched via `streamlit run` ─
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.create_embeddings import run_embedding_pipeline
from embeddings.embedding_model import EmbeddingModel
from llm.generator import Generator
from retrieval.adaptive_retriever import AdaptiveRetriever
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from utils.config import (
    APP_ICON,
    APP_TITLE,
    BOOKS_DIR,
    FAISS_DIR,
    FAISS_INDEX_PATH,
    EMBEDDINGS_DIR,
    DEBUG_MODE
)
from vectordb.faiss_manager import FAISSManager
import json
from utils.logger import logger

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Medical Adaptive RAG",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS – DARK PREMIUM THEME
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── OFFLINE FONT STACK ──────────────────────────────────────────
       No external Google Fonts import.  We use 'Segoe UI' which is
       pre-installed on every Windows 10/11 machine, with standard
       system-font fallbacks for other OSes.  This keeps the UI
       fully offline — zero HTTPS requests.
       ────────────────────────────────────────────────────────────── */

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #e0e0e0;
        font-size: 2rem;
        margin: 0;
    }
    .main-header p {
        color: #a0a0c0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Chunk card */
    .chunk-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #00d2ff;
        transition: transform 0.2s;
    }
    .chunk-card:hover {
        transform: translateX(4px);
        border-left-color: #3a7bd5;
    }
    .chunk-card .source-label {
        color: #00d2ff;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .chunk-card .scores {
        color: #8888aa;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .chunk-card .chunk-text {
        color: #c8c8e0;
        font-size: 0.88rem;
        margin-top: 8px;
        line-height: 1.5;
    }

    /* Confidence badges */
    .badge-high {
        background-color: #00c853;
        color: #ffffff;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #ff9100;
        color: #ffffff;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #ff1744;
        color: #ffffff;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Sidebar tweaks */
    .sidebar-status {
        padding: 10px;
        border-radius: 8px;
        margin: 6px 0;
        font-weight: 500;
    }
    .status-ok {
        background-color: rgba(0,200,83,0.15);
        color: #00c853;
        border: 1px solid rgba(0,200,83,0.3);
    }
    .status-missing {
        background-color: rgba(255,23,68,0.12);
        color: #ff1744;
        border: 1px solid rgba(255,23,68,0.3);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []

if "app_started" not in st.session_state:
    pdf_files_global = sorted(BOOKS_DIR.glob("*.pdf"))
    pdf_filenames = [p.name for p in pdf_files_global]
    pdf_count = len(pdf_filenames)
    
    faiss_mgr_global = FAISSManager()
    
    # Check if index exists and if it matches current PDFs
    needs_rebuild = False
    
    if faiss_mgr_global.index_exists_on_disk():
        faiss_mgr_global.load_index()
        # Find which books are in the index
        indexed_books = set([meta.get("book_name") for meta in faiss_mgr_global.metadata])
        
        # Check if they match current pdfs exactly
        # Note: metadata 'book_name' might have '.pdf' or not. Let's compare safely.
        indexed_books_normalized = {b.replace(".pdf", "") for b in indexed_books}
        current_pdfs_normalized = {p.replace(".pdf", "") for p in pdf_filenames}
        
        if indexed_books_normalized != current_pdfs_normalized:
            needs_rebuild = True
    else:
        needs_rebuild = True
        
    if needs_rebuild and pdf_count > 0:
        if DEBUG_MODE:
            logger.info("--- Change detected in PDFs. Rebuilding index automatically ---")
        run_embedding_pipeline()
        faiss_mgr_global = FAISSManager()
        if faiss_mgr_global.index_exists_on_disk():
            faiss_mgr_global.load_index()
            
    num_embeddings = faiss_mgr_global.index.ntotal if faiss_mgr_global.index else 0
    num_chunks = len(faiss_mgr_global.metadata)
    faiss_status = "Ready" if faiss_mgr_global.is_ready() else "Not Ready"

    st.session_state.pdf_filenames = pdf_filenames
    st.session_state.pdf_count = pdf_count
    st.session_state.num_chunks = num_chunks
    st.session_state.num_embeddings = num_embeddings
    st.session_state.app_started = True

# ═══════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model() -> EmbeddingModel:
    """Load the sentence-transformer once and cache it."""
    return EmbeddingModel()


@st.cache_resource(show_spinner="Loading re-ranker model…")
def load_reranker() -> Reranker:
    """Load the cross-encoder once and cache it."""
    return Reranker()


@st.cache_resource(show_spinner="Loading FAISS index…")
def load_faiss_manager() -> Optional[FAISSManager]:
    """Load FAISS index from disk (returns None if missing)."""
    mgr = FAISSManager()
    if mgr.index_exists_on_disk():
        mgr.load_index()
        return mgr
    return None


@st.cache_resource(show_spinner="Connecting to Ollama…")
def load_generator() -> Generator:
    """Load the LLM Generator connection once and cache it."""
    return Generator()


def get_retrieval_system():
    """
    Assemble the full retrieval pipeline from cached components.
    Returns (adaptive_retriever, generator, error_message).
    """
    emb_model = load_embedding_model()
    reranker = load_reranker()
    faiss_mgr = load_faiss_manager()

    if faiss_mgr is None or not faiss_mgr.is_ready():
        return None, None, "faiss_missing"

    base_retriever = Retriever(emb_model, faiss_mgr)
    adaptive = AdaptiveRetriever(base_retriever, reranker)
    gen = load_generator()

    return adaptive, gen, None


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown("---")

    # ── Status Panel (Internal Checks Only) ──────────────────────────
    # PDFs
    pdf_files = sorted(BOOKS_DIR.glob("*.pdf"))

    # FAISS
    _faiss_exists = FAISS_INDEX_PATH.exists()

    # Ollama
    try:
        gen_check = load_generator()
        ollama_ok = gen_check.check_ollama_running()
    except Exception:
        ollama_ok = False

    # ── Hardcoded Generation Model ───────────────────────────────────
    selected_model = "llama3.2:3b"








# ═══════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>⚕️ Medical Adaptive RAG</h1>
    <p>Ask questions from your medical textbooks — fully offline, powered by Adaptive Retrieval</p>
</div>
""", unsafe_allow_html=True)

# ── Check readiness ──────────────────────────────────────────────────
pdf_files_check = sorted(BOOKS_DIR.glob("*.pdf"))
if not pdf_files_check:
    st.error("No medical PDF documents were found in the local library.")
    st.stop()

if not FAISS_INDEX_PATH.exists():
    st.error("Knowledge base is not initialized. Please build embeddings.")
    st.stop()

# Load the retrieval system
adaptive_retriever, generator, init_error = get_retrieval_system()

if init_error == "faiss_missing":
    st.error("Knowledge base is not initialized. Please build embeddings.")
    st.stop()

# ── Chat View ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════
# CHAT VIEW
# ═══════════════════════════════════════════════════════════════════════
with st.container():
    # ── Chat History ─────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat Input ───────────────────────────────────────────────────────
    user_query: Optional[str] = st.chat_input(
        "Ask a medical question (e.g. 'What are the symptoms of diabetes?')"
    )

    if user_query:
        # Validate input
        if len(user_query.strip()) < 3:
            st.warning("Please enter a valid question (at least 3 characters).")
            st.stop()

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # ── Process answer ───────────────────────────────────────────────
        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=True)

            total_start = time.time()

            # Step 1: Adaptive Retrieval (Identical across all models)
            status.write("🔍 Retrieving relevant context…")
            retrieval_start = time.time()
            search_result: Dict[str, Any] = adaptive_retriever.adaptive_search(user_query)
            retrieval_time = round(time.time() - retrieval_start, 2)

            chunks: List[Dict[str, Any]] = search_result["chunks"]
            confidence: str = search_result["confidence"]
            complexity: str = search_result["complexity"]
            k_used: int = search_result["k_expanded_to"]
            best_score: float = search_result["best_score"]

            if not chunks:
                logger.warning("No chunks were retrieved from the document store.")

            if DEBUG_MODE:
                logger.info(f"Number of retrieved chunks for each query: {len(chunks)}")

            # Prepare citations string for logging
            citations = "; ".join([
                f"{c.get('book_name', 'Unknown')}(p{c.get('page_number', '?')})" 
                for c in chunks
            ])

            # Step 2: LLM Generation (Ablation Target)
            status.write(f"🧠 Generating answer with **{selected_model}**…")
            llm_result: Dict[str, Any] = generator.generate(user_query, chunks, selected_model)
            answer_text: str = llm_result["answer"]
            llm_time: float = llm_result["elapsed_secs"]
            prompt_time: float = llm_result.get("prompt_time", 0.0)
            tokens_gen: int = llm_result.get("tokens", 0)
            prompt_length: int = llm_result.get("prompt_length", 0)


            if "I could not find the answer" in answer_text or "I couldn't find sufficient information" in answer_text or "Insufficient information" in answer_text:
                confidence = "Low"

            # ── Extractor LLM Call for Category & Treatment ──────────────
            extract_prompt = (
                f"You are a medical classifier.\n"
                f"Context: {citations}\n\n"
                f"Based ONLY on the retrieved context chunks above, output a JSON object with two keys:\n"
                f'1. "category": Choose exactly one from [Anatomy, Oncology, Pathology, Diagnosis, Treatment, Surgery, Radiotherapy, Chemotherapy, Pharmacology, Immunology, Prevention, Prognosis, General Medicine, Other].\n'
                f'2. "treatment": If the context contains treatment information, provide a concise summary. Otherwise, write "Not Applicable".\n'
                f"Respond ONLY with valid JSON. Do not include markdown code blocks or any other text.\n"
                f"JSON:\n"
            )
            
            status.write("🧩 Extracting metadata…")
            try:
                extract_res = generator.generate(extract_prompt, chunks, selected_model)
                extract_text = extract_res["answer"].strip()
                # Clean up any markdown blocks if the LLM adds them
                if extract_text.startswith("```json"):
                    extract_text = extract_text[7:]
                if extract_text.startswith("```"):
                    extract_text = extract_text[3:]
                if extract_text.endswith("```"):
                    extract_text = extract_text[:-3]
                    
                meta_json = json.loads(extract_text.strip())
                category = meta_json.get("category", "Other")
                treatment = meta_json.get("treatment", "Not Applicable")
            except Exception as e:
                if DEBUG_MODE:
                    logger.error(f"Metadata extraction failed: {e}")
                category = "Other"
                treatment = "Not Applicable"

            total_time = round(time.time() - total_start, 2)
            status.update(label=f"Done in {total_time}s", state="complete")

            # --- Detailed Console Logging (Req 10 and Req 13) ---
            if DEBUG_MODE:
                logger.info("--- Retrieval & Generation Validation ---")
                logger.info(f"PDFs loaded: {st.session_state.get('pdf_filenames', [])}")
                logger.info(f"Number of chunks: {st.session_state.get('num_chunks', 0)}")
                logger.info(f"FAISS vector count: {st.session_state.get('num_embeddings', 0)}")
                logger.info(f"Retrieval time: {search_result.get('retrieval_time', 0.0)}s")
                logger.info(f"Reranking time: {search_result.get('rerank_time', 0.0)}s")
                logger.info(f"Prompt building time: {prompt_time}s")
                logger.info(f"LLM generation time: {llm_time}s")
                logger.info(f"Total response time: {total_time}s")
                logger.info(f"Retrieved pages: {[c.get('page_number') for c in chunks]}")
                logger.info(f"Retrieved similarity scores: {[c.get('score') for c in chunks]}")
                logger.info("-----------------------------------------")

            # ── Display Answer ───────────────────────────────────────────
            st.markdown(f"**Answer:**\n\n{answer_text}")
            
            if chunks and not ("Insufficient information" in answer_text):
                from collections import defaultdict
                sources_grouped = defaultdict(set)
                for c in chunks:
                    if c.get('book_name') and c.get('book_name') != 'None' and c.get('score', 0.0) >= 0.0:
                        sources_grouped[c.get('book_name')].add(str(c.get('page_number', '0')))
                
                if sources_grouped:
                    sources_md = "\n\n**Sources**\n"
                    for book, pages in sources_grouped.items():
                        sources_md += f"\n*{book}*\n"
                        sorted_pages = sorted(list(pages), key=lambda x: int(x.split('-')[0]) if x.replace('-','').isdigit() else 0)
                        for page in sorted_pages:
                            sources_md += f"- Pages {page}\n"
                    st.markdown(sources_md)
                    answer_text += sources_md

            # ── Metrics Row ──────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)

            badge_class = f"badge-{confidence.lower()}"
            m1.markdown(
                f'Confidence: <span class="{badge_class}">{confidence}</span>',
                unsafe_allow_html=True,
            )
            m2.metric("Retrieval", f"{retrieval_time}s")
            m3.metric("LLM Time", f"{llm_time}s")
            m4.metric("Total", f"{total_time}s")

            # ── Category & Treatment Row ─────────────────────────────────
            st.markdown(f"**Category:** {category}")
            st.markdown(f"**Treatment:** {treatment}")
            
            # ── Restored Retrieved Sources ───────────────────────────────
            with st.expander("📚 View Retrieved Sources", expanded=False):
                if len(chunks) == 1 and chunks[0].get("score") == 0.0:
                    st.info("No sources were retrieved for this query.")
                else:
                    for i, c in enumerate(chunks):
                        book = c.get('book_name', 'Unknown')
                        page = c.get('page_number', '?')
                        c_id = c.get('chunk_id', 'N/A')
                        sim = c.get('score', 0.0)
                        rr = c.get('rerank_score', 0.0)
                        text_preview = c.get('text', '')[:150].replace('\n', ' ') + '...'
                        
                        st.markdown(f"""
                        <div class="chunk-card">
                            <div class="source-label">Source {i+1} | {book} (Page {page})</div>
                            <div class="scores">ID: {c_id} | Similarity: {sim:.4f} | Reranker: {rr:.4f}</div>
                            <div class="chunk-text">{text_preview}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Save to history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_text}
            )

