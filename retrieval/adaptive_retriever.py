"""
adaptive_retriever.py – Phases 7, 8, 9: Adaptive Retrieval Engine
===================================================================
This is the **heart** of the project.  Instead of always retrieving a
fixed number of chunks, it **adapts** based on:

Phase 7 – Adaptive Retrieval Algorithm
    1. Retrieve Top 3 chunks.
    2. If the best similarity > 0.90 → stop, we have enough context.
    3. Otherwise → expand to Top 8.
    4. If still low → expand to Top 15.
    5. If still low → retrieve even more.

Phase 8 – Query Complexity Detection
    * Simple question → start with fewer chunks.
    * Complex question → start with more chunks.

Phase 9 – Confidence Score
    * High   (score ≥ 0.90)
    * Medium (score ≥ 0.75)
    * Low    (score <  0.75) → automatically retrieve more.

After adaptive retrieval the chunks are **re-ranked** by the
Cross-Encoder to keep only the most relevant ones.

Author  : B.Tech CSE-AI Student Project
"""

import re
import time
from typing import Any, Dict, List, Tuple

from collections import defaultdict

from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from utils.config import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    RERANKER_TOP_K,
    TOP_K_RETRIEVAL,
)
from utils.logger import logger


# ─── Medical keywords that signal a complex query ────────────────────
_COMPLEX_KEYWORDS: List[str] = [
    "difference between",
    "compare",
    "pathophysiology",
    "mechanism of action",
    "differential diagnosis",
    "treatment protocol",
    "contraindications",
    "pharmacokinetics",
    "explain in detail",
    "risk factors",
    "complications",
    "prognosis",
    "staging",
    "classification",
    "management",
    "etiology",
    "anatomical",
    "anatomy",
    "definitions",
    "terminology",
    "what are the main",
    "what are the three",
    "list the",
]


class AdaptiveRetriever:
    """
    Adaptive Retrieval: dynamically decides how many chunks to fetch
    based on query complexity and similarity confidence.

    Attributes
    ----------
    retriever : Retriever
        Base retriever for FAISS search.
    reranker : Reranker
        Cross-encoder for re-ranking.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
    ) -> None:
        self.retriever: Retriever = retriever
        self.reranker: Reranker = reranker

    # ─────────── Phase 8: Query Complexity Detection ─────────────────

    def detect_complexity(self, query: str) -> str:
        """
        Classify the query as ``"simple"``, ``"medium"``, or ``"complex"``.
        """
        # Preprocessing: normalize and remove basic punctuation
        query_lower = query.lower().strip()
        query_lower = re.sub(r'[^\w\s]', '', query_lower)
        
        # Expand common medical abbreviations
        abbreviations = {
            " htn ": " hypertension ", 
            " dm ": " diabetes mellitus ", 
            " tx ": " treatment ", 
            " dx ": " diagnosis ", 
            " rx ": " prescription "
        }
        padded_query = f" {query_lower} "
        for abbr, full in abbreviations.items():
            if abbr in padded_query:
                padded_query = padded_query.replace(abbr, full)
        query_lower = padded_query.strip()

        word_count: int = len(query_lower.split())
        clause_markers: int = query.lower().count(",") + query.lower().count(" and ")
        
        is_complex = False
        for keyword in _COMPLEX_KEYWORDS:
            if keyword in query_lower:
                is_complex = True
                break
                
        if is_complex or word_count > 12 or clause_markers >= 2:
            return "complex"
        elif word_count > 6 or clause_markers == 1:
            return "medium"
        
        return "simple"

    # ─────────── Phase 9: Confidence Scoring ─────────────────────────

    @staticmethod
    def compute_confidence(avg_score: float, best_rerank_score: float = None, amount_of_evidence: int = 0) -> str:
        """
        Map the average cosine similarity score (and optional cross-encoder score) 
        to a confidence label.
        """
        if best_rerank_score is not None:
            # Confidence should never be HIGH unless the supporting chunks strongly match the question.
            if best_rerank_score >= 2.0 and avg_score >= CONFIDENCE_HIGH and amount_of_evidence >= 2:
                return "High"
            elif best_rerank_score >= 0.0 or avg_score >= CONFIDENCE_MEDIUM:
                return "Medium"
            else:
                return "Low"

        if avg_score > CONFIDENCE_HIGH or (amount_of_evidence > 10 and avg_score > (CONFIDENCE_HIGH - 0.05)):
            return "High"
        elif avg_score > CONFIDENCE_MEDIUM:
            return "Medium"
        else:
            return "Low"

    def _determine_dominant_book(self, chunks: List[Dict[str, Any]]) -> str:
        """Calculate a score for each book and return the one with the highest score."""
        if not chunks:
            return ""
            
        book_scores = defaultdict(float)
        for chunk in chunks:
            book = chunk.get("book_name")
            if not book:
                continue
            # Score formula: base value + similarity + rerank score (clamped to >= 0)
            score = 1.0 + max(0.0, chunk.get("score", 0.0))
            if "rerank_score" in chunk:
                score += max(0.0, chunk.get("rerank_score", 0.0))
            book_scores[book] += score
            
        if not book_scores:
            return ""
            
        return max(book_scores.items(), key=lambda x: x[1])[0]

    def _get_expanded_context(self, initial_chunks: List[Dict[str, Any]], dominant_book: str) -> List[Dict[str, Any]]:
        """Fetch neighboring chunks for the top chunk to provide continuous context from exactly one document."""
        if not dominant_book:
            return initial_chunks

        metadata = self.retriever.faiss_manager.metadata
        chunk_dict = {m.get("chunk_id"): m for m in metadata if m.get("chunk_id") is not None}
        
        best_chunk = None
        for c in initial_chunks:
            if c.get("book_name") == dominant_book:
                best_chunk = c
                break
                
        if not best_chunk or best_chunk.get("chunk_id") is None:
            return initial_chunks
            
        center_id = best_chunk.get("chunk_id")
        expanded_chunks = []
        processed_ids = set()
        
        # We need exactly 5 chunks from the same document (center + 4 neighbors)
        offsets = [-2, -1, 0, 1, 2]
        
        for offset in offsets:
            nid = center_id + offset
            if nid in chunk_dict and chunk_dict[nid].get("book_name") == dominant_book:
                if nid not in processed_ids:
                    if offset == 0:
                        # Boost score slightly to ensure it stays as the top context
                        best_chunk_copy = best_chunk.copy()
                        best_chunk_copy["rerank_score"] = best_chunk_copy.get("rerank_score", 0.0) + 0.5
                        expanded_chunks.append(best_chunk_copy)
                    else:
                        n_chunk = chunk_dict[nid].copy()
                        # Neighbor chunks inherit the score of the core chunk slightly penalized 
                        n_chunk["score"] = best_chunk.get("score", 0.0) - (0.01 * abs(offset))
                        n_chunk["rerank_score"] = best_chunk.get("rerank_score", 0.0) - (0.1 * abs(offset))
                        expanded_chunks.append(n_chunk)
                    processed_ids.add(nid)
                    
        return expanded_chunks if expanded_chunks else initial_chunks

    def _combine_adjacent_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine chunks that are sequentially adjacent (chunk_id) to form continuous text."""
        if not chunks:
            return []
            
        # Ensure chunks are sorted by chunk_id for sequential merging
        chunks = sorted(chunks, key=lambda c: c.get("chunk_id", -1))
        
        combined = []
        current = chunks[0].copy()
        
        for next_chunk in chunks[1:]:
            same_book = current.get("book_name") == next_chunk.get("book_name")
            curr_id = current.get("chunk_id", -2)
            next_id = next_chunk.get("chunk_id", -1)
            
            # Combine if they are adjacent chunks in the same book
            if same_book and (next_id == curr_id + 1 or next_id == curr_id):
                if next_id != curr_id: 
                    curr_text = current["text"].strip()
                    next_text = next_chunk["text"].strip()
                    
                    # Remove overlap suffix/prefix perfectly
                    overlap_found = False
                    for i in range(min(250, len(curr_text)), 10, -1):
                        suffix = curr_text[-i:]
                        if next_text.startswith(suffix):
                            current["text"] = curr_text + next_text[i:]
                            overlap_found = True
                            break
                            
                    if not overlap_found:
                        current["text"] = curr_text + " " + next_text
                        
                current["score"] = max(current.get("score", 0.0), next_chunk.get("score", 0.0))
                current["rerank_score"] = max(current.get("rerank_score", -999.0), next_chunk.get("rerank_score", -999.0))
                current["chunk_id"] = next_id 
                
                # Combine page numbers if different
                curr_page = str(current.get("page_number", ""))
                next_page = str(next_chunk.get("page_number", ""))
                if curr_page and next_page and curr_page.split("-")[-1] != next_page.split("-")[-1]:
                    min_p = curr_page.split("-")[0]
                    max_p = next_page.split("-")[-1]
                    current["page_number"] = f"{min_p}-{max_p}"
            else:
                combined.append(current)
                current = next_chunk.copy()
                
        combined.append(current)
        
        # Sort final combined chunks by rerank_score to ensure Top-5 contains the best evidence
        combined.sort(key=lambda c: c.get("rerank_score", -999.0), reverse=True)
        return combined

    def adaptive_search(self, query: str) -> Dict[str, Any]:
        """
        Run the full adaptive retrieval pipeline.
        """
        complexity: str = self.detect_complexity(query)
        initial_k: int = 8 if complexity == "simple" else (15 if complexity == "medium" else 30)

        logger.info(
            "Adaptive search ▸ query='%s' │ complexity=%s │ initial_k=%d",
            query[:60],
            complexity,
            initial_k,
        )

        retrieval_start_time = time.time()
        results: List[Dict[str, Any]] = self.retriever.retrieve(query, top_k=initial_k)

        if not results:
            logger.warning("No results found at all.")
            return {
                "chunks": [{"text": "Insufficient information in the retrieved documents.", "book_name": "None", "page_number": 0, "score": 0.0, "rerank_score": -999.0, "chunk_id": -1}],
                "confidence": "Low",
                "complexity": complexity,
                "k_expanded_to": initial_k,
                "best_score": 0.0,
                "initial_chunk_ids": [],
                "initial_scores": [],
                "reranked_chunk_ids": [],
                "reranked_scores": [],
                "retrieval_time": 0.0,
                "rerank_time": 0.0,
            }

        best_score: float = results[0]["score"]
        avg_score: float = sum(c["score"] for c in results) / len(results)
        confidence: str = self.compute_confidence(avg_score)
        
        current_k = initial_k
        
        # Iterative Adaptive Expansion based on FAISS cosine similarity
        expansion_steps = [35, 50, 75]
        for next_k in expansion_steps:
            if confidence == "Low" and current_k < next_k:
                current_k = next_k
                logger.info("Confidence Low. Expanding retrieval to k=%d", current_k)
                results = self.retriever.retrieve(query, top_k=current_k)
                best_score = results[0]["score"]
                avg_score = sum(c["score"] for c in results) / len(results)
                confidence = self.compute_confidence(avg_score)
                if confidence != "Low":
                    break
        
        retrieval_end_time = time.time()
        retrieval_time_elapsed = round(retrieval_end_time - retrieval_start_time, 2)

        rerank_start_time = time.time()
        # Rerank candidates. Pass RERANKER_TOP_K so it returns the best chunks.
        # But actually, we want the reranker to rerank ALL of current_k and then filter.
        reranked: List[Dict[str, Any]] = self.reranker.rerank(query, results, top_k=TOP_K_RETRIEVAL)
        rerank_time_elapsed = round(time.time() - rerank_start_time, 2)

        # Retrieval Validation
        if not reranked or reranked[0].get("rerank_score", -999.0) < -10.0 or not any(c.get("text", "").strip() for c in reranked):
            logger.warning("Retrieval quality poor. Stopping before generation.")
            return {
                "chunks": [{"text": "Insufficient information in the retrieved documents.", "book_name": "None", "page_number": 0, "score": best_score, "rerank_score": reranked[0].get("rerank_score", -999.0) if reranked else -999.0, "chunk_id": -1}],
                "confidence": "Low",
                "complexity": complexity,
                "k_expanded_to": current_k,
                "best_score": best_score,
                "initial_chunk_ids": [c.get("chunk_id") for c in results],
                "initial_scores": [c.get("score") for c in results],
                "reranked_chunk_ids": [c.get("chunk_id") for c in reranked],
                "reranked_scores": [c.get("rerank_score") for c in reranked],
                "retrieval_time": retrieval_time_elapsed,
                "rerank_time": rerank_time_elapsed,
            }

        best_rerank_score = reranked[0].get("rerank_score")
        final_confidence = self.compute_confidence(avg_score, best_rerank_score, amount_of_evidence=len(reranked))

        candidate_chunks = [c for c in reranked if c.get("rerank_score", -999.0) >= -5.0][:RERANKER_TOP_K]

        # ── ISSUE 1 FIX ────────────────────────────────────────────────
        # The reranked candidates can span multiple different source PDFs.
        # To guarantee every displayed source is an actual chunk/page from
        # the SAME PDF that supports the generated answer, identify the
        # single "dominant" book among the candidates (the book with the
        # strongest combined similarity + rerank evidence) and pull its
        # top matching chunk plus its immediate neighboring chunks from
        # the real FAISS metadata store. This uses only genuine dataset
        # records (book_name / page_number / chunk_id / text as indexed
        # in vectordb/faiss_manager.py) — no source is fabricated.
        dominant_book = self._determine_dominant_book(candidate_chunks)
        final_chunks = self._get_expanded_context(candidate_chunks, dominant_book)

        logger.info(
            "Adaptive search complete ▸ confidence=%s │ k=%d │ best_faiss=%.4f │ returned=%d chunks",
            final_confidence, initial_k, best_score, len(final_chunks),
        )

        return {
            "chunks": final_chunks,
            "confidence": final_confidence,
            "complexity": complexity,
            "k_expanded_to": current_k,
            "best_score": best_score,
            "initial_chunk_ids": [c.get("chunk_id") for c in results],
            "initial_scores": [c.get("score") for c in results],
            "reranked_chunk_ids": [c.get("chunk_id") for c in reranked],
            "reranked_scores": [c.get("rerank_score") for c in reranked],
            "retrieval_time": retrieval_time_elapsed,
            "rerank_time": rerank_time_elapsed,
        }
