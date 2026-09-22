"""
prompt_builder.py – Phase 11: Prompt Construction
===================================================
Builds a carefully crafted prompt that tells the LLM:

* Answer **only** from the provided context.
* **Never** guess or use outside knowledge.
* If the information is missing, say so explicitly.

Each chunk in the context is labelled with its source book name and
page number so the user can verify the answer.

Author  : B.Tech CSE-AI Student Project
"""

from typing import Any, Dict, List


# ─── System instruction (always prepended) ───────────────────────────
_SYSTEM_INSTRUCTION: str = (
    "You are a strict, factual medical AI assistant.\n"
    "1. Answer ONLY from the retrieved context provided below.\n"
    "2. Never use external knowledge. Never hallucinate. Never guess. Never generate unsupported facts.\n"
    "3. Never output phrases such as 'Based on general knowledge', 'I can provide an educated guess', "
    "'The answer is not explicitly stated', 'It is likely', 'Probably', 'Maybe', 'I think', 'The question is asking about...', "
    "'This question asks...', 'According to the question...', 'I will explain...', or 'According to general medical knowledge'.\n"
    "4. If the context does not contain sufficient information to answer the question, you MUST return EXACTLY the phrase:\n"
    "Insufficient information in the retrieved documents.\n"
    "Do not add conversational text or apologize.\n"
    "5. Answer directly with the medical answer. Start immediately with the facts. Keep answers short and concise. Avoid unnecessary explanations. Avoid repeating information.\n"
    "6. If multiple chunks support the answer, combine them into one concise response.\n"
    "7. Always cite the Book Name and Page Number for facts in your answer.\n"
)


def build_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Construct a full prompt string from a user query and retrieved chunks.

    Parameters
    ----------
    query : str
        The user's natural-language question.
    chunks : list[dict]
        Re-ranked chunks from adaptive retrieval.  Each dict should have:
        ``book_name``, ``page_number``, ``text``, and optionally
        ``score`` / ``rerank_score``.

    Returns
    -------
    str
        A ready-to-send prompt string for the LLM.

    Example output
    --------------
    ::

        [System]
        You are a helpful …

        [Context]
        --- Source 1 (Book: Robbins Pathology, Page: 142) ---
        The mitral valve …

        [Question]
        What are the symptoms of mitral valve prolapse?

        [Answer]
    """
    # ── Build the context block ──────────────────────────────────────
    context_lines: List[str] = []
    seen_sentences = set()
    
    for idx, chunk in enumerate(chunks, start=1):
        book: str = chunk.get("book_name", "Unknown Book")
        page: int = chunk.get("page_number", 0)
        text: str = chunk.get("text", "")

        # Deduplicate sentences to keep the prompt compact
        sentences = text.replace(".\n", ". ").split(". ")
        unique_sentences = []
        for s in sentences:
            clean_s = s.strip()
            if not clean_s:
                continue
            norm_s = clean_s.lower()
            # Only deduplicate reasonably long sentences to avoid filtering out short common phrases
            if len(norm_s) > 20:
                if norm_s not in seen_sentences:
                    seen_sentences.add(norm_s)
                    unique_sentences.append(clean_s)
            else:
                unique_sentences.append(clean_s)

        if not unique_sentences:
            continue
            
        compact_text = ". ".join(unique_sentences)
        if not compact_text.endswith("."):
            compact_text += "."

        context_lines.append(
            f"--- Source {idx} (Book: {book}, Page: {page}) ---\n{compact_text}"
        )

    context_block: str = "\n\n".join(context_lines)

    # ── Assemble the full prompt ─────────────────────────────────────
    prompt: str = (
        f"[System]\n{_SYSTEM_INSTRUCTION}\n\n"
        f"[Context]\n{context_block}\n\n"
        f"[Question]\n{query}\n\n"
        f"[Answer]\n"
    )

    return prompt
