"""
generator.py – Phase 12: Local LLM Generation via Ollama
==========================================================
Connects to a **local** Ollama instance (no cloud APIs!) and sends
the assembled prompt. Supports dynamically specifying the model name
for ablation studies.

IMPORTANT – FULLY OFFLINE
--------------------------
All HTTP requests in this file go ONLY to ``localhost:11434``,
which is Ollama's local REST server running on YOUR machine.
No data is ever sent to any cloud API, external server, or
third-party service.  The ``requests`` library is used purely
for local inter-process communication (Python ↔ Ollama).

Prerequisites
-------------
1. Install Ollama: https://ollama.com
2. Pull required models (e.g., `ollama pull qwen2.5:0.5b`)

Author  : B.Tech CSE-AI Student Project
"""

import time
from typing import Any, Dict, List, Optional

import requests

from llm.prompt_builder import build_prompt
from utils.config import (
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_REPEAT_PENALTY,
    LLM_NUM_PREDICT,
    LLM_NUM_CTX,
    OLLAMA_BASE_URL,
    DEBUG_MODE,
    MAX_CONTEXT_LENGTH,
)
from utils.logger import logger


class Generator:
    """
    Generate answers using a local Ollama LLM.

    Attributes
    ----------
    base_url : str
        Ollama REST API base URL (default ``http://localhost:11434``).
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        """
        Initialise the generator.

        Parameters
        ----------
        base_url : str
            Ollama server URL.
        """
        self.base_url: str = base_url
        self.session = requests.Session()
        logger.info("Initialising LLM Generator (Ollama @ %s)", base_url)

    # ───────────────── Ollama connectivity helpers ───────────────────

    def check_ollama_running(self) -> bool:
        """
        Check whether the Ollama server is reachable.

        Returns
        -------
        bool
            ``True`` if Ollama responds, ``False`` otherwise.
        """
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_available_models(self) -> List[str]:
        """
        Ask Ollama which models have been pulled locally.

        Returns
        -------
        list[str]
            Model names available on the server.
        """
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()
            models: List[str] = [
                m["name"] for m in data.get("models", [])
            ]
            return models
        except Exception as exc:
            logger.warning("Could not list Ollama models: %s", exc)
            return []

    def check_model_pulled(self, model_name: str) -> bool:
        """
        Check if a specific model is available in Ollama.
        """
        available: List[str] = self.list_available_models()
        return any(model_name in m for m in available)

    # ───────────────── Main generation method ────────────────────────

    def _truncate_context(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Truncate chunks until the prompt fits the max context length."""
        prompt = build_prompt(query, chunks)
        while len(prompt) > MAX_CONTEXT_LENGTH and len(chunks) > 1:
            chunks.pop() # Remove least relevant chunk
            prompt = build_prompt(query, chunks)
        return prompt

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        Build a prompt from *query* + *chunks* and generate an answer using *model_name*.
        """
        # 1. Prompt Validation
        if not query or not query.strip():
            logger.error("Empty user query provided.")
            return {"answer": "No query provided.", "model": model_name, "elapsed_secs": 0.0, "tokens": 0, "error": "empty_query", "prompt_length": 0}

        if not chunks:
            logger.error("Empty context provided.")
            return {"answer": "No relevant information was found in the documents.", "model": model_name, "elapsed_secs": 0.0, "tokens": 0, "error": "empty_context", "prompt_length": 0}

        # 2. Check Ollama is running
        if DEBUG_MODE:
            logger.info("--- DEBUG MODE ENABLED ---")
            logger.info(f"Ollama Host: {self.base_url}")
            logger.info(f"Model name: {model_name}")
            logger.info("Connection status: Checking...")

        if not self.check_ollama_running():
            error_msg = "Ollama server is not running. Start it using: ollama serve"
            logger.error(error_msg)
            return {"answer": "", "model": model_name, "elapsed_secs": 0.0, "tokens": 0, "error": "ollama_not_running", "prompt_length": 0}

        if DEBUG_MODE:
            logger.info("Connection status: Connected to Ollama")

        if not self.check_model_pulled(model_name):
            error_msg = f"Model '{model_name}' is not installed. Download it using: ollama pull {model_name}"
            logger.error(error_msg)
            return {"answer": "", "model": model_name, "elapsed_secs": 0.0, "tokens": 0, "error": "model_not_found", "prompt_length": 0}

        # 3. Build & Truncate Prompt
        prompt_start = time.time()
        prompt = self._truncate_context(query, chunks)
        prompt_time = round(time.time() - prompt_start, 2)
        
        if len(prompt) == 0:
            logger.error("Generated prompt is empty.")
            return {"answer": "", "model": model_name, "elapsed_secs": 0.0, "tokens": 0, "error": "empty_prompt", "prompt_length": 0, "prompt_time": prompt_time}

        if DEBUG_MODE:
            logger.info(f"Ollama Request Started")
            logger.info(f"Number of retrieved chunks: {len(chunks)}")
            logger.info(f"Retrieved chunk IDs: {[c.get('chunk_id') for c in chunks]}")
            logger.info(f"Retrieved book names: {[c.get('book_name') for c in chunks]}")
            logger.info(f"Retrieved page numbers: {[c.get('page_number') for c in chunks]}")
            logger.info(f"Similarity scores: {[c.get('score') for c in chunks]}")
            logger.info(f"Reranker scores: {[c.get('rerank_score') for c in chunks]}")
            logger.info(f"Total context characters: {sum(len(c.get('text', '')) for c in chunks)}")
            logger.info(f"Prompt length: {len(prompt)}")
            logger.info(f"Complete prompt sent to Ollama:\n{prompt}\n")

        # 4. Call Ollama REST API with Retry Logic
        payload: Dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                "top_k": LLM_TOP_K,
                "repeat_penalty": LLM_REPEAT_PENALTY,
                "num_predict": LLM_NUM_PREDICT,
                "num_ctx": LLM_NUM_CTX,
            },
        }

        max_retries = 1
        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                if DEBUG_MODE:
                    logger.info(f"Attempt {attempt + 1}: Sending prompt to Ollama ({model_name})...")
                else:
                    logger.info("Sending prompt to Ollama (%s) …", model_name)
                    
                resp = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=LLM_TIMEOUT,
                )
                resp.raise_for_status()

                result: Dict[str, Any] = resp.json()
                answer: str = result.get("response", "").strip()
                tokens: int = result.get("eval_count", 0)
                elapsed: float = round(time.time() - start, 2)
                
                if DEBUG_MODE:
                    logger.info("Ollama request completed.")
                    logger.info(f"Ollama response length: {len(answer)}")
                    logger.info(f"LLM execution time: {elapsed}s")
                    logger.info(f"Final generated answer:\n{answer}\n")
                else:
                    logger.info("LLM response received in %.2f s. (%d tokens)", elapsed, tokens)

                if not answer:
                    if attempt < max_retries:
                        if DEBUG_MODE: logger.warning("Empty response received. Retrying...")
                        continue
                    else:
                        logger.error("Generation genuinely failed after retry (empty response).")
                        return {"answer": "", "model": model_name, "elapsed_secs": elapsed, "tokens": tokens, "error": "empty_response", "prompt_length": len(prompt), "prompt_time": prompt_time}

                return {"answer": answer, "model": model_name, "elapsed_secs": elapsed, "tokens": tokens, "error": None, "prompt_length": len(prompt), "prompt_time": prompt_time}

            except requests.Timeout:
                elapsed = round(time.time() - start, 2)
                logger.error(f"Ollama Timeout after {LLM_TIMEOUT}s on attempt {attempt + 1}")
                if attempt < max_retries:
                    if DEBUG_MODE: logger.warning("Retrying due to timeout...")
                    continue
                return {"answer": "", "model": model_name, "elapsed_secs": elapsed, "tokens": 0, "error": "timeout", "prompt_length": len(prompt), "prompt_time": prompt_time}

            except requests.RequestException as exc:
                elapsed = round(time.time() - start, 2)
                logger.error(f"Error communicating with Ollama: {exc}")
                if attempt < max_retries:
                    if DEBUG_MODE: logger.warning("Retrying due to request error...")
                    continue
                return {"answer": "", "model": model_name, "elapsed_secs": elapsed, "tokens": 0, "error": "request_error", "prompt_length": len(prompt), "prompt_time": prompt_time}
            
            except ValueError as exc:
                elapsed = round(time.time() - start, 2)
                logger.error(f"JSON parsing error: {exc}")
                if attempt < max_retries:
                    if DEBUG_MODE: logger.warning("Retrying due to JSON error...")
                    continue
                return {"answer": "", "model": model_name, "elapsed_secs": elapsed, "tokens": 0, "error": "json_error", "prompt_length": len(prompt), "prompt_time": prompt_time}
