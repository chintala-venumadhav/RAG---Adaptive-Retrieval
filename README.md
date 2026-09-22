# ⚕️ Adaptive RAG for Medical PDF Question Answering

> **Fully offline** AI-powered question answering system that reads medical textbook PDFs and answers your questions using Adaptive Retrieval-Augmented Generation.

---

## 🌟 Highlights
| Feature | Detail |
|---|---|
| 🔒 **100% Offline** | No OpenAI, Gemini, Claude, Cohere, Groq, or HuggingFace APIs. Everything runs on your laptop. |
| 🧠 **Adaptive Retrieval** | Dynamically adjusts how many chunks to retrieve based on query complexity and confidence. |
| 📊 **Re-Ranking** | Cross-encoder re-ranks FAISS results for higher accuracy. |
| 🏥 **Medical Focus** | Optimised for medical textbook PDFs with source citations (Book + Page). |
| 💻 **CPU Friendly** | Runs on Intel i5 / 16 GB RAM / No GPU required. |
| 🎨 **Dark Theme UI** | Professional Streamlit interface with confidence badges and metrics. |

---

## 🔒 Fully Offline Guarantee

This project is designed to run **100% offline** after a one-time model download. Here is exactly what runs where:

### ✅ Always Local (No Network Required)
| Component | Technology | Network? |
|-----------|-----------|----------|
| PDF Reading | PyMuPDF (fitz) | ❌ No |
| Text Cleaning | Python regex | ❌ No |
| Text Chunking | LangChain Text Splitters | ❌ No |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | ❌ No |
| Vector Database | FAISS (faiss-cpu) | ❌ No |
| Re-Ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | ❌ No |
| LLM Inference | Ollama (localhost:11434) | ❌ No |
| Web UI | Streamlit (localhost:8501) | ❌ No |
| Fonts | Segoe UI (pre-installed on Windows) | ❌ No |

### 📥 One-Time Downloads (First Run Only)
These models are downloaded **once** and cached permanently on your machine:

| Model | Size | Purpose | Cached Location |
|-------|------|---------|----------------|
| `all-MiniLM-L6-v2` | ~80 MB | Text embeddings | `~/.cache/torch/sentence_transformers/` |
| `ms-marco-MiniLM-L-6-v2` | ~80 MB | Re-ranking | `~/.cache/torch/sentence_transformers/` |
| `llama3.2:3b` | ~2 GB | Answer generation | `~/.ollama/models/` |

After these downloads, you can **disconnect from the internet entirely** and the system will work perfectly.

### 🚫 What This Project Does NOT Use
- ❌ OpenAI API
- ❌ Google Gemini API
- ❌ Anthropic Claude API
- ❌ Cohere API
- ❌ Groq API
- ❌ HuggingFace Inference API
- ❌ Pinecone Cloud
- ❌ Weaviate Cloud
- ❌ Chroma Cloud
- ❌ Any API keys or environment variables
- ❌ Any external HTTPS requests after model downloads

---

## 📁 Project Structure

```
Adaptive-RAG/
│
├── data/
│      ├── books/                  # Place your medical PDFs here
│      ├── embeddings/             # Pickle files (auto-generated)
│      ├── faiss/                  # FAISS index files (auto-generated)
│
├── preprocessing/
│      ├── pdf_loader.py           # Phase 1: Read PDFs with PyMuPDF
│      ├── cleaner.py              # Phase 2: Clean extracted text
│      ├── chunker.py              # Phase 3: Split into 500-char chunks
│
├── embeddings/
│      ├── embedding_model.py      # Phase 4: SentenceTransformer wrapper
│      ├── create_embeddings.py    # Pipeline: PDFs → Chunks → Embeddings → FAISS
│
├── vectordb/
│      ├── faiss_manager.py        # Phase 5: FAISS index CRUD operations
│
├── retrieval/
│      ├── retriever.py            # Phase 6: Base FAISS retriever
│      ├── adaptive_retriever.py   # Phases 7-9: Adaptive retrieval + confidence
│      ├── reranker.py             # Phase 10: Cross-encoder re-ranking
│
├── llm/
│      ├── prompt_builder.py       # Phase 11: Prompt construction
│      ├── generator.py            # Phase 12: Ollama LLM (localhost only)
│
├── ui/
│      ├── streamlit_app.py        # Phase 13: Streamlit dark-theme UI
│
├── utils/
│      ├── config.py               # All paths, thresholds, model names
│      ├── logger.py               # File + console logging
│
├── requirements.txt               # Python dependencies (all offline)
├── README.md                      # This file
├── run.py                         # Entry point launcher
```

---

## 🧠 How Adaptive Retrieval Works

Traditional RAG always retrieves a fixed number of chunks. **Adaptive RAG** is smarter:

```
User Question
     │
     ▼
┌─────────────────────────┐
│ Phase 8: Detect          │
│ Query Complexity         │
│ (Simple vs Complex)      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 7: Initial Search  │
│ Simple → Top 3 chunks    │
│ Complex → Top 8 chunks   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 9: Check           │
│ Confidence Score         │
│                          │
│ High (≥0.90) → Stop ✅  │
│ Medium (≥0.75) → Top 8  │
│ Low (<0.75) → Top 15    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 10: Re-Rank        │
│ Cross-Encoder scores     │
│ Keep best 5 chunks       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Phase 12: Generate       │
│ Answer via Ollama LLM    │
│ (localhost only)         │
└─────────────────────────┘
```

---

## 📋 Prerequisites

- **Python 3.11** (Highly recommended. Python 3.14 is currently **not supported** by several core AI libraries like PyTorch and FAISS because stable pre-compiled binaries are not yet available).
- **Ollama** installed on your system (for local LLM inference).

---

## 🚀 Windows 11 Execution Guide

Follow these exact steps to set up and run the Adaptive RAG system on Windows 11 using the Python Launcher (`py`).

**Step 1**

Open PowerShell

**Step 2**

Navigate to the project folder
```powershell
cd "C:\Users\VENU MADHAV\OneDrive\ドキュメント\Adaptive Retrival\Adaptive-RAG"
```

**Step 3**

Create a virtual environment
```powershell
py -m venv .venv
```

**Step 4**

Activate the virtual environment
```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
and activate again.

**Step 5**

Upgrade pip
```powershell
py -m pip install --upgrade pip
```

**Step 6**

Install all project requirements
```powershell
py -m pip install -r requirements.txt
```

**Step 7**

Verify installation
```powershell
py -m pip list
```

**Step 8**

Install Ollama and pull a model (one-time setup)
```powershell
ollama pull llama3.2:3b
```

**Step 9**

Build embeddings (first time, or after adding new PDFs)
```powershell
py embeddings\create_embeddings.py
```

**Step 10**

Run the Streamlit application
```powershell
py -m streamlit run ui\streamlit_app.py
```

**Step 11**

Automatically open the browser
```text
http://localhost:8501
```

---

## 🛠️ Troubleshooting Guide

If you encounter errors during setup, follow these solutions:

### • Virtual Environment Errors
**Error:** `py : The term 'py' is not recognized...`
**Fix:** Ensure the Python Launcher for Windows is installed. When installing Python, make sure to check "py launcher" in the installer. Re-run the Python installer and modify your installation to include it.

### • PowerShell Execution Policy Errors
**Error:** `cannot be loaded because running scripts is disabled on this system.`
**Fix:** Windows blocks script execution by default. Run this command to temporarily allow script execution for your current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then try activating the virtual environment again.

### • pip Errors
**Error:** `pip is not recognized as an internal or external command` or `ModuleNotFoundError: No module named 'pip'`
**Fix:** Always use pip through the Python launcher:
```powershell
py -m ensurepip --upgrade
py -m pip install --upgrade pip
```

### • FAISS Installation Errors
**Error:** `Failed building wheel for faiss-cpu` or `Could not find a version that satisfies the requirement faiss-cpu`
**Fix:** If the default installation fails, automatically install a compatible version:
```powershell
py -m pip install faiss-cpu
```
*Alternatively, install from conda (if using Anaconda/Miniconda):* `conda install -c pytorch faiss-cpu`

### • Torch Installation Errors
**Error:** `MemoryError` or massive downloads failing when installing `torch` (which is a dependency of `sentence-transformers`).
**Fix:** Install the CPU-only version of PyTorch explicitly automatically:
```powershell
py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
After successful CPU torch installation, re-run Step 6.

### • Streamlit Errors
**Error:** `Streamlit requires raw Python` or `module not found`.
**Fix:** Ensure you activated the virtual environment before installing requirements. If the command fails, use the exact explicit command:
```powershell
py -m streamlit run ui\streamlit_app.py
```

### • Ollama Errors
**Error:** Application says LLM failed or `Connection refused`
**Fix:** 
1. **Detect it:** If Ollama is not installed, go to [https://ollama.com/](https://ollama.com/) and download the Windows installer.
2. **Install:** Run the installer and launch Ollama.
3. **Pull Model:** Open a new PowerShell and pull the required model:
   ```powershell
   ollama pull llama3.2:3b
   ```

---

## 🔮 Future Improvements

- Support for more document formats (DOCX, EPUB)
- Multi-language medical Q&A
- Chat history persistence across sessions
- PDF annotation highlighting
- Medical terminology autocomplete
- Batch question processing
- Export answers to PDF/DOCX reports
