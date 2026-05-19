# RAG Demo

A minimal, self-contained **Retrieval-Augmented Generation (RAG)** demo using open-source tools only — no API keys, no cloud services required.

Given a set of your own documents, it lets you ask questions and compare two answers side by side:

- **Pure LLM** — what the model knows from training alone
- **LLM + RAG** — the model's answer grounded in your documents

---

## How It Works

```
Your documents (.txt / .md / .pdf / .docx / .doc)
        │
        ▼
  [rag-build.py]
  1. Load & parse documents
  2. Split into overlapping chunks
  3. Embed each chunk → vector          (sentence-transformers, local)
  4. Store vectors on disk              (FAISS)
        │
        ▼
   FAISS vector database
        │
        ▼
  [rag-chat.py]
  User question
     ├─► LLM directly                  → Pure LLM answer
     └─► Top-K similar chunks (FAISS)
             └─► LLM + context         → RAG answer  +  source files cited
```

**Embedding model** (`sentence-transformers/all-MiniLM-L6-v2`) is downloaded from HuggingFace on first run and cached locally under `model_cache/` — subsequent runs and the chat script work fully offline.

**LLM** is served by [Ollama](https://ollama.com) running on your machine. Any model you have pulled works (`gemma3:1b`, `mistral`, `phi3`, etc.).

---

## Project Structure

```
.
├── rag-build.py      # Ingest documents → FAISS vector database
├── rag-chat.py       # Interactive chat: pure LLM vs. LLM + RAG
├── rag-utils.py      # Shared helper: local-first embedding model loader
├── requirements.txt
└── model_cache/      # Auto-created; stores the embedding model locally
```

---

## Requirements

- Python 3.10 - 3.12
- [Ollama](https://ollama.com) installed and running locally

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rag-demo.git
cd rag-demo

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull an LLM into Ollama (one-time, choose any model)
ollama pull gemma3:1b

# 5. (Optional) Install LibreOffice for .doc (legacy Word) support
#    macOS:   brew install libreoffice
#    Ubuntu:  sudo apt install libreoffice
#    Windows: https://www.libreoffice.org/download/
```

---

## Usage

### Step 1 — Build the vector database

Point `rag-build.py` at your documents. You can mix individual files and directories; `.txt`, `.md`, and `.pdf` files are picked up automatically.

```bash
python rag-build.py <db_path> <file_or_dir> [<file_or_dir> ...]
```

Examples:

```bash
# Index a whole folder
python rag-build.py ./mydb  docs/

# Mix files and directories
python rag-build.py ./mydb  report.pdf  notes/  README.md
```

Output:

```
=== Step 1: Loading documents ===
  [load] docs/guide.pdf
  [load] docs/notes.md
  Loaded 2 document(s).

=== Step 2: Splitting into chunks ===
  Created 38 chunk(s) (chunk_size=500, overlap=50).

=== Step 3: Embedding chunks (this may take a moment) ===
  [embed] Downloading from HuggingFace Hub …
  [cache] Model saved to 'model_cache/sentence-transformers__all-MiniLM-L6-v2'.

=== Step 4: Saving database to './mydb' ===
  Done. 38 vectors saved.
```

The embedding model is only downloaded once; all future runs load it from `model_cache/`.

### Step 2 — Chat

```bash
python rag-chat.py <db_path>
```

Example:

```bash
python rag-chat.py ./mydb
```

Then type any question:

```
You: What is the return policy?

────────────────────────────────────────────────────────────
  1 · Pure LLM  (no context from your documents)
────────────────────────────────────────────────────────────
Return policies vary by retailer, but typically allow returns
within 30 days with a receipt…

────────────────────────────────────────────────────────────
  2 · LLM + RAG  (answer grounded in your documents)
────────────────────────────────────────────────────────────
According to the provided documentation, items may be returned
within 14 days in original packaging. Sale items are final sale.

  [Sources used]
    • docs/policy.pdf
```

Type `quit` or `exit` to stop.

---

## Configuration

All tunable constants sit at the top of each script.

| Constant | File | Default | Description |
|---|---|---|---|
| `EMBEDDING_MODEL` | both | `sentence-transformers/all-MiniLM-L6-v2` | Any HuggingFace sentence-transformer |
| `OLLAMA_MODEL` | `rag-chat.py` | `gemma3:1b` | Any model pulled into Ollama |
| `CHUNK_SIZE` | `rag-build.py` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `rag-build.py` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `rag-chat.py` | `4` | Number of chunks retrieved per query |

### Model cache location

By default the embedding model is cached in `./model_cache/`. Override with an environment variable:

```bash
export RAG_MODEL_CACHE=/shared/models
python rag-build.py ./mydb docs/
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` + `langchain-community` | Document loaders, text splitter, RAG chain |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `langchain-ollama` | Ollama LLM integration |
| `faiss-cpu` | Local vector store |
| `sentence-transformers` | Embedding model runtime |
| `pymupdf` | PDF parsing |
| `unstructured` | Markdown parsing |
| `python-docx` + `docx2txt` | .docx parsing |
| LibreOffice *(system, optional)* | .doc (legacy binary) parsing |

---

## .gitignore recommendations

Add these to avoid committing large binary files:

```
model_cache/
*.faiss
*.pkl
__pycache__/
.venv/
```
