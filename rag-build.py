#!/usr/bin/env python3
"""
rag-build.py — Ingest documents into a FAISS vector database.

Supported file types:  .txt  |  .md  |  .pdf  |  .docx  |  .doc

Usage:
    python rag-build.py <db_path> <file_or_dir> [<file_or_dir> ...]

Examples:
    python rag-build.py ./mydb docs/
    python rag-build.py ./mydb report.pdf notes/ readme.md letter.docx

Dependencies:
    pip install langchain langchain-community langchain-huggingface \
                faiss-cpu sentence-transformers \
                pymupdf                         # PDF support (via PyMuPDF)
                python-docx docx2txt            # .docx support

    .doc (legacy binary format) additionally requires LibreOffice:
        macOS:   brew install libreoffice
        Ubuntu:  sudo apt install libreoffice
        Windows: https://www.libreoffice.org/download/
"""

import sys
import os
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    Docx2txtLoader,                  # .docx — pure Python, no system deps
    UnstructuredWordDocumentLoader,  # .doc  — requires LibreOffice
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import importlib
load_embeddings = importlib.import_module("rag-utils").load_embeddings


# ── Configuration ────────────────────────────────────────────────────────────

# Embedding model — loaded from local cache if available, else downloaded (~90 MB)
# Cache location: ./model_cache/  (override with RAG_MODEL_CACHE env var)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk size and overlap for splitting documents
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # characters shared between adjacent chunks

# Map each supported extension to its LangChain loader class + kwargs
FILE_LOADERS = {
    ".txt":  (TextLoader,                   {"encoding": "utf-8"}),
    ".md":   (UnstructuredMarkdownLoader,   {}),
    ".pdf":  (PyMuPDFLoader,                {}),
    ".docx": (Docx2txtLoader,               {}),  # pure Python via python-docx
    ".doc":  (UnstructuredWordDocumentLoader, {}), # requires LibreOffice
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def loader_for_file(path: Path):
    """Return an instantiated LangChain loader for a single file, or None."""
    ext = path.suffix.lower()
    if ext not in FILE_LOADERS:
        print(f"[warn] Unsupported file type '{ext}', skipping: {path}")
        return None
    cls, kwargs = FILE_LOADERS[ext]
    return cls(str(path), **kwargs)


def collect_documents(sources: list[str]):
    """Load documents from a mix of files and directories."""
    docs = []

    for source in sources:
        p = Path(source)
        if not p.exists():
            print(f"[warn] Skipping '{source}' — path not found.")
            continue

        if p.is_dir():
            # Walk the directory and dispatch each file to the right loader
            print(f"[info] Scanning directory: {p}")
            files = [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                loader = loader_for_file(f)
                if loader:
                    print(f"  [load] {f}")
                    try:
                        docs.extend(loader.load())
                    except Exception as e:
                        print(f"  [warn] Failed to load {f}: {e}")
        else:
            loader = loader_for_file(p)
            if loader:
                print(f"[info] Loading file: {p}")
                try:
                    docs.extend(loader.load())
                except Exception as e:
                    print(f"[warn] Failed to load {p}: {e}")

    return docs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    db_path = sys.argv[1]
    sources = sys.argv[2:]

    # 1. Load raw documents
    print("\n=== Step 1: Loading documents ===")
    documents = collect_documents(sources)
    if not documents:
        print("[error] No documents loaded. Check your paths.")
        sys.exit(1)
    print(f"  Loaded {len(documents)} document(s).")

    # 2. Split into chunks
    print("\n=== Step 2: Splitting into chunks ===")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunk(s) "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    # 3. Embed and store
    print("\n=== Step 3: Embedding chunks (this may take a moment) ===")
    print(f"  Model: {EMBEDDING_MODEL}")
    embeddings = load_embeddings(EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Persist to disk
    print(f"\n=== Step 4: Saving database to '{db_path}' ===")
    os.makedirs(db_path, exist_ok=True)
    vectorstore.save_local(db_path)
    print(f"  Done. {len(chunks)} vectors saved.")


if __name__ == "__main__":
    main()
