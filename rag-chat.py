#!/usr/bin/env python3
"""
rag-chat.py — Compare pure-LLM vs. RAG-augmented responses side by side.

Usage:
    python rag-chat.py <db_path>

Example:
    python rag-chat.py ./mydb

Prerequisites:
    • Ollama running locally with a pulled model, e.g.:
        ollama pull gemma3:1b
    • The FAISS database built by rag-build.py

Dependencies:
    pip install langchain langchain-community langchain-huggingface \
                langchain-ollama faiss-cpu sentence-transformers \
                pymupdf                          # PDF support (via PyMuPDF)
"""

import sys
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
import importlib
load_embeddings = importlib.import_module("rag-utils").load_embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ── Configuration ────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match rag-build.py
                                                               # loaded from local cache if available
OLLAMA_MODEL    = "gemma3:1b"   # change to any model you have pulled in Ollama
TOP_K           = 4            # number of chunks retrieved for RAG context


# ── Prompts ──────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. Use ONLY the context below to answer "
        "the question. If the context does not contain enough information, "
        "say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


# ── Display helpers ───────────────────────────────────────────────────────────

def divider(label: str):
    width = 60
    print("\n" + "─" * width)
    print(f"  {label}")
    print("─" * width)


def print_sources(docs):
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(src)
    if sources:
        print("\n  [Sources used]")
        for s in sources:
            print(f"    • {s}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    db_path = sys.argv[1]

    # ── Load vector store ────────────────────────────────────────────────────
    print(f"\nLoading vector database from '{db_path}' …")
    embeddings   = load_embeddings(EMBEDDING_MODEL)
    vectorstore  = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    print("  Vector store ready.")

    # ── Load LLM ─────────────────────────────────────────────────────────────
    print(f"Connecting to Ollama model '{OLLAMA_MODEL}' …")
    llm = OllamaLLM(model=OLLAMA_MODEL)
    print("  LLM ready.")

    # ── Build RAG chain (LCEL) ────────────────────────────────────────────────
    # Modern LangChain uses the LangChain Expression Language (LCEL) pipeline
    # instead of the removed RetrievalQA class.
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    print("\nType your question and press Enter.")
    print("Type  'quit'  or  'exit'  to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Bye!")
            break

        # ── 1. Pure LLM (no retrieval) ────────────────────────────────────
        divider("1 · Pure LLM  (no context from your documents)")
        pure_answer = llm.invoke(question)
        print(pure_answer)

        # ── 2. RAG-augmented ─────────────────────────────────────────────
        divider("2 · LLM + RAG  (answer grounded in your documents)")
        source_docs = retriever.invoke(question)
        rag_answer  = rag_chain.invoke(question)
        print(rag_answer)
        print_sources(source_docs)

        print()


if __name__ == "__main__":
    main()
