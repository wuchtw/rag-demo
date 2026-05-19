"""
rag-utils.py — Local-first embedding model loader shared by rag-build.py and rag-chat.py.

Resolution order
────────────────
1. Local snapshot directory  (MODEL_CACHE_DIR/<safe_model_name>/)
   Checked first; works fully offline once the model has been cached.

2. HuggingFace Hub download
   Used when no local snapshot exists.  After downloading, the model is
   saved to MODEL_CACHE_DIR so future runs never need the network.

Override the cache directory at runtime:
    export RAG_MODEL_CACHE=/path/to/my/models
"""

import os
import re
import warnings
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings


# ── Configuration ─────────────────────────────────────────────────────────────

# Default cache directory: <repo_root>/model_cache/
# Override with the RAG_MODEL_CACHE environment variable.
_DEFAULT_CACHE = Path(__file__).parent / "model_cache"
MODEL_CACHE_DIR = Path(os.environ.get("RAG_MODEL_CACHE", _DEFAULT_CACHE))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_dirname(model_name: str) -> str:
    """Turn 'org/model-name' into a safe directory name 'org__model-name'."""
    return re.sub(r"[/\\]", "__", model_name)


def _local_model_path(model_name: str) -> Path:
    return MODEL_CACHE_DIR / _safe_dirname(model_name)


def _is_cached(model_name: str) -> bool:
    """Return True if a usable local snapshot already exists."""
    p = _local_model_path(model_name)
    # A saved sentence-transformers model always contains config.json
    return (p / "config.json").exists()


def _save_model_locally(model_name: str, embeddings: HuggingFaceEmbeddings) -> None:
    """Persist the underlying SentenceTransformer model to MODEL_CACHE_DIR."""
    dest = _local_model_path(model_name)
    dest.mkdir(parents=True, exist_ok=True)
    # The internal SentenceTransformer is exposed as `_client` in recent
    # langchain-huggingface releases; older versions used `client`.
    st_model = getattr(embeddings, "_client", None) or getattr(embeddings, "client", None)
    if st_model is None:
        raise AttributeError(
            "Cannot locate the underlying SentenceTransformer on this version of "
            "langchain-huggingface. Expected '_client' or 'client' attribute."
        )
    st_model.save(str(dest))
    print(f"  [cache] Model saved to '{dest}'.")


# ── Public API ────────────────────────────────────────────────────────────────

def load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance, preferring a local cache.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. "sentence-transformers/all-MiniLM-L6-v2".

    Returns
    -------
    HuggingFaceEmbeddings
    """
    local_path = _local_model_path(model_name)

    if _is_cached(model_name):
        print(f"  [embed] Loading model from local cache: '{local_path}'")
        # Pass the local directory path; no network access needed.
        return HuggingFaceEmbeddings(model_name=str(local_path))

    print(f"  [embed] Local cache not found for '{model_name}'.")
    print(f"  [embed] Downloading from HuggingFace Hub …")
    print(f"  [embed] Tip: set HF_TOKEN env var to avoid rate limits on large models.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unauthenticated.*", category=UserWarning)
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print(f"  [embed] Caching model for future offline use …")
    _save_model_locally(model_name, embeddings)

    return embeddings
