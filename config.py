
"""
Application-wide configuration and constants.
"""


# ---------------------------------------------------------------------------
# FAISS STORAGE
# ---------------------------------------------------------------------------

INDEX_DIR = "saved_index"

INDEX_FILE = "faiss.index"

META_FILE = "metadata.pkl"


# ---------------------------------------------------------------------------
# GROQ FALLBACK MODELS
# ---------------------------------------------------------------------------

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]


# ---------------------------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------------------------

# Open-source SentenceTransformer model.
#
# It runs locally and does not require an API key.

EMBED_MODEL_NAME = (
    "all-MiniLM-L6-v2"
)

