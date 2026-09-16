"""App-wide configuration and constants."""

INDEX_DIR = "saved_index"
INDEX_FILE = "faiss.index"
META_FILE = "metadata.pkl"

# Used only if we can't fetch the live model list from Groq (e.g. no API key yet).
# The app fetches the current list automatically once a valid key is entered.
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, open-source embedding model
