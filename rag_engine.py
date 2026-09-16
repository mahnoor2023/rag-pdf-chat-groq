"""
RAGEngine: everything related to turning PDFs into a searchable FAISS index.
Kept separate from app.py so the UI file stays clean and this logic is
reusable/testable on its own.
"""

import os
import pickle

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import INDEX_DIR, INDEX_FILE, META_FILE, EMBED_MODEL_NAME

os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, INDEX_FILE)
META_PATH = os.path.join(INDEX_DIR, META_FILE)


class RAGEngine:
    """Holds the FAISS index + chunk metadata and exposes simple operations."""

    def __init__(self):
        self.index = None
        self.chunks = []          # list of {"text", "source", "page"}
        self.processed_files = []
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBED_MODEL_NAME)
        return self._embedder

    # ------------------------------------------------------------------ #
    # Extraction & chunking
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_pdf_pages(file):
        """Return list of (page_number, text) for a PDF file-like object."""
        reader = PdfReader(file)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i + 1, text))
        return pages

    @staticmethod
    def chunk_text(text, chunk_size=400, overlap=50):
        """Simple word-based sliding-window chunking."""
        words = text.split()
        if not words:
            return []
        step = max(chunk_size - overlap, 1)
        chunks = []
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            if piece.strip():
                chunks.append(piece)
            if start + chunk_size >= len(words):
                break
        return chunks

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    def add_pdf(self, file, chunk_size=400, overlap=50):
        """Extract, chunk, embed, and index a single uploaded PDF. Returns #chunks added."""
        if file.name in self.processed_files:
            return 0
        new_chunks = []
        for page_num, text in self.extract_pdf_pages(file):
            for piece in self.chunk_text(text, chunk_size, overlap):
                new_chunks.append({"text": piece, "source": file.name, "page": page_num})
        if new_chunks:
            self._add_chunks(new_chunks)
        self.processed_files.append(file.name)
        return len(new_chunks)

    def _add_chunks(self, new_chunks):
        texts = [c["text"] for c in new_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

    def retrieve(self, query, k=4):
        """Return top-k most relevant chunks (with similarity score) for a query."""
        if self.index is None or self.index.ntotal == 0:
            return []
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results

    def clear(self):
        self.index = None
        self.chunks = []
        self.processed_files = []

    # ------------------------------------------------------------------ #
    # Persistence (save/load FAISS index + metadata to disk)
    # ------------------------------------------------------------------ #
    def save(self):
        if self.index is None:
            return False
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "processed_files": self.processed_files}, f)
        return True

    def load(self):
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return False
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.processed_files = data["processed_files"]
        return True
