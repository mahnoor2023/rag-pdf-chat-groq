
"""
RAGEngine

Responsible for:

1. PDF text extraction
2. Text chunking
3. SentenceTransformer embeddings
4. FAISS indexing
5. Similarity search
6. Saving/loading the index
"""

import os
import pickle

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_DIR,
    INDEX_FILE,
    META_FILE,
    EMBED_MODEL_NAME,
)


# ---------------------------------------------------------------------------
# STORAGE
# ---------------------------------------------------------------------------

os.makedirs(
    INDEX_DIR,
    exist_ok=True,
)

INDEX_PATH = os.path.join(
    INDEX_DIR,
    INDEX_FILE,
)

META_PATH = os.path.join(
    INDEX_DIR,
    META_FILE,
)


# ---------------------------------------------------------------------------
# RAG ENGINE
# ---------------------------------------------------------------------------

class RAGEngine:
    """
    Manages PDF documents, embeddings and FAISS search.
    """

    def __init__(self):

        self.index = None

        self.chunks = []

        self.processed_files = []

        self._embedder = None

    # ------------------------------------------------------------------
    # EMBEDDING MODEL
    # ------------------------------------------------------------------

    @property
    def embedder(self):

        if self._embedder is None:

            self._embedder = SentenceTransformer(
                EMBED_MODEL_NAME
            )

        return self._embedder

    # ------------------------------------------------------------------
    # PDF EXTRACTION
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pdf_pages(file):
        """
        Extract text page-by-page from a PDF.

        Returns:
            [
                (page_number, text),
                ...
            ]
        """

        reader = PdfReader(file)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = (
                page.extract_text()
                or ""
            )

            text = text.strip()

            if text:

                pages.append(
                    (
                        page_number,
                        text,
                    )
                )

        return pages

    # ------------------------------------------------------------------
    # TEXT CHUNKING
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(
        text,
        chunk_size=400,
        overlap=50,
    ):
        """
        Split text into overlapping word-based chunks.
        """

        words = text.split()

        if not words:

            return []

        # Prevent invalid overlap.
        overlap = min(
            overlap,
            chunk_size - 1,
        )

        step = max(
            chunk_size - overlap,
            1,
        )

        chunks = []

        for start in range(
            0,
            len(words),
            step,
        ):

            piece = " ".join(
                words[
                    start:start + chunk_size
                ]
            )

            if piece.strip():

                chunks.append(
                    piece.strip()
                )

            # Stop once the final chunk is reached.
            if (
                start + chunk_size
                >= len(words)
            ):

                break

        return chunks

    # ------------------------------------------------------------------
    # ADD PDF
    # ------------------------------------------------------------------

    def add_pdf(
        self,
        file,
        chunk_size=400,
        overlap=50,
    ):
        """
        Extract, chunk, embed and index a PDF.

        Returns:
            Number of new chunks.
        """

        if file.name in self.processed_files:

            return 0

        new_chunks = []

        pages = self.extract_pdf_pages(
            file
        )

        for page_number, text in pages:

            pieces = self.chunk_text(
                text,
                chunk_size,
                overlap,
            )

            for piece in pieces:

                new_chunks.append(
                    {
                        "text": piece,
                        "source": file.name,
                        "page": page_number,
                    }
                )

        if new_chunks:

            self._add_chunks(
                new_chunks
            )

        self.processed_files.append(
            file.name
        )

        return len(new_chunks)

    # ------------------------------------------------------------------
    # ADD CHUNKS TO FAISS
    # ------------------------------------------------------------------

    def _add_chunks(
        self,
        new_chunks,
    ):
        """
        Generate embeddings and add them to FAISS.
        """

        texts = [
            chunk["text"]
            for chunk in new_chunks
        ]

        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Ensure float32 for FAISS.
        embeddings = embeddings.astype(
            "float32"
        )

        # Normalize vectors.
        faiss.normalize_L2(
            embeddings
        )

        # Create index if necessary.
        if self.index is None:

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

        # Add embeddings.
        self.index.add(
            embeddings
        )

        # Store metadata.
        self.chunks.extend(
            new_chunks
        )

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query,
        k=3,
    ):
        """
        Retrieve the most relevant PDF chunks.
        """

        if (
            self.index is None
            or self.index.ntotal == 0
        ):

            return []

        # Limit k.
        k = min(
            max(k, 1),
            self.index.ntotal,
        )

        # Embed question.
        query_embedding = self.embedder.encode(
            [query],
            convert_to_numpy=True,
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        # Normalize.
        faiss.normalize_L2(
            query_embedding
        )

        # Search.
        scores, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index == -1:

                continue

            if index >= len(
                self.chunks
            ):

                continue

            chunk = self.chunks[
                index
            ]

            results.append(
                {
                    **chunk,
                    "score": float(score),
                }
            )

        return results

    # ------------------------------------------------------------------
    # CLEAR
    # ------------------------------------------------------------------

    def clear(self):

        self.index = None

        self.chunks = []

        self.processed_files = []

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    def save(self):

        if self.index is None:

            return False

        faiss.write_index(
            self.index,
            INDEX_PATH,
        )

        with open(
            META_PATH,
            "wb",
        ) as file:

            pickle.dump(
                {
                    "chunks": self.chunks,
                    "processed_files": self.processed_files,
                },
                file,
            )

        return True

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load(self):

        if not (
            os.path.exists(
                INDEX_PATH
            )
            and os.path.exists(
                META_PATH
            )
        ):

            return False

        self.index = faiss.read_index(
            INDEX_PATH
        )

        with open(
            META_PATH,
            "rb",
        ) as file:

            data = pickle.load(
                file
            )

        self.chunks = data.get(
            "chunks",
            [],
        )

        self.processed_files = data.get(
            "processed_files",
            [],
        )

        return True

