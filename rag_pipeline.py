"""
rag_pipeline.py
Core RAG logic: chunking, embedding, FAISS indexing, retrieval, generation.
"""

from __future__ import annotations

import io
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    text:   str
    source: str
    index:  int


@dataclass
class RetrievedContext:
    text:   str
    source: str
    score:  float


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline:
      1. Parse uploaded files (PDF / TXT)
      2. Chunk with LangChain's RecursiveCharacterTextSplitter
      3. Embed with a local sentence-transformers model
      4. Index with FAISS (IndexFlatIP = cosine after normalization)
      5. Retrieve top-k chunks per query
      6. Generate answer with OpenRouter LLM (strict grounding prompt)
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        openrouter_api_key: str | None = None,
        model_name: str = "mistralai/mistral-7b-instruct:free",
        embed_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        top_k: int = 3,
    ):
        self.api_key    = openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = model_name
        self.top_k      = top_k

        # Embedding model (runs locally, no API key needed)
        self.embedder = SentenceTransformer(embed_model)

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Will be populated after build_index()
        self.chunks:     list[Chunk] = []
        self.index:      faiss.Index | None = None
        self.embed_dim:  int = 0

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(self, uploaded_files: list[Any]) -> dict:
        """Parse, chunk, embed, and index the uploaded files."""
        all_chunks: list[Chunk] = []

        for uf in uploaded_files:
            raw_text = self._extract_text(uf)
            splits   = self.splitter.split_text(raw_text)
            for i, split in enumerate(splits):
                all_chunks.append(Chunk(text=split, source=uf.name, index=i))

        if not all_chunks:
            raise ValueError("No text could be extracted from the uploaded files.")

        self.chunks = all_chunks

        # Embed
        texts      = [c.text for c in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        self.embed_dim = embeddings.shape[1]

        # FAISS index (inner product = cosine after normalization)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings)

        return {
            "n_docs":   len(uploaded_files),
            "n_chunks": len(all_chunks),
            "embed_dim": self.embed_dim,
        }

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Retrieve relevant chunks and generate a grounded answer."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        k = top_k or self.top_k

        # 1. Embed query
        q_emb = self.embedder.encode([question], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        # 2. FAISS search
        scores, indices = self.index.search(q_emb, k)
        scores, indices = scores[0], indices[0]

        contexts: list[RetrievedContext] = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            contexts.append(RetrievedContext(
                text=chunk.text,
                source=chunk.source,
                score=float(score),
            ))

        # 3. Generate answer
        answer = self._generate(question, contexts)

        return {
            "answer":   answer,
            "contexts": [{"text": c.text, "source": c.source, "score": c.score} for c in contexts],
        }

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _generate(self, question: str, contexts: list[RetrievedContext]) -> str:
        context_block = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in contexts
        )

        system_prompt = textwrap.dedent("""
            You are a helpful AI assistant for a construction marketplace.
            Answer the user's question ONLY using the provided context below.
            If the answer cannot be found in the context, say:
            "I couldn't find information about this in the provided documents."
            Do NOT use any external knowledge. Be concise and factual.
        """).strip()

        user_prompt = f"""Context:
{context_block}

Question: {question}

Answer (strictly based on the context above):"""

        if not self.api_key:
            return (
                "⚠️ No OpenRouter API key provided. "
                "Add your key in the sidebar to enable answer generation.\n\n"
                "**Retrieved context is shown above** — you can read it directly."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.2,
        }

        resp = requests.post(self.OPENROUTER_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        return data["choices"][0]["message"]["content"].strip()

    # ── Text extraction ───────────────────────────────────────────────────────

    def _extract_text(self, uploaded_file: Any) -> str:
        name = uploaded_file.name.lower()
        raw  = uploaded_file.read()

        if name.endswith(".pdf"):
            if not PDF_SUPPORT:
                raise ImportError("pdfplumber is required to read PDF files. pip install pdfplumber")
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n\n".join(pages)

        # Treat everything else as text
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
