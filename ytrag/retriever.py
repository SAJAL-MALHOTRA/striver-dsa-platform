"""
Retriever implementations for YouTube Lecture RAG.

Includes:
1. BM25Retriever (Lexical sparse search)
2. DenseRetriever (Vector cosine similarity using sentence-transformers)
3. HybridRetriever (Reciprocal Rank Fusion combining BM25 + Dense)

Every retriever conforms to the single contract expected by evaluate.py:
    retrieve(query: str, k: int) -> list[RetrievalResult]
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi

from .models import Chunk, RetrievalResult
from .phonetic_normalizer import normalize_dsa_text
from .query_expander import expand_dsa_query


BASE_DIR = Path(__file__).resolve().parent.parent


def simple_tokenize(text: str) -> list[str]:
    """Lowercase, phonetically normalize, and extract alphanumeric word tokens."""
    norm = normalize_dsa_text(text)
    return re.findall(r"\b\w+\b", norm.lower())


class BM25Retriever:
    """
    Lexical keyword retriever using BM25Okapi.
    Fast, deterministic, and requires zero neural network weights.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [simple_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        expanded_query = expand_dsa_query(query)
        tokens = simple_tokenize(expanded_query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)
        
        # Pair chunk with score and sort descending
        scored = list(zip(self.chunks, doc_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for chunk, score in scored[:k]:
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    retriever_name="bm25",
                )
            )
        return results


class DenseRetriever:
    """
    Dense semantic retriever using SentenceTransformers and brute-force cosine similarity.
    At the scale of thousands of lecture chunks, a normalized NumPy matrix product
    completes in single-digit milliseconds without needing a heavy vector database.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        lazy: bool = False,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.embeddings = None
        self._ready = False

        if not lazy:
            self.load()

    @property
    def is_ready(self) -> bool:
        return self._ready and self.embeddings is not None

    def load(self):
        if self._ready:
            return

        from sentence_transformers import SentenceTransformer

        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)

        chunk_signature = hashlib.sha256("".join(c.chunk_id for c in self.chunks).encode("utf-8")).hexdigest()[:16]
        filename = f"emb_{self.model_name.replace('/', '_')}_{len(self.chunks)}_{chunk_signature}.npy"

        # Check candidate locations: data/ first (precomputed in git), then eval/.cache/
        candidate_files = [
            BASE_DIR / "data" / filename,
            BASE_DIR / "eval" / ".cache" / filename,
        ]

        for cache_file in candidate_files:
            if cache_file.exists():
                try:
                    loaded = np.load(str(cache_file))
                    if loaded.shape[0] == len(self.chunks):
                        self.embeddings = loaded
                        self._ready = True
                        return
                except Exception:
                    pass

        if self.embeddings is None:
            texts = [c.text for c in self.chunks]
            self.embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            try:
                cache_dir = BASE_DIR / "eval" / ".cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_file = cache_dir / f"tmp_{filename}"
                with open(tmp_file, "wb") as f:
                    np.save(f, self.embeddings)
                tmp_file.replace(cache_dir / filename)
            except Exception:
                pass

        self._ready = True

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if not self._ready or self.embeddings is None or self.model is None:
            return []

        clean_query = normalize_dsa_text(query)
        query_emb = self.model.encode(
            [clean_query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]

        # Cosine similarity is dot product because embeddings are unit-normalized
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    retriever_name="dense",
                )
            )
        return results


class HybridRetriever:
    """
    Hybrid retriever that fuses BM25 lexical rankings with Dense semantic rankings
    using Reciprocal Rank Fusion (RRF).

    RRF_score(d) = sum(1 / (rrf_k + rank_i(d)))
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        rrf_k: int = 60,
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.chunks = bm25_retriever.chunks

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        # Retrieve a wider pool from each source (e.g. 20 candidates)
        pool_size = max(k * 4, 20)
        bm25_results = self.bm25.retrieve(query, k=pool_size)
        dense_results = self.dense.retrieve(query, k=pool_size)

        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        # Accumulate RRF scores from BM25 ranks (only positive keyword hits)
        for rank, res in enumerate(bm25_results):
            if res.score <= 0.0:
                continue
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + (rank + 1))

        # Accumulate RRF scores from Dense ranks (only above noise floor)
        for rank, res in enumerate(dense_results):
            if res.score < 0.20:
                continue
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + (rank + 1))

        # Sort chunks by fused RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for cid in sorted_cids[:k]:
            results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=float(rrf_scores[cid]),
                    retriever_name="hybrid_rrf",
                )
            )
        return results
