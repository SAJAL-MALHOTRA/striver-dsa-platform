"""
Data models for the YT Lecture RAG system.

These are the shared types that every module speaks.
Retrieval, chunking, and evaluation all import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Video:
    """A single YouTube video."""
    video_id: str          # "dQw4w9WgXcQ"
    title: str
    duration_sec: int      # total length in seconds

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


@dataclass(frozen=True)
class Segment:
    """Raw Whisper output: a single transcribed segment with timing."""
    start: float           # seconds, from Whisper
    end: float
    text: str


@dataclass(frozen=True)
class Chunk:
    """
    What actually gets embedded and retrieved.

    chunk_id is deterministic so upserts are idempotent:
    same video + same start second = same chunk_id.
    """
    chunk_id: str          # f"{video_id}:{start_sec}"
    video_id: str
    video_title: str
    start_sec: int
    end_sec: int
    text: str

    @property
    def url(self) -> str:
        lead_sec = max(0, self.start_sec - 5)
        return f"https://www.youtube.com/watch?v={self.video_id}&t={lead_sec}s"

    @property
    def timestamp(self) -> str:
        """Human-readable timestamp: '12:04' or '1:12:04'."""
        m, s = divmod(self.start_sec, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"


@dataclass
class RetrievalResult:
    """
    A single retrieved chunk with its score.

    This is the output of any retriever — dense, BM25, hybrid, reranked.
    The evaluator doesn't care how the score was computed,
    only where the chunk lands in the video timeline.
    """
    chunk: Chunk
    score: float           # higher is better (normalise if needed)
    retriever_name: str    # "bm25", "dense", "hybrid", "hybrid+reranker"


@dataclass
class GoldenQuery:
    """
    One entry in the golden evaluation set.

    ground_truth_intervals is a list because a concept might be
    explained in more than one place across the playlist.
    Each interval is (video_id, start_sec, end_sec).
    """
    query_id: str
    query: str
    ground_truth_intervals: list[tuple[str, int, int]]  # (video_id, start, end)
    category: str = ""                     # "dsa", "ml", "dev"
    query_type: str = ""                   # "conceptual", "exact_keyword", "syntax"
    notes: str = ""                        # free-text, for your own reference
