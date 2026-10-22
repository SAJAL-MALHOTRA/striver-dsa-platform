"""
Evaluation harness for timestamp-level retrieval.

This is the first thing we build, before any retriever exists.
Every retrieval strategy gets scored by the same functions,
so comparisons are apples-to-apples.

Metrics
-------
- Hit Rate@k : For each query, does ANY of the top-k retrieved chunks
  overlap with a ground-truth interval by at least `overlap_threshold`
  seconds?  Binary per query, averaged across all queries.

- MRR (Mean Reciprocal Rank) : For each query, what is the reciprocal
  of the rank of the FIRST hit?  If no hit in top-k, reciprocal rank = 0.

- Precision@k : Of the top-k chunks, what fraction are hits?

Why overlap-based matching
--------------------------
A retrieved chunk with interval [700, 775] and a ground-truth [720, 810]
share 55 seconds of overlap.  That's a hit — the student would hear the
right explanation.  Exact start-second matching is too brittle because
chunking boundaries are arbitrary.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import Chunk, GoldenQuery, RetrievalResult


# ---------------------------------------------------------------------------
# Overlap logic
# ---------------------------------------------------------------------------

def interval_overlap(
    a_vid: str, a_start: int, a_end: int,
    b_vid: str, b_start: int, b_end: int,
) -> int:
    """
    Seconds of overlap between two intervals.
    Returns 0 if they're from different videos or don't overlap.
    """
    if a_vid != b_vid:
        return 0
    overlap = min(a_end, b_end) - max(a_start, b_start)
    return max(0, overlap)


def is_hit(
    result: RetrievalResult,
    ground_truths: list[tuple[str, int, int]],
    overlap_threshold: int = 10,
) -> bool:
    """
    Does this retrieved chunk overlap with ANY ground-truth interval
    by at least `overlap_threshold` seconds?
    """
    c = result.chunk
    for vid, gt_start, gt_end in ground_truths:
        if interval_overlap(vid, gt_start, gt_end, c.video_id, c.start_sec, c.end_sec) >= overlap_threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-query scoring
# ---------------------------------------------------------------------------

@dataclass
class QueryScore:
    """Score for a single query against one retriever."""
    query_id: str
    query_text: str
    retriever: str
    k: int
    hit: bool                # did any of the top-k chunks hit?
    reciprocal_rank: float   # 1/rank_of_first_hit, or 0.0
    precision_at_k: float    # fraction of top-k that are hits
    top_k_results: list[dict] = field(default_factory=list)  # for inspection


def score_query(
    query: GoldenQuery,
    results: list[RetrievalResult],
    k: int = 3,
    overlap_threshold: int = 10,
) -> QueryScore:
    """
    Score a single query's retrieval results against ground truth.

    Parameters
    ----------
    query : GoldenQuery
        The query with ground-truth intervals.
    results : list[RetrievalResult]
        Retrieved chunks, already sorted by score descending.
    k : int
        Evaluate only the top-k results.
    overlap_threshold : int
        Minimum seconds of overlap to count as a hit.
    """
    top_k = results[:k]
    retriever = top_k[0].retriever_name if top_k else "unknown"

    hits = [is_hit(r, query.ground_truth_intervals, overlap_threshold) for r in top_k]

    # Hit Rate: binary — was there at least one hit?
    hit = any(hits)

    # MRR: reciprocal rank of the first hit
    reciprocal_rank = 0.0
    for i, h in enumerate(hits):
        if h:
            reciprocal_rank = 1.0 / (i + 1)
            break

    # Precision@k
    precision = sum(hits) / len(hits) if hits else 0.0

    # Keep details for debugging
    details = []
    for i, r in enumerate(top_k):
        details.append({
            "rank": i + 1,
            "chunk_id": r.chunk.chunk_id,
            "video_id": r.chunk.video_id,
            "interval": f"{r.chunk.timestamp} – {r.chunk.end_sec}s",
            "score": round(r.score, 4),
            "is_hit": hits[i] if i < len(hits) else False,
        })

    return QueryScore(
        query_id=query.query_id,
        query_text=query.query,
        retriever=retriever,
        k=k,
        hit=hit,
        reciprocal_rank=reciprocal_rank,
        precision_at_k=precision,
        top_k_results=details,
    )


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------

@dataclass
class EvalReport:
    """Aggregated evaluation across all queries for one retriever."""
    retriever: str
    k: int
    num_queries: int
    hit_rate: float        # average of per-query hit (0 or 1)
    mrr: float             # mean reciprocal rank
    mean_precision: float  # average precision@k
    per_query: list[QueryScore] = field(default_factory=list)

    def summary_row(self) -> dict:
        """One row for the ablation table."""
        return {
            "retriever": self.retriever,
            "k": self.k,
            "queries": self.num_queries,
            "hit_rate": f"{self.hit_rate:.1%}",
            "mrr": f"{self.mrr:.3f}",
            "precision@k": f"{self.mean_precision:.1%}",
        }


def evaluate(
    queries: list[GoldenQuery],
    retrieve_fn,
    retriever_name: str,
    k: int = 3,
    overlap_threshold: int = 10,
) -> EvalReport:
    """
    Run a full evaluation.

    Parameters
    ----------
    queries : list[GoldenQuery]
        The golden evaluation set.
    retrieve_fn : callable
        A function: (query_text: str, k: int) -> list[RetrievalResult].
        This is the ONLY interface a retriever must implement.
    retriever_name : str
        Label for this retriever (e.g., "bm25", "dense_bge_m3").
    k : int
        Number of results to retrieve and evaluate.
    overlap_threshold : int
        Minimum seconds of overlap to count as a hit.

    Returns
    -------
    EvalReport with aggregate metrics and per-query breakdowns.
    """
    scores: list[QueryScore] = []

    for query in queries:
        results = retrieve_fn(query.query, k)
        # Tag results with the retriever name
        tagged = [
            RetrievalResult(chunk=r.chunk, score=r.score, retriever_name=retriever_name)
            for r in results
        ]
        qs = score_query(query, tagged, k=k, overlap_threshold=overlap_threshold)
        scores.append(qs)

    num = len(scores)
    hit_rate = sum(s.hit for s in scores) / num if num else 0.0
    mrr = sum(s.reciprocal_rank for s in scores) / num if num else 0.0
    mean_prec = sum(s.precision_at_k for s in scores) / num if num else 0.0

    return EvalReport(
        retriever=retriever_name,
        k=k,
        num_queries=num,
        hit_rate=hit_rate,
        mrr=mrr,
        mean_precision=mean_prec,
        per_query=scores,
    )


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def ablation_table(reports: list[EvalReport]) -> str:
    """
    Format multiple EvalReports as a Markdown table for the README.

    This is the artifact that makes the project portfolio-worthy:
    a single table comparing every retrieval strategy on the same
    golden set with real numbers.
    """
    if not reports:
        return "(no reports)"

    header = "| Retriever | k | Queries | Hit Rate@k | MRR | Precision@k |"
    sep    = "|-----------|---|---------|------------|-----|-------------|"
    rows = [header, sep]
    for r in sorted(reports, key=lambda x: x.hit_rate, reverse=True):
        d = r.summary_row()
        rows.append(
            f"| {d['retriever']} | {d['k']} | {d['queries']} "
            f"| {d['hit_rate']} | {d['mrr']} | {d['precision@k']} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Golden set I/O
# ---------------------------------------------------------------------------

def load_golden_set(path: Path) -> list[GoldenQuery]:
    """Load the golden evaluation set from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = []
    for item in data:
        queries.append(GoldenQuery(
            query_id=item["query_id"],
            query=item["query"],
            ground_truth_intervals=[
                (gt["video_id"], gt["start_sec"], gt["end_sec"])
                for gt in item["ground_truth_intervals"]
            ],
            category=item.get("category", ""),
            query_type=item.get("query_type", ""),
            notes=item.get("notes", ""),
        ))
    return queries


def save_golden_set(queries: list[GoldenQuery], path: Path) -> None:
    """Save the golden evaluation set to a JSON file."""
    data = []
    for q in queries:
        data.append({
            "query_id": q.query_id,
            "query": q.query,
            "ground_truth_intervals": [
                {"video_id": vid, "start_sec": s, "end_sec": e}
                for vid, s, e in q.ground_truth_intervals
            ],
            "category": q.category,
            "query_type": q.query_type,
            "notes": q.notes,
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_report(report: EvalReport, path: Path) -> None:
    """Save a full eval report to JSON for later comparison."""
    data = {
        "retriever": report.retriever,
        "k": report.k,
        "num_queries": report.num_queries,
        "hit_rate": report.hit_rate,
        "mrr": report.mrr,
        "mean_precision": report.mean_precision,
        "per_query": [
            {
                "query_id": qs.query_id,
                "query": qs.query_text,
                "hit": qs.hit,
                "reciprocal_rank": qs.reciprocal_rank,
                "precision_at_k": qs.precision_at_k,
                "top_k_results": qs.top_k_results,
            }
            for qs in report.per_query
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
