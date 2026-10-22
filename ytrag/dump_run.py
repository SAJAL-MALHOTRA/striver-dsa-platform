"""
Bridge utility to dump run files from ytrag retrievers for eval_harness.py.

Conforms to section 10 of LABELING_PROTOCOL.md:
    {"q001": [{"video_id": "v1", "start_sec": 118.0, "end_sec": 193.0}, ...]}
"""

import json
from pathlib import Path
from typing import Callable
from .models import RetrievalResult


def dump_run(
    retrieve_fn: Callable[[str, int], list[RetrievalResult]],
    golden_path: Path,
    out_path: Path,
    n: int = 30,
) -> None:
    """
    Execute retrieve_fn on each query in the golden file and dump a run JSON.
    
    Parameters
    ----------
    retrieve_fn : callable
        Function: (query: str, n: int) -> list[RetrievalResult]
    golden_path : Path
        Path to golden.json containing {"queries": [...]}
    out_path : Path
        Destination path for the run JSON.
    n : int
        Number of raw results to dump per query (default 30, as required for dedup).
    """
    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    queries = data.get("queries", data)
    run_dict: dict[str, list[dict]] = {}

    for q in queries:
        qid = q["id"]
        qtext = q["query"]
        results = retrieve_fn(qtext, n)
        
        run_dict[qid] = [
            {
                "video_id": r.chunk.video_id,
                "start_sec": float(r.chunk.start_sec),
                "end_sec": float(r.chunk.end_sec),
            }
            for r in results
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run_dict, f, indent=2)
    print(f"Dumped {len(run_dict)} queries to {out_path} (raw n={n})")
