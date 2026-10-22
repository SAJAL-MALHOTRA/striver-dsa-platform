#!/usr/bin/env python3
"""Diagnostics for the lecture retriever. Companion to eval_harness.py (same folder).

Three questions, answered from run files + golden set + transcripts:

 1. Candidate-pool ceiling: if a perfect reranker reordered the top-N candidates
    (per retriever, and the union of all retrievers), what Hit rate is reachable?
 2. Signed start offsets: for results that DO contain the answer, does the chunk
    start too early (user sits through lead-in) or too late (setup is missed)?
 3. Sub-chunk seek refinement: if we seek to the best-matching segment window
    INSIDE the retrieved chunk instead of the chunk start, how much of the
    seek-vs-overlap gap closes? Includes an oracle ceiling.

Example:
    python diagnostics.py --golden eval/golden.json --split dev \
        --runs bm25=runs/run_bm25.json dense=runs/run_dense.json hybrid=runs/run_hybrid.json \
        --transcripts transcripts/ --out diag_dev.md --csv diag_dev.csv

Hit definitions, dedup and loaders come from eval_harness.py, so the numbers are
directly comparable with its reports. Standard library only.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import eval_harness as eh
from eval_harness import HitConfig, Span, dedup, is_hit

OFFSET_BINS = [(-math.inf, -60), (-60, -20), (-20, 0), (0, 20), (20, 60), (60, math.inf)]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def frac(x: int, n: int) -> str:
    return f"{x}/{n} ({100 * x / n:.0f}%)" if n else "0/0"


def md_table(header, rows) -> list:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def quantile(sorted_vals, p: float):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def sha12(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def any_hit(results, q, cfg) -> bool:
    return any(is_hit(r, g, cfg) for r in results for g in q.relevant)


def first_rank_of(q, spans, cfg):
    for rank, r in enumerate(spans, start=1):
        if any(is_hit(r, g, cfg) for g in q.relevant):
            return rank
    return None


def ranked_for(run: dict, q, dedup_overlap: float, max_k: int) -> list:
    return dedup(run.get(q.id, []), dedup_overlap)[:max_k]


# --------------------------------------------------------------------------
# 1. candidate-pool ceiling
# --------------------------------------------------------------------------
def pool_recall(queries, runs: dict, n: int, cfg: HitConfig):
    """Raw top-n per retriever (NO dedup: this is what a reranker would receive).

    Returns ({source: hit_count, "UNION": hit_count}, mean_union_size).
    A perfect reranker over a pool reaches Hit@k == pool recall for every k.
    """
    out = {name: sum(any_hit(run.get(q.id, [])[:n], q, cfg) for q in queries)
           for name, run in runs.items()}
    union_hits, sizes = 0, []
    for q in queries:
        pool: dict = {}
        for run in runs.values():
            for r in run.get(q.id, [])[:n]:
                pool[(r.video_id, r.start, r.end)] = r
        sizes.append(len(pool))
        if any_hit(pool.values(), q, cfg):
            union_hits += 1
    out["UNION"] = union_hits
    return out, (mean(sizes) if sizes else 0.0)


# --------------------------------------------------------------------------
# 2. signed start offsets
# --------------------------------------------------------------------------
def classify(offset: float, gt_len: float, seek_cfg: HitConfig) -> str:
    if offset < -seek_cfg.lead_tol_sec:
        return "early"
    if offset > seek_cfg.seek_max_frac * gt_len:
        return "late"
    return "ok"


def start_offsets(queries, run: dict, overlap_cfg, seek_cfg, max_k, dedup_overlap) -> list:
    """For each query: the first deduped result that overlap-hits the answer.
    Returns [(query_id, rank, signed_offset_sec, 'early'|'ok'|'late')]."""
    rows = []
    for q in queries:
        for rank, r in enumerate(ranked_for(run, q, dedup_overlap, max_k), start=1):
            g = next((g for g in q.relevant if is_hit(r, g, overlap_cfg)), None)
            if g is not None:
                off = r.start - g.start
                rows.append((q.id, rank, off, classify(off, g.length, seek_cfg)))
                break
    return rows


# --------------------------------------------------------------------------
# 3. sub-chunk seek refinement
# --------------------------------------------------------------------------
class SegmentIndex:
    """BM25 over sliding windows of `window` consecutive transcript segments."""

    def __init__(self, transcripts: dict, window: int = 1, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.rows: dict = {}
        self.starts: dict = {}
        df: Counter = Counter()
        total = n = 0
        for vid, segs in transcripts.items():
            segs = sorted(segs, key=lambda s: s[0])
            rows = []
            for i, seg in enumerate(segs):
                toks = [t for s in segs[i:i + window] for t in eh.tokens(s[2])]
                tf = Counter(toks)
                rows.append((seg[0], tf, len(toks)))
                df.update(tf.keys())
                total += len(toks)
                n += 1
            self.rows[vid] = rows
            self.starts[vid] = [r[0] for r in rows]
        self.df, self.n = df, n
        self.avgdl = (total / n) if n and total else 1.0

    def _score(self, qtoks, tf, dl) -> float:
        s = 0.0
        for t in qtoks:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def _indices(self, r: Span) -> range:
        starts = self.starts.get(r.video_id)
        if not starts:
            return range(0)
        return range(bisect.bisect_left(starts, r.start - 0.5), bisect.bisect_left(starts, r.end))

    def segment_starts(self, r: Span) -> list:
        starts = self.starts.get(r.video_id, [])
        return [starts[i] for i in self._indices(r)]

    def refine(self, r: Span, qtoks, pad: float) -> Span:
        """Move the seek point to the best-scoring window inside the chunk.
        Falls back to the chunk start when there is no transcript or no lexical evidence."""
        idxs = self._indices(r)
        if not idxs or not qtoks:
            return r
        rows = self.rows[r.video_id]
        best_i, best = None, 0.0
        for i in idxs:
            s = self._score(qtoks, rows[i][1], rows[i][2])
            if s > best + 1e-12:   # ties keep the earliest window
                best_i, best = i, s
        if best_i is None:
            return r
        return Span(r.video_id, max(r.start, rows[best_i][0] - pad), r.end)


def oracle_hit(r: Span, q, index: SegmentIndex, seek_cfg: HitConfig) -> bool:
    """Ceiling for refinement: is ANY segment start inside the chunk a seek-hit?"""
    starts = [r.start] + index.segment_starts(r)
    return any(is_hit(Span(r.video_id, s, r.end), g, seek_cfg) for s in starts for g in q.relevant)


def refined_hits(queries, run, index, window_index_pad, seek_cfg, ks, max_k, dedup_overlap) -> dict:
    """Seek Hit@k after refinement. window_index_pad = pad seconds."""
    hits = {k: 0 for k in ks}
    for q in queries:
        qtoks = set(eh.tokens(q.query))
        ranked = ranked_for(run, q, dedup_overlap, max_k)
        refined = [index.refine(r, qtoks, window_index_pad) for r in ranked]
        fr = first_rank_of(q, refined, seek_cfg)
        for k in ks:
            hits[k] += int(fr is not None and fr <= k)
    return hits


def oracle_hits(queries, run, index, seek_cfg, ks, max_k, dedup_overlap) -> dict:
    hits = {k: 0 for k in ks}
    for q in queries:
        ranked = ranked_for(run, q, dedup_overlap, max_k)
        fr = next((i for i, r in enumerate(ranked, 1) if oracle_hit(r, q, index, seek_cfg)), None)
        for k in ks:
            hits[k] += int(fr is not None and fr <= k)
    return hits


def plain_hits(queries, run, cfg, ks, max_k, dedup_overlap) -> dict:
    hits = {k: 0 for k in ks}
    for q in queries:
        fr = first_rank_of(q, ranked_for(run, q, dedup_overlap, max_k), cfg)
        for k in ks:
            hits[k] += int(fr is not None and fr <= k)
    return hits


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def build_report(queries, golden_sha, runs, overlap_cfg, seek_cfg, ks, primary_k, pool_ns,
                 dedup_overlap, transcripts, windows, pads, split, note=None) -> str:
    n = len(queries)
    max_k = max(ks)
    md = ["# Retriever diagnostics", "",
          f"- golden `sha256:{golden_sha[:12]}`, {n} queries, split = **{split}**",
          f"- eval_harness.py `sha256:{sha12(eh.__file__)}` · diagnostics.py `sha256:{sha12(__file__)}`",
          f"- dedup overlap {dedup_overlap:.0%}; seek window: start in "
          f"[gt.start - {seek_cfg.lead_tol_sec:g}s, gt.start + {seek_cfg.seek_max_frac:g} x len(gt)]"]
    md += ([f"- {note}"] if note else []) + [""]
    if split == "test":
        md += ["> TEST split: do not choose refinement parameters from this output.", ""]
    if n < 30:
        md += [f"> n = {n}: every count below is within a few queries of noise. Read patterns, not rankings.", ""]

    # ---- 1. pool ceiling
    md += ["## 1. Candidate-pool ceiling (what a perfect reranker could reach)", "",
           "Raw top-N per retriever, no dedup. UNION merges all retrievers' top-N. "
           "A perfect reranker over a pool reaches this hit rate at every k.", ""]
    for dname, cfg in (("overlap", overlap_cfg), ("seek", seek_cfg)):
        header = ["source"] + [f"top-{N}" for N in pool_ns]
        per_n = {N: pool_recall(queries, runs, N, cfg) for N in pool_ns}
        rows = [[name] + [frac(per_n[N][0][name], n) for N in pool_ns] for name in list(runs) + ["UNION"]]
        rows.append(["mean union size"] + [f"{per_n[N][1]:.1f}" for N in pool_ns])
        md += [f"**Hit definition: `{dname}`**", ""] + md_table(header, rows) + [""]
    big = max(pool_ns)
    pool_overlap = pool_recall(queries, runs, big, overlap_cfg)[0]
    md += [f"Headroom under `overlap` (current deduped Hit@{primary_k} vs top-{big} pool):", ""]
    for name, run in runs.items():
        cur = plain_hits(queries, run, overlap_cfg, [primary_k], max_k, dedup_overlap)[primary_k]
        md.append(f"- {name}: {cur} -> {pool_overlap[name]} queries "
                  f"(+{pool_overlap[name] - cur} possible from reranking this retriever alone); "
                  f"union ceiling {pool_overlap['UNION']}")
    md.append("")

    # ---- 2. offsets
    md += ["## 2. Where do chunk starts land? (first result that contains the answer)", "",
           "Signed offset = result.start - gt.start. Negative = chunk starts before the answer "
           "(lead-in), positive = starts inside it (setup missed).", ""]
    summary_rows, bin_rows = [], []
    for name, run in runs.items():
        rows = start_offsets(queries, run, overlap_cfg, seek_cfg, max_k, dedup_overlap)
        offs = sorted(r[2] for r in rows)
        cls = Counter(r[3] for r in rows)
        q_ = lambda p: "-" if not offs else f"{quantile(offs, p):+.0f}"
        summary_rows.append([name, frac(len(rows), n), cls["early"], cls["ok"], cls["late"],
                             q_(0.1), q_(0.25), q_(0.5), q_(0.75), q_(0.9)])
        bin_rows.append([name] + [sum(1 for o in offs if lo < o <= hi) for lo, hi in OFFSET_BINS])
    md += md_table(["retriever", "contains answer", "early", "ok (seek hit)", "late",
                    "p10", "p25", "median", "p75", "p90 (s)"], summary_rows) + ["", "Histogram (seconds):", ""]
    labels = ["<=-60", "-60..-20", "-20..0", "0..20", "20..60", ">60"]
    md += md_table(["retriever"] + labels, bin_rows) + [""]

    # ---- 3. refinement
    md += ["## 3. Sub-chunk seek refinement", ""]
    if transcripts is None:
        md += ["Skipped: pass --transcripts to evaluate refinement.", ""]
        return "\n".join(md)
    md += ["Seek to the best-matching transcript window inside each retrieved chunk "
           "(BM25 over `w` consecutive segments, minus `pad` seconds, never before the chunk start). "
           "Rows are seek Hit@k. `oracle` is the ceiling: any segment start inside the chunk that lands "
           "in the seek window. Every (w, pad) pair is another experiment on this split; keep the grid small.", ""]
    kset = sorted({1, primary_k})
    header = ["strategy"] + [f"{name} @{k}" for name in runs for k in kset]

    def row(label, fn):
        cells = []
        for run in runs.values():
            h = fn(run)
            cells += [frac(h[k], n) for k in kset]
        return [label] + cells

    table = [row("overlap ceiling (chunk contains answer)",
                 lambda run: plain_hits(queries, run, overlap_cfg, kset, max_k, dedup_overlap)),
             row("chunk_start (current seek)",
                 lambda run: plain_hits(queries, run, seek_cfg, kset, max_k, dedup_overlap))]
    for w in windows:
        index = SegmentIndex(transcripts, window=w)
        for pad in pads:
            table.append(row(f"refine w={w} pad={pad:g}s",
                             lambda run, index=index, pad=pad:
                             refined_hits(queries, run, index, pad, seek_cfg, kset, max_k, dedup_overlap)))
    oracle_index = SegmentIndex(transcripts, window=1)
    table.append(row("oracle (best segment start in chunk)",
                     lambda run: oracle_hits(queries, run, oracle_index, seek_cfg, kset, max_k, dedup_overlap)))
    md += md_table(header, table) + [""]
    md += ["How to read: the gap between `chunk_start` and `overlap ceiling` is what refinement is "
           "trying to close; `oracle` shows how much of it is closable at all from segment boundaries. "
           "If `refine` approaches `oracle`, snap-to-best-window is worth shipping; if `oracle` itself "
           "is far below the ceiling, chunk length or stride is the problem. `oracle` can exceed the "
           "overlap ceiling: a chunk may hold the start of the answer without 30s of it. Refinement "
           "scores lexically (BM25), so it falls back to the chunk start when the query shares no words "
           "with the chunk; expect little gain on paraphrase and symptom queries.", ""]
    return "\n".join(md)


def write_csv(path, queries, runs, overlap_cfg, seek_cfg, max_k, dedup_overlap):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "type", "retriever", "overlap_rank", "seek_rank", "offset_sec", "offset_class"])
        for name, run in runs.items():
            offs = {r[0]: r for r in start_offsets(queries, run, overlap_cfg, seek_cfg, max_k, dedup_overlap)}
            for q in queries:
                ranked = ranked_for(run, q, dedup_overlap, max_k)
                o = offs.get(q.id)
                w.writerow([q.id, q.type, name, first_rank_of(q, ranked, overlap_cfg) or "",
                            first_rank_of(q, ranked, seek_cfg) or "",
                            f"{o[2]:.1f}" if o else "", o[3] if o else ""])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--runs", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--transcripts")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--primary-k", type=int, default=3)
    ap.add_argument("--pool-n", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--dedup-overlap", type=float, default=0.5)
    ap.add_argument("--lead-tol", type=float, default=20.0)
    ap.add_argument("--seek-max-frac", type=float, default=0.5)
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--overlap-cap", type=float, default=30.0)
    ap.add_argument("--refine-windows", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--refine-pads", nargs="+", type=float, default=[3.0])
    ap.add_argument("--out")
    ap.add_argument("--csv")
    ap.add_argument("--verified-only", action="store_true",
                    help="use only queries a human verified (verified=true in the golden file)")
    ap.add_argument("--allow-test", action="store_true",
                    help="permit diagnostics on the test split; this is a look at held-out data and counts as using it")
    args = ap.parse_args(argv)

    if args.split in ("test", "all") and not args.allow_test:
        print("error: diagnostics on the test split are a peek at held-out data. Use --split dev, or pass "
              "--allow-test if you accept that the test split is no longer untouched.", file=sys.stderr)
        return 2
    try:
        queries, sha = eh.load_golden(args.golden)
    except eh.GoldenError as e:
        print("Golden file is invalid:", *e.errors, sep="\n  - ", file=sys.stderr)
        return 2
    selected, label_note, sel_err = eh.select_queries(queries, args.split, args.verified_only)
    if sel_err:
        print(sel_err, file=sys.stderr)
        return 2
    if not selected:
        print(f"No queries in split '{args.split}'" + (" after the verified-only filter." if args.verified_only else "."),
              file=sys.stderr)
        return 2
    runs = {}
    for item in args.runs:
        name, sep, path = item.partition("=")
        if not sep:
            print(f"--runs entries must look like NAME=PATH, got {item!r}", file=sys.stderr)
            return 2
        runs[name] = eh.load_run(path)
    transcripts = eh.load_transcripts(args.transcripts) if args.transcripts else None
    ks = sorted(set(args.k) | {args.primary_k})
    overlap_cfg = HitConfig("overlap", args.lead_tol, args.seek_max_frac, args.overlap_frac, args.overlap_cap)
    seek_cfg = HitConfig("seek", args.lead_tol, args.seek_max_frac, args.overlap_frac, args.overlap_cap)

    report = build_report(selected, sha, runs, overlap_cfg, seek_cfg, ks, args.primary_k,
                          sorted(set(args.pool_n)), args.dedup_overlap, transcripts,
                          args.refine_windows, args.refine_pads, args.split, label_note)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    if args.csv:
        write_csv(args.csv, selected, runs, overlap_cfg, seek_cfg, max(ks), args.dedup_overlap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
