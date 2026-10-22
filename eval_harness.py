#!/usr/bin/env python3
"""Strict, bias-aware evaluation for timestamped lecture retrieval.

Pure standard library. Works on *run files* (ranked results dumped from any
retriever), so it never imports your retrieval code.

Golden file  (see LABELING_PROTOCOL.md):
    {"queries": [{"id": "q001", "query": "...", "type": "paraphrase",
                  "split": "dev",
                  "relevant": [{"video_id": "v1", "start_sec": 140, "end_sec": 245}]}]}

Run file (one per retriever, ranked best-first, dump >= 30 raw results/query):
    {"q001": [{"video_id": "v1", "start_sec": 118.0, "end_sec": 193.0}, ...]}

Example:
    python eval_harness.py --golden golden.json --split dev \
        --runs bm25=runs_bm25.json dense=runs_dense.json hybrid=runs_hybrid.json \
        --compare bm25 hybrid --transcripts transcripts/ --out report.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

ALLOWED_TYPES = ("lexical", "paraphrase", "conceptual", "symptom")
ALLOWED_SPLITS = ("dev", "test")
MIN_SPAN_SEC = 5.0
LONG_SPAN_SEC = 240.0


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    video_id: str
    start: float
    end: float

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    type: str
    split: str
    relevant: tuple  # tuple[Span, ...]
    verified: bool | None = None   # None: the golden file carries no verification flag


@dataclass(frozen=True)
class HitConfig:
    """Formal definition of "this result is a hit for this ground-truth span".

    seek    (default; matches what the UI does, player.seekTo(result.start)):
            hit iff same video and
            gt.start - lead_tol_sec <= result.start <= gt.start + seek_max_frac * len(gt)
            i.e. the user lands shortly before the answer, with most of it still ahead.
    overlap hit iff same video and
            intersection(result, gt) >= min(overlap_frac * len(gt), overlap_cap_sec)
    """
    definition: str = "seek"
    lead_tol_sec: float = 20.0
    seek_max_frac: float = 0.5
    overlap_frac: float = 0.5
    overlap_cap_sec: float = 30.0


class GoldenError(Exception):
    def __init__(self, errors):
        super().__init__("\n".join(errors))
        self.errors = list(errors)


# --------------------------------------------------------------------------
# Hit definition and temporal deduplication
# --------------------------------------------------------------------------
def overlap_sec(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def is_hit(result: Span, gt: Span, cfg: HitConfig) -> bool:
    if result.video_id != gt.video_id:
        return False
    if cfg.definition == "seek":
        lo = gt.start - cfg.lead_tol_sec
        hi = gt.start + cfg.seek_max_frac * gt.length
        return lo <= result.start <= hi
    if cfg.definition == "overlap":
        need = min(cfg.overlap_frac * gt.length, cfg.overlap_cap_sec)
        return overlap_sec(result.start, result.end, gt.start, gt.end) >= need
    raise ValueError(f"unknown hit definition: {cfg.definition!r}")


def dedup(results: list, overlap_frac: float) -> list:
    """Temporal NMS. Walk results best-first; drop a result if it overlaps an
    already-kept result in the same video by >= overlap_frac of the SHORTER
    span. overlap_frac <= 0 disables deduplication."""
    if overlap_frac <= 0:
        return list(results)
    kept: list = []
    for r in results:
        duplicate = False
        for k in kept:
            if k.video_id != r.video_id:
                continue
            ov = overlap_sec(k.start, k.end, r.start, r.end)
            if ov >= overlap_frac * min(k.length, r.length):
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
    return kept


# --------------------------------------------------------------------------
# Per-query scoring
# --------------------------------------------------------------------------
@dataclass
class QueryScore:
    first_rank: int | None
    start_error_sec: float | None
    span_ranks: tuple   # per relevant span: rank of its first hit, or None
    ranked: tuple       # results actually scored (deduped, truncated)


def score_query(q: Query, raw_results: list, cfg: HitConfig, max_k: int,
                dedup_overlap: float) -> QueryScore:
    ranked = dedup(raw_results, dedup_overlap)[:max_k]
    span_ranks: list = [None] * len(q.relevant)
    first_rank = None
    start_error = None
    for rank, r in enumerate(ranked, start=1):
        for gi, g in enumerate(q.relevant):
            if is_hit(r, g, cfg):
                if span_ranks[gi] is None:
                    span_ranks[gi] = rank
                if first_rank is None:
                    first_rank = rank
                    start_error = abs(r.start - g.start)
    return QueryScore(first_rank, start_error, tuple(span_ranks), tuple(ranked))


def hit_at(s: QueryScore, k: int) -> bool:
    return s.first_rank is not None and s.first_rank <= k


def rr_at(s: QueryScore, k: int) -> float:
    return 1.0 / s.first_rank if hit_at(s, k) else 0.0


def recall_at(s: QueryScore, k: int) -> float:
    found = sum(1 for r in s.span_ranks if r is not None and r <= k)
    return found / len(s.span_ranks)


def evaluate_run(queries, run: dict, cfg: HitConfig, max_k: int,
                 dedup_overlap: float) -> dict:
    return {q.id: score_query(q, run.get(q.id, []), cfg, max_k, dedup_overlap)
            for q in queries}


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def wilson(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    values = list(values)
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value on discordant pairs (b = A-only, c = B-only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare_runs(queries, sa: dict, sb: dict, k: int, n_boot: int, seed: int) -> dict:
    ids = [q.id for q in queries]
    ha = [hit_at(sa[i], k) for i in ids]
    hb = [hit_at(sb[i], k) for i in ids]
    a_only = [i for i, x, y in zip(ids, ha, hb) if x and not y]
    b_only = [i for i, x, y in zip(ids, ha, hb) if y and not x]
    both = sum(1 for x, y in zip(ha, hb) if x and y)
    neither = sum(1 for x, y in zip(ha, hb) if not x and not y)
    diffs = [rr_at(sa[i], k) - rr_at(sb[i], k) for i in ids]
    lo, hi = bootstrap_ci(diffs, n_boot, seed)
    return {"k": k, "a_only": a_only, "b_only": b_only, "both": both,
            "neither": neither, "p_mcnemar": mcnemar_exact(len(a_only), len(b_only)),
            "mrr_diff": mean(diffs) if diffs else 0.0, "mrr_diff_ci": [lo, hi]}


def summarize(queries, scores: dict, ks, n_boot: int, seed: int):
    rows = []
    for k in ks:
        hits = [1.0 if hit_at(scores[q.id], k) else 0.0 for q in queries]
        rrs = [rr_at(scores[q.id], k) for q in queries]
        recs = [recall_at(scores[q.id], k) for q in queries]
        h_lo, h_hi = wilson(int(sum(hits)), len(hits))
        m_lo, m_hi = bootstrap_ci(rrs, n_boot, seed)
        rows.append({"k": k, "n": len(hits), "hit": mean(hits), "hit_ci": [h_lo, h_hi],
                     "mrr": mean(rrs), "mrr_ci": [m_lo, m_hi], "span_recall": mean(recs)})
    errs = [scores[q.id].start_error_sec for q in queries
            if scores[q.id].start_error_sec is not None]
    return rows, (median(errs) if errs else None), len(errs)


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def load_golden(path):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    items = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise GoldenError(["golden file must be a list or {'queries': [...]}"])
    errors: list = []
    queries: list = []
    seen: set = set()
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"entry #{idx}: not an object")
            continue
        qid = str(item.get("id", f"#{idx}"))
        where = f"query {qid}"
        if qid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(qid)
        text = str(item.get("query", "")).strip()
        if not text:
            errors.append(f"{where}: empty 'query'")
        qtype, split = item.get("type"), item.get("split")
        if qtype not in ALLOWED_TYPES:
            errors.append(f"{where}: type {qtype!r} not in {ALLOWED_TYPES}")
        if split not in ALLOWED_SPLITS:
            errors.append(f"{where}: split {split!r} not in {ALLOWED_SPLITS}")
        rel = item.get("relevant")
        spans: list = []
        if not isinstance(rel, list) or not rel:
            errors.append(f"{where}: 'relevant' must be a non-empty list")
        else:
            for s in rel:
                try:
                    sp = Span(str(s["video_id"]), float(s["start_sec"]), float(s["end_sec"]))
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{where}: malformed span {s!r}")
                    continue
                if sp.length < MIN_SPAN_SEC:
                    errors.append(f"{where}: span {sp.start}-{sp.end} shorter than {MIN_SPAN_SEC}s")
                spans.append(sp)
        verified = item.get("verified")
        if verified is not None and not isinstance(verified, bool):
            errors.append(f"{where}: 'verified' must be true or false, got {verified!r}")
            verified = None
        queries.append(Query(qid, text, qtype, split, tuple(spans), verified))
    if errors:
        raise GoldenError(errors)
    return queries, sha


def select_queries(queries, split: str, verified_only: bool = False):
    """-> (selected, note, error). `split` is dev, test or all.
    verified_only keeps only queries a human verified (verified == True in the golden file)."""
    pool = list(queries) if split == "all" else [q for q in queries if q.split == split]
    if not verified_only:
        return pool, None, None
    if not any(q.verified is not None for q in queries):
        return [], None, ("--verified-only needs a golden file with verification flags "
                          "(produce one with verify_labels.py apply)")
    kept = [q for q in pool if q.verified is True]
    note = (f"label filter: verified-only ({len(kept)} of {len(pool)} {split} queries; "
            f"{len(pool) - len(kept)} unverified excluded)")
    return kept, note, None


def golden_warnings(queries) -> list:
    w: list = []
    n = len(queries)
    if n < 50:
        w.append(f"only {n} queries; aim for 50+ before drawing conclusions")
    if not n:
        return w
    unverified = sum(1 for q in queries if q.verified is False)
    if unverified:
        w.append(f"{unverified} of {n} queries are marked unverified; use --verified-only for final numbers")
    by_type = Counter(q.type for q in queries)
    for t in ALLOWED_TYPES:
        if by_type[t] / n < 0.10:
            w.append(f"query type '{t}' is only {by_type[t]}/{n}; the set is not balanced")
    videos = Counter(sp.video_id for q in queries for sp in {s.video_id: s for s in q.relevant}.values())
    if len(videos) < 5:
        w.append(f"only {len(videos)} distinct videos; protocol asks for 5+")
    for vid, c in videos.items():
        if c / n > 0.20:
            w.append(f"video '{vid}' holds {c}/{n} queries (>20%)")
    if not any(q.split == "test" for q in queries):
        w.append("no 'test' split queries; you have nothing held out")
    for q in queries:
        for sp in q.relevant:
            if sp.length > LONG_SPAN_SEC:
                w.append(f"{q.id}: span of {sp.length:.0f}s is long; consider splitting it")
    return w


def load_run(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    run: dict = {}
    for qid, results in data.items():
        spans = []
        for r in results:
            sp = Span(str(r["video_id"]), float(r["start_sec"]), float(r["end_sec"]))
            if sp.length <= 0:
                raise ValueError(f"{path}: query {qid}: result with non-positive length")
            spans.append(sp)
        run[qid] = spans
    return run


# --------------------------------------------------------------------------
# Lexical-leakage audit (does the query reuse the answer's own words?)
# --------------------------------------------------------------------------
STOPWORDS = set("""a an and are as at be but by can do does for from how i if in into is it its
me my of on or so than that the their then there these they this to up use used using was we
what when where which who why will with without you your""".split())


def _norm(tok: str) -> str:
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def tokens(text: str) -> list:
    return [_norm(t) for t in re.findall(r"[a-z0-9]+", text.lower())
            if len(t) > 1 and t not in STOPWORDS]


def load_transcripts(directory) -> dict:
    out: dict = {}
    for p in Path(directory).glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        segs = data["segments"] if isinstance(data, dict) else data
        out[p.stem] = [
            (
                float(s["start"]),
                float(s.get("end", s["start"] + s.get("duration", 0.0))),
                str(s["text"]),
            )
            for s in segs
        ]
    return out


def lexical_overlap(q: Query, transcripts: dict):
    """Fraction of the query's content words that appear in the ground-truth span."""
    qt = set(tokens(q.query))
    if not qt:
        return None
    pool: set = set()
    found = False
    for g in q.relevant:
        segs = transcripts.get(g.video_id)
        if segs is None:
            continue
        found = True
        for s, e, t in segs:
            if e > g.start and s < g.end:
                pool.update(tokens(t))
    return len(qt & pool) / len(qt) if found else None


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def run_report(queries, sha, runs: dict, cfgs: dict, ks, primary_k, dedup_overlap,
               compare, transcripts, n_boot, seed, split, warnings, notes=None):
    md: list = []
    js: dict = {"golden_sha256": sha, "n_queries": len(queries), "split": split,
                "dedup_overlap": dedup_overlap, "ks": list(ks), "primary_k": primary_k,
                "hit_definitions": {n: vars(c) for n, c in cfgs.items()}, "notes": list(notes or []),
                "results": {}}
    md += ["# Retrieval evaluation report", "",
           f"- golden: `sha256:{sha[:12]}`, {len(queries)} queries, split = **{split}**",
           f"- temporal dedup: overlap >= {dedup_overlap:.0%} of the shorter span"
           if dedup_overlap > 0 else "- temporal dedup: OFF",
           f"- primary k = {primary_k}; CIs: Wilson (hit rate), bootstrap (MRR), {n_boot} resamples"]
    md += [f"- {x}" for x in (notes or [])] + [""]
    if split == "test":
        md += ["> TEST split: evaluate each final configuration once. Do not tune against it.", ""]
    if warnings:
        md += ["## Golden-set warnings", ""] + [f"- {x}" for x in warnings] + [""]

    if transcripts is not None:
        md += ["## Lexical-leakage audit", ""]
        vals = [(q, lexical_overlap(q, transcripts)) for q in queries]
        scored = [(q, v) for q, v in vals if v is not None]
        if scored:
            md.append(f"Mean query-to-answer word overlap: **{mean(v for _, v in scored):.2f}** "
                      f"over {len(scored)} queries ({len(vals) - len(scored)} skipped: no transcript).")
            md.append("")
            md += ["| type | n | mean overlap | share >= 0.8 |", "|---|---:|---:|---:|"]
            for t in ALLOWED_TYPES:
                sub = [v for q, v in scored if q.type == t]
                if sub:
                    md.append(f"| {t} | {len(sub)} | {mean(sub):.2f} | "
                              f"{sum(1 for v in sub if v >= 0.8) / len(sub):.0%} |")
            low = sum(1 for _, v in scored if v < 0.5) / len(scored)
            md.append("")
            if low < 0.30:
                md.append(f"WARNING: only {low:.0%} of queries have <50% overlap with their answer. "
                          "The set favors lexical retrieval (BM25) by construction.")
            leaky = [(q, v) for q, v in scored if v >= 0.8]
            if leaky:
                md += ["", "Highly leaky queries (>= 0.8):"]
                md += [f"- {q.id} ({v:.2f}): {q.query}" for q, v in leaky[:15]]
            js["lexical_audit"] = {"mean": mean(v for _, v in scored), "share_below_0.5": low}
        else:
            md.append("No transcripts matched the golden video ids.")
        md.append("")

    scores_by_def: dict = {}
    for dname, cfg in cfgs.items():
        md += [f"## Results: hit definition `{dname}`", "", describe_cfg(cfg), ""]
        scores_by_def[dname] = {}
        js["results"][dname] = {}
        md += ["| retriever | k | Hit@k | 95% CI | MRR@k | 95% CI | span recall@k |",
               "|---|---:|---:|---|---:|---|---:|"]
        extra: list = []
        for name, run in runs.items():
            sc = evaluate_run(queries, run, cfg, max(ks), dedup_overlap)
            scores_by_def[dname][name] = sc
            rows, med_err, n_err = summarize(queries, sc, ks, n_boot, seed)
            js["results"][dname][name] = {"rows": rows, "median_start_error_sec": med_err}
            for r in rows:
                md.append(f"| {name} | {r['k']} | {pct(r['hit'])} | "
                          f"{pct(r['hit_ci'][0])} to {pct(r['hit_ci'][1])} | {r['mrr']:.3f} | "
                          f"{r['mrr_ci'][0]:.3f} to {r['mrr_ci'][1]:.3f} | {pct(r['span_recall'])} |")
            if med_err is not None:
                extra.append(f"- {name}: median |start error| of first hit = {med_err:.1f}s (n={n_err})")
        if extra:
            md += ["", "Seek precision:"] + extra

        md += ["", f"### Hit@{primary_k} by query type", "",
               "| type | n | " + " | ".join(runs) + " |", "|---|---:|" + "---:|" * len(runs)]
        for t in ALLOWED_TYPES:
            sub = [q for q in queries if q.type == t]
            if not sub:
                continue
            cells = [pct(mean(1.0 if hit_at(scores_by_def[dname][n][q.id], primary_k) else 0.0
                              for q in sub)) for n in runs]
            md.append(f"| {t} | {len(sub)} | " + " | ".join(cells) + " |")

        missed_all = [q for q in queries
                      if not any(hit_at(scores_by_def[dname][n][q.id], primary_k) for n in runs)]
        if missed_all:
            md += ["", f"### Missed by every retriever at k={primary_k} (check labels first)", ""]
            md += [f"- {q.id} [{q.type}]: {q.query}" for q in missed_all]

        if compare:
            a, b = compare
            c = compare_runs(queries, scores_by_def[dname][a], scores_by_def[dname][b],
                             primary_k, n_boot, seed)
            js["results"][dname][f"compare:{a}-vs-{b}"] = c
            md += ["", f"### Paired comparison at k={primary_k}: {a} vs {b}", "",
                   f"- {a} only: {len(c['a_only'])} · {b} only: {len(c['b_only'])} · "
                   f"both: {c['both']} · neither: {c['neither']}",
                   f"- exact McNemar p = {c['p_mcnemar']:.4f}",
                   f"- MRR difference ({a} - {b}) = {c['mrr_diff']:+.3f} "
                   f"(95% CI {c['mrr_diff_ci'][0]:+.3f} to {c['mrr_diff_ci'][1]:+.3f})"]
            lookup = {q.id: q for q in queries}
            for label, ids in ((a, c["a_only"]), (b, c["b_only"])):
                if ids:
                    md += ["", f"Only {label} found:"] + [f"- {i}: {lookup[i].query}" for i in ids]
        md.append("")
    return "\n".join(md), js


def describe_cfg(cfg: HitConfig) -> str:
    if cfg.definition == "seek":
        return (f"Hit iff same video and gt.start - {cfg.lead_tol_sec:g}s <= result.start <= "
                f"gt.start + {cfg.seek_max_frac:g} x len(gt).")
    return (f"Hit iff same video and overlap >= min({cfg.overlap_frac:g} x len(gt), "
            f"{cfg.overlap_cap_sec:g}s).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--runs", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument("--primary-k", type=int, default=3)
    ap.add_argument("--hit-def", choices=["seek", "overlap", "both"], default="both")
    ap.add_argument("--lead-tol", type=float, default=20.0)
    ap.add_argument("--seek-max-frac", type=float, default=0.5)
    ap.add_argument("--overlap-frac", type=float, default=0.5)
    ap.add_argument("--overlap-cap", type=float, default=30.0)
    ap.add_argument("--dedup-overlap", type=float, default=0.5, help="0 disables")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--transcripts", help="dir of <video_id>.json transcripts for the leakage audit")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--json-out")
    ap.add_argument("--verified-only", action="store_true",
                    help="score only queries a human verified (verified=true in the golden file)")
    ap.add_argument("--prereg", help="pre-registration lock (preregister.py); required for the test split")
    ap.add_argument("--unregistered", action="store_true",
                    help="run the test split WITHOUT a lock; the report is stamped as exploratory")
    ap.add_argument("--allow-rerun", action="store_true",
                    help="allow a second test run under a lock; the report is stamped as a re-run")
    args = ap.parse_args(argv)

    try:
        queries, sha = load_golden(args.golden)
    except GoldenError as e:
        print("Golden file is invalid:", *e.errors, sep="\n  - ", file=sys.stderr)
        return 2
    warnings = golden_warnings(queries)
    if args.validate_only:
        selected, label_note, sel_err = select_queries(queries, "all", False)
    else:
        selected, label_note, sel_err = select_queries(queries, args.split, args.verified_only)
    if sel_err:
        print(sel_err, file=sys.stderr)
        return 2
    if not selected:
        print(f"No queries to score in split '{args.split}'"
              + (" after the verified-only filter." if args.verified_only else "."), file=sys.stderr)
        return 2
    transcripts = load_transcripts(args.transcripts) if args.transcripts else None
    ks = sorted(set(args.k) | {args.primary_k})

    runs: dict = {}
    run_paths: dict = {}
    for item in args.runs:
        name, sep, path = item.partition("=")
        if not sep:
            print(f"--runs entries must look like NAME=PATH, got {item!r}", file=sys.stderr)
            return 2
        runs[name] = load_run(path)
        run_paths[name] = path
    if args.compare and not all(n in runs for n in args.compare):
        print("--compare names must match --runs names", file=sys.stderr)
        return 2

    if args.validate_only or not runs:
        lines = [f"golden sha256:{sha[:12]}  {len(queries)} queries",
                 "types: " + ", ".join(f"{t}={sum(1 for q in queries if q.type == t)}" for t in ALLOWED_TYPES),
                 "splits: " + ", ".join(f"{s}={sum(1 for q in queries if q.split == s)}" for s in ALLOWED_SPLITS)]
        lines += [f"WARNING: {w}" for w in warnings]
        if transcripts is not None:
            vals = [lexical_overlap(q, transcripts) for q in queries]
            vals = [v for v in vals if v is not None]
            if vals:
                lines.append(f"mean lexical overlap with answer spans: {mean(vals):.2f} "
                             f"(share <0.5: {sum(1 for v in vals if v < 0.5) / len(vals):.0%})")
        print("\n".join(lines))
        return 0

    for name, run in runs.items():
        missing = [q.id for q in selected if q.id not in run]
        if missing:
            print(f"WARNING: run '{name}' has no results for {len(missing)} queries "
                  "(counted as misses)", file=sys.stderr)

    notes = [label_note] if label_note else []
    reg_kind = None
    if args.split in ("test", "all"):
        if args.unregistered:
            notes.append("**UNREGISTERED TEST RUN: no pre-registration lock was used. "
                         "Treat these numbers as exploratory, not as the test result.**")
        elif not args.prereg:
            print("error: the test split needs a pre-registration lock (--prereg prereg.lock.json; see "
                  "preregister.py), or --unregistered to peek, which stamps the report as exploratory.",
                  file=sys.stderr)
            return 2
        else:
            import preregister as pr
            try:
                lock = pr.load_lock(args.prereg)
            except pr.PreregError as e:
                print("error:", *e.problems, sep="\n  - ", file=sys.stderr)
                return 2
            eval_cfg = {"split": args.split, "verified_only": args.verified_only, "hit_def": args.hit_def,
                        "k": ks, "primary_k": args.primary_k, "dedup_overlap": args.dedup_overlap,
                        "lead_tol": args.lead_tol, "seek_max_frac": args.seek_max_frac,
                        "overlap_frac": args.overlap_frac, "overlap_cap": args.overlap_cap}
            problems = pr.check_lock(lock, args.golden, eval_cfg, runs)
            prior = pr.prior_runs(args.prereg)
            if prior and not args.allow_rerun:
                problems.append(f"a test run was already recorded on {prior[0]['utc']}; the test split is "
                                "single-use (--allow-rerun stamps the report as a re-run)")
            if problems:
                print("This run does not match the pre-registration lock; nothing was scored:",
                      *problems, sep="\n  - ", file=sys.stderr)
                return 2
            reg_kind = "test_rerun" if prior else "test_run"
            notes += pr.banner(lock, prior, reg_kind)

    cfgs: dict = {}
    for d in (["seek", "overlap"] if args.hit_def == "both" else [args.hit_def]):
        cfgs[d] = HitConfig(d, args.lead_tol, args.seek_max_frac, args.overlap_frac, args.overlap_cap)
    report, js = run_report(selected, sha, runs, cfgs, ks, args.primary_k, args.dedup_overlap,
                            tuple(args.compare) if args.compare else None, transcripts,
                            args.boot, args.seed, args.split, warnings, notes)
    if reg_kind:
        try:   # record BEFORE showing anything: a test run that was seen but not logged would defeat the lock
            pr.record_run(args.prereg, reg_kind, report, run_paths)
        except OSError as e:
            print(f"error: could not write the ledger ({e}); the report was not shown", file=sys.stderr)
            return 2
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(js, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
