"""
Temporal & semantic chunking logic for YouTube video transcripts.

Takes a list of timed segments and produces overlapping time-window Chunks.
Preserves second-level start and end timestamps so retrieval results can
directly jump to the exact moment in the video.

Includes:
- Semantic boundary snapping: snaps chunk boundaries to speaker pauses (silence >= 1.2s),
  topic transition cues ("now let's code this", "optimal approach", "edge case"), and
  sentence ends rather than cutting explanations mid-sentence.
- Whisper repetition loop filtering: drops hallucination loops that poison BM25/dense search.
- Contextual title prefixing: provides global problem context for dense embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional
from .models import Chunk, Segment

_SENTENCE_SPLIT = re.compile(r"[.!?\n]+")

_TOPIC_TRANSITION_PATTERN = re.compile(
    r"\b(now let'?s code|let'?s write the code|moving to the code|code part|"
    r"next approach|better approach|optimal approach|brute force approach|"
    r"time complexity|space complexity|edge case|corner case|dry run|"
    r"problem statement|let'?s understand the question)\b",
    re.I,
)


def is_repetitive(text: str, threshold: float = 0.6) -> bool:
    """True if one sentence makes up most of the chunk.

    This detects Whisper's loop hallucination surviving into a chunk. These are
    retrieval poison: they match everything and say nothing.
    """
    sentences = [s.strip().lower() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) < 3:
        return False
    most_common = Counter(sentences).most_common(1)[0][1]
    return most_common / len(sentences) >= threshold


def find_semantic_boundary(
    segments: list[Segment],
    target_sec: float,
    tolerance_sec: float = 12.0,
) -> Optional[float]:
    """
    Finds a natural semantic boundary (speaker pause >= 1.2s, topic transition cue,
    or sentence completion) near target_sec within [target - tolerance, target + tolerance].
    """
    best_boundary = None
    best_priority = -1  # 3: topic cue, 2: long pause (>=1.2s), 1: sentence end with pause (>=0.4s)

    for i in range(len(segments) - 1):
        s_curr = segments[i]
        s_next = segments[i + 1]

        boundary = s_curr.end
        dist = abs(boundary - target_sec)
        if dist > tolerance_sec:
            continue

        gap = max(0.0, s_next.start - s_curr.end)

        # Priority 3: Topic transition cue in upcoming segment
        if _TOPIC_TRANSITION_PATTERN.search(s_next.text):
            priority = 3
        # Priority 2: Natural speaker pause / silence >= 1.2s
        elif gap >= 1.2:
            priority = 2
        # Priority 1: Sentence closure punctuation + moderate pause >= 0.4s
        elif s_curr.text.strip().endswith((".", "!", "?")) and gap >= 0.4:
            priority = 1
        else:
            continue

        if priority > best_priority or (priority == best_priority and best_boundary is not None and dist < abs(best_boundary - target_sec)):
            best_priority = priority
            best_boundary = boundary

    return best_boundary


def chunk_segments(
    video_id: str,
    video_title: str,
    segments: list[Segment],
    window_sec: int = 75,
    overlap_sec: int = 15,
    prefix_title: bool = True,
    use_semantic_boundaries: bool = True,
) -> list[Chunk]:
    """
    Groups timed transcript segments into sliding temporal windows with semantic pause-snapping.

    Parameters
    ----------
    video_id : str
        YouTube video ID.
    video_title : str
        Human-readable video title.
    segments : list[Segment]
        Sorted list of transcript segments with start and end times in seconds.
    window_sec : int
        Target duration of each chunk in seconds (default 75s).
    overlap_sec : int
        Overlap between consecutive windows in seconds (default 15s).
    prefix_title : bool
        Whether to prefix video title for conceptual retrieval context (default True).
    use_semantic_boundaries : bool
        Whether to snap boundaries to speaker pauses and topic transition cues (default True).

    Returns
    -------
    list[Chunk]
        Chunk objects with accurate start_sec and end_sec timestamps.
    """
    if not segments:
        return []

    chunks: list[Chunk] = []
    step_sec = max(1, window_sec - overlap_sec)

    # Total video duration covered by segments
    max_end_time = max(s.end for s in segments)

    current_window_start = int(math.floor(segments[0].start))

    while current_window_start < max_end_time:
        target_window_end = current_window_start + window_sec
        current_window_end = target_window_end

        # Optionally snap window end to natural semantic boundary if available
        if use_semantic_boundaries and len(segments) > 2:
            tolerance = min(15.0, max(6.0, window_sec * 0.18))
            boundary = find_semantic_boundary(segments, float(target_window_end), tolerance_sec=tolerance)
            if boundary is not None and boundary > current_window_start + overlap_sec:
                current_window_end = int(round(boundary))

        # Collect all segments that overlap with this window [start, end]
        window_texts = []
        actual_start = None
        actual_end = None

        for seg in segments:
            # Check overlap: seg.start < window_end and seg.end > window_start
            if seg.start < current_window_end and seg.end > current_window_start:
                clean_text = seg.text.strip()
                if clean_text:
                    window_texts.append(clean_text)
                    if actual_start is None:
                        actual_start = int(math.floor(seg.start))
                    actual_end = int(math.ceil(seg.end))

        if window_texts:
            combined_text = " ".join(window_texts).strip()
            # Drop Whisper loop hallucinations
            if combined_text and not is_repetitive(combined_text):
                s_sec = actual_start if actual_start is not None else current_window_start
                e_sec = actual_end if actual_end is not None else current_window_end

                chunk_id = f"{video_id}:{s_sec}"
                chunk_body = f"{video_title}\n\n{combined_text}" if (prefix_title and video_title) else combined_text
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        video_id=video_id,
                        video_title=video_title,
                        start_sec=s_sec,
                        end_sec=e_sec,
                        text=chunk_body,
                    )
                )

        current_window_start += step_sec

    # Deduplicate consecutive identical or near-identical chunks if any
    unique_chunks: list[Chunk] = []
    seen_ids = set()
    for c in chunks:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            unique_chunks.append(c)

    return unique_chunks
