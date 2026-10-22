"""
Hinglish-to-DSA Phonetic & Slang Normalizer.

Solves the 'Hinglish Phonetic Collision' trap where speech-to-text (Whisper/YouTube)
corrupts algorithmic terminology during transcription, causing BM25 and vector drift.

Examples:
- "dry wine" / "try run" -> "dry run"
- "can dance" / "canadians" -> "kadane's algorithm"
- "nums of eye" / "numbers of i" -> "nums[i]"
- "sab array" / "sub array sham" -> "subarray"
- "deep table" / "db table" -> "dp table"
- "adj list" / "adjacent list" -> "adjacency list"
"""

from __future__ import annotations

import re

# Canonical mapping of phonetic speech corruptions to standard DSA terminology
PHONETIC_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # Kadane's Algorithm
    (re.compile(r"\b(can\s+dance|canadians?|kadan(?:e)?|keda(?:n|ne))\b", re.I), "kadane"),
    (re.compile(r"\bkadane(?:['’]s)?\s+algo(?:rithm)?\b", re.I), "kadane algorithm"),

    # Dry run
    (re.compile(r"\b(dry\s+wine|try\s+run|dry\s+done|try\s+done)\b", re.I), "dry run"),

    # Subarray / Subsequence
    (re.compile(r"\b(sab\s+array|sub\s+array\s+sham|sub\s+baray|sub\s+arre)\b", re.I), "subarray"),
    (re.compile(r"\b(sub\s+sequence|sab\s+sequence)\b", re.I), "subsequence"),

    # Two pointer
    (re.compile(r"\b(to\s+pointer|too\s+pointer|two\s+point(?:ers?)?)\b", re.I), "two pointers"),
    (re.compile(r"\b(slow\s+pointer\s+and\s+fast\s+pointer|tortoise\s+hare)\b", re.I), "tortoise and hare"),

    # Dynamic Programming
    (re.compile(r"\b(deep\s+table|db\s+table|deep\s+p\s+table)\b", re.I), "dp table"),
    (re.compile(r"\b(memo\s+ization|memorization|memoiz)\b", re.I), "memoization"),
    (re.compile(r"\b(tab\s+ulation|table\s+ation)\b", re.I), "tabulation"),

    # Array indexing syntax spoken in Hindi/English
    (re.compile(r"\b(numbers?\s+of\s+(?:eye|i)|nums?\s+of\s+(?:eye|i)|num\s+i)\b", re.I), "nums[i]"),
    (re.compile(r"\b(numbers?\s+of\s+(?:jay|j)|nums?\s+of\s+(?:jay|j)|num\s+j)\b", re.I), "nums[j]"),
    (re.compile(r"\b(arr\s+of\s+i|array\s+of\s+i|array\s+of\s+eye)\b", re.I), "arr[i]"),
    (re.compile(r"\b(arr\s+of\s+mid|array\s+of\s+mid)\b", re.I), "arr[mid]"),

    # Graphs & Trees
    (re.compile(r"\b(adj\s+list|adjacent\s+list|edge\s+list\s+matrix)\b", re.I), "adjacency list"),
    (re.compile(r"\b(breath\s+first\s+search|bf\s+s)\b", re.I), "breadth first search"),
    (re.compile(r"\b(deep\s+first\s+search|df\s+s)\b", re.I), "depth first search"),

    # Bit manipulation
    (re.compile(r"\b(beat\s+manipulation|beats\s+manipulation)\b", re.I), "bit manipulation"),
    (re.compile(r"\b(xor\s+operator|x\s+or\s+operator|x\s+o\s+r)\b", re.I), "xor"),

    # Sorting & Searching
    (re.compile(r"\b(binary\s+search\s+tree|b\s+s\s+t)\b", re.I), "bst"),
    (re.compile(r"\b(merge\s+short|mark\s+sort)\b", re.I), "merge sort"),
    (re.compile(r"\b(quick\s+short|kwick\s+sort)\b", re.I), "quick sort"),

    # In-place
    (re.compile(r"\b(in\s+please|in\s+peace|in\s+phase)\b", re.I), "in-place"),

    # Time & Space Complexities
    (re.compile(r"\b(order\s+of\s+n|o\s+of\s+n|big\s+o\s+of\s+n)\b", re.I), "O(N)"),
    (re.compile(r"\b(order\s+of\s+one|o\s+of\s+one|big\s+o\s+of\s+one)\b", re.I), "O(1)"),
    (re.compile(r"\b(order\s+of\s+n\s+square|o\s+n\s+square|o\s+n\s+two)\b", re.I), "O(N^2)"),
    (re.compile(r"\b(order\s+of\s+log\s+n|o\s+log\s+n)\b", re.I), "O(log N)"),
]


def normalize_dsa_text(text: str) -> str:
    """
    Applies phonetic corrections to raw transcript segments or user queries.
    Preserves original casing structure while correcting corrupted algorithmic terms.
    """
    if not text:
        return ""

    cleaned = text
    for pattern, replacement in PHONETIC_REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    return cleaned
