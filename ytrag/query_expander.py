"""
Query Expansion and Pedagogical Speech Bridge (HyDE for DSA).

Bridges the vocabulary gap between code-syntax student queries:
    "nums[left] == nums[left-1]"
and Hinglish pedagogical lecture speech:
    "agar pichla element same tha toh duplicate skip kar denge"
"""

from __future__ import annotations

import re
from typing import Optional
from .phonetic_normalizer import normalize_dsa_text

# Code pattern to pedagogical lecture speech expansions
SYNTAX_SPEECH_MAP: list[tuple[re.Pattern, list[str]]] = [
    # Duplicate skipping
    (
        re.compile(r"(nums|arr)\[(?:left|i)\]\s*==\s*(nums|arr)\[(?:left|i)\s*-\s*1\]", re.I),
        ["pichla element same", "duplicate triplet", "skip duplicate element", "continue next iteration"]
    ),
    (
        re.compile(r"(nums|arr)\[(?:right|j)\]\s*==\s*(nums|arr)\[(?:right|j)\s*\+\s*1\]", re.I),
        ["agla element same", "duplicate skip", "right decrement pointer"]
    ),

    # Binary search boundary
    (
        re.compile(r"low\s*<=\s*high|low\s*<\s*high", re.I),
        ["binary search pointer cross", "termination condition", "search space divide", "equal pointer condition"]
    ),
    (
        re.compile(r"mid\s*=\s*(?:low\s*\+\s*high)\s*/\s*2|low\s*\+\s*\(high\s*-\s*low\)\s*/\s*2", re.I),
        ["middle element index", "overflow safe mid", "half partition"]
    ),

    # Two pointers / Cycle detection
    (
        re.compile(r"fast\s*->\s*next|fast\s*=\s*fast\s*->\s*next\s*->\s*next", re.I),
        ["tortoise and hare", "fast moves two steps", "cycle loop detection", "distance reduces by one", "relative speed"]
    ),
    (
        re.compile(r"slow\s*=\s*slow\s*->\s*next", re.I),
        ["slow pointer one step", "floyd cycle detection"]
    ),

    # Kadane subarray reset
    (
        re.compile(r"sum\s*<\s*0|current_sum\s*<\s*0", re.I),
        ["negative sum reset zero", "positive prefix", "discard negative subarray", "kadane maximum sum"]
    ),

    # Recursion base case
    (
        re.compile(r"if\s*\((?:n|i)\s*==\s*0|if\s*\((?:n|i)\s*<=\s*0", re.I),
        ["recursion base case", "stack unwind", "recursion tree termination"]
    ),

    # Bit manipulation
    (
        re.compile(r"n\s*&\s*\(n\s*-\s*1\)", re.I),
        ["turn off rightmost set bit", "power of two check", "count set bits"]
    ),
    (
        re.compile(r"\^\s*=", re.I),
        ["xor cancels duplicate", "single appearing element", "a xor a is zero"]
    ),

    # Sliding Window & Two Pointers
    (
        re.compile(r"right\s*-\s*left\s*\+\s*1|window_sum\s*>\s*k", re.I),
        ["sliding window length", "shrink window from left", "expand right pointer", "valid window condition"]
    ),

    # Linked List Reversal
    (
        re.compile(r"curr\s*->\s*next\s*=\s*prev|curr\.next\s*=\s*prev", re.I),
        ["reverse linked list", "break link point backward", "three pointers prev curr next", "head connection reverse"]
    ),

    # Monotonic Stack (Next Greater Element)
    (
        re.compile(r"st\.top\(\)\s*<=\s*(?:nums|arr)|st\.pop\(\)", re.I),
        ["monotonic stack pop smaller elements", "next greater element", "strictly increasing stack", "nearest smaller element"]
    ),

    # Merge Intervals
    (
        re.compile(r"intervals\[i\]\[0\]\s*<=\s*last|max\(last\[1\]", re.I),
        ["overlapping intervals merge", "sort by start time", "interval intersection", "extend end boundary"]
    ),

    # Euclidean Algorithm / GCD
    (
        re.compile(r"gcd\(b,\s*a\s*%\s*b\)|while\s*\(\s*a\s*>\s*0\s*&&\s*b\s*>\s*0\s*\)", re.I),
        ["euclidean algorithm greatest common divisor", "modulo remainder division", "gcd time complexity log n"]
    ),

    # DP Pick / Not-Pick
    (
        re.compile(r"take\s*=\s*.*not_take|dp\[i\]\[w\]\s*=\s*max", re.I),
        ["pick not pick recurrence", "include or exclude element", "subproblem memoization", "dp state transition table"]
    ),

    # Graph Traversal (BFS / DFS / Visited)
    (
        re.compile(r"vis\[.*\]\s*=\s*1|visited\[.*\]\s*=\s*True", re.I),
        ["visited array mark unvisited", "level order traversal queue", "cycle in undirected graph", "connected components bfs"]
    ),
]


def expand_dsa_query(query: str) -> str:
    """
    Expands a student code/technical query into an enriched hybrid query string
    incorporating both technical identifiers and pedagogical lecture phrases.
    """
    if not query.strip():
        return ""

    # 1. Normalize phonetics first
    normalized = normalize_dsa_text(query)
    expanded_terms: list[str] = [normalized]

    # 2. Check for code syntax to spoken-speech mappings
    for pattern, speech_keywords in SYNTAX_SPEECH_MAP:
        if pattern.search(query) or pattern.search(normalized):
            expanded_terms.extend(speech_keywords)

    return " ".join(expanded_terms)
