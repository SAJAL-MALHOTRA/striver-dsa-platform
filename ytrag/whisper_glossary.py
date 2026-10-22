"""
Whisper DSA Custom Prompt & Glossary Module.

Provides domain-specific prompt prefixes and glossary tokens for Whisper
transcription to dramatically reduce Hinglish phonetic collisions at the
source (transcription time) rather than patching downstream.

The Whisper `initial_prompt` parameter biases the decoder's vocabulary
toward DSA terminology, preventing corruptions like:
  "can dance" -> "Kadane's"
  "dry wine"  -> "dry run"
  "nums of eye" -> "nums[i]"

Usage:
  import whisper
  model = whisper.load_model("medium")
  result = model.transcribe(
      audio_path,
      initial_prompt=get_whisper_dsa_prompt(),
      language="en",
  )
"""

from __future__ import annotations

# Core DSA vocabulary that Whisper frequently mis-transcribes in Hinglish lectures
DSA_GLOSSARY_TOKENS = [
    # Data Structures
    "array", "linked list", "stack", "queue", "deque", "priority queue",
    "binary tree", "binary search tree", "BST", "AVL tree", "trie",
    "heap", "min-heap", "max-heap", "hash map", "hash set", "hash table",
    "adjacency list", "adjacency matrix", "graph", "directed graph",
    "segment tree", "Fenwick tree", "disjoint set", "union-find",

    # Algorithms & Techniques
    "Kadane's algorithm", "Kadane", "Floyd's cycle detection",
    "tortoise and hare", "two pointers", "sliding window",
    "binary search", "merge sort", "quick sort", "insertion sort",
    "bubble sort", "counting sort", "radix sort", "topological sort",
    "Dijkstra", "Bellman-Ford", "Kruskal", "Prim's", "BFS", "DFS",
    "breadth first search", "depth first search",
    "dynamic programming", "DP", "memoization", "tabulation",
    "greedy", "backtracking", "recursion", "divide and conquer",
    "bit manipulation", "XOR", "AND", "OR",

    # Array/Index Syntax
    "nums[i]", "nums[j]", "nums[left]", "nums[right]", "nums[mid]",
    "arr[i]", "arr[j]", "arr[mid]", "arr[low]", "arr[high]",
    "left pointer", "right pointer", "low pointer", "high pointer",
    "mid pointer", "slow pointer", "fast pointer",

    # Common DSA Terms
    "subarray", "subsequence", "substring", "subset",
    "in-place", "swap", "partition", "pivot",
    "dry run", "trace", "edge case", "corner case",
    "time complexity", "space complexity", "auxiliary space",
    "O(N)", "O(1)", "O(N^2)", "O(log N)", "O(N log N)",
    "overflow", "underflow", "integer overflow",
    "null pointer", "nullptr", "segmentation fault",
    "base case", "recursive call", "call stack",
    "visited array", "frequency map", "prefix sum", "suffix sum",
    "monotonic stack", "next greater element",

    # Striver-specific vocabulary
    "Striver", "take U forward", "A2Z DSA sheet",
    "brute force", "better approach", "optimal approach",
    "intuition", "observation",
]

# Hinglish transition phrases Whisper should recognize cleanly
HINGLISH_CUE_PHRASES = [
    "let's understand", "let's see", "let's code",
    "so basically", "so the idea is", "now moving on",
    "agar ye condition", "toh hum", "pehle hum",
    "isko hum", "yahan pe", "samajhte hain",
    "time complexity kya hogi", "space complexity kya hogi",
    "dry run karte hain", "edge case dekho",
]


def get_whisper_dsa_prompt(include_hinglish: bool = True) -> str:
    """
    Generates a Whisper initial_prompt string biased toward DSA vocabulary.

    This prompt is prepended to the Whisper decoder context, causing the model
    to prefer DSA terms over phonetically similar everyday words.

    Parameters
    ----------
    include_hinglish : bool
        Whether to include Hinglish transition cue phrases (default True).

    Returns
    -------
    str
        A prompt string suitable for whisper.transcribe(initial_prompt=...).
    """
    tokens = list(DSA_GLOSSARY_TOKENS)
    if include_hinglish:
        tokens.extend(HINGLISH_CUE_PHRASES)

    # Whisper's initial_prompt is limited to ~224 tokens, so we construct
    # a dense, natural-sounding prompt that packs maximum vocabulary bias
    prompt = (
        "This is a Data Structures and Algorithms lecture by Striver from Take U Forward. "
        "Topics include: " + ", ".join(tokens[:60]) + ". "
        "The lecturer uses Hinglish (Hindi-English mix) and discusses "
        + ", ".join(tokens[60:90]) + ". "
        "Key terms: " + ", ".join(tokens[90:]) + "."
    )
    return prompt


def get_whisper_glossary_dict() -> dict[str, str]:
    """
    Returns a dictionary mapping common Whisper mis-transcriptions to correct DSA terms.

    Can be used as a post-transcription correction pass in addition to the
    phonetic_normalizer module.
    """
    return {
        "can dance": "Kadane's",
        "canadians": "Kadane's",
        "kadan": "Kadane",
        "dry wine": "dry run",
        "try run": "dry run",
        "dry done": "dry run",
        "numbers of eye": "nums[i]",
        "nums of eye": "nums[i]",
        "sab array": "subarray",
        "sub array sham": "subarray",
        "to pointer": "two pointers",
        "too pointer": "two pointers",
        "deep table": "DP table",
        "db table": "DP table",
        "adj list": "adjacency list",
        "breath first search": "breadth first search",
        "deep first search": "depth first search",
        "beat manipulation": "bit manipulation",
        "merge short": "merge sort",
        "quick short": "quick sort",
        "in please": "in-place",
        "in peace": "in-place",
        "mark sort": "merge sort",
        "x or": "XOR",
        "x o r": "XOR",
    }
