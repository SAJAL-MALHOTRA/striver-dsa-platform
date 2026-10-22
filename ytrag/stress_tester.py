"""
Edge-Case Stress-Testing Suite & Technical Interview "Grill Me" Engine for DSA.

Provides:
- Canonical corner-case suites for competitive programming & interview traps:
  - All negative arrays (Kadane 0 vs INT_MIN trap)
  - All duplicate elements (3Sum / QuickSort partition trap)
  - Single-element & empty boundaries
  - Large values (integer overflow trap)
  - Odd/Even length linked list null pointer traps
  - Binary search underflow/overflow bounds
- "Grill Me" interview simulator:
  - Generates sharp, targeted conceptual & edge-case interview questions based on student code
  - Evaluates student answers with scoring (1-10), verdict, and FAANG gold-standard answers
  - Seamless offline fallback + live LLM support (Groq / Grok / OpenAI)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Optional

from ytrag.llm_provider import get_provider_details

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    name: str
    description: str
    sample_input: str
    trap_warning: str


# Canonical edge case suites for primary DSA problem categories
CANONICAL_EDGE_CASES: dict[str, list[TestCase]] = {
    "array": [
        TestCase(
            name="All Negative Numbers",
            description="Array containing exclusively negative integers.",
            sample_input="[-2, -3, -4, -1, -2, -1, -5, -3]",
            trap_warning="Kadane trap: if max_sum is initialized to 0 instead of INT_MIN / arr[0], it will return 0 instead of -1.",
        ),
        TestCase(
            name="All Identical Duplicates",
            description="Array where every element has the exact same value.",
            sample_input="[2, 2, 2, 2, 2]",
            trap_warning="Two-pointer / 3Sum trap: failing to advance pointers on duplicates causes infinite loops or duplicate triplets.",
        ),
        TestCase(
            name="Single Element Boundary",
            description="Array with exactly 1 element.",
            sample_input="[42]",
            trap_warning="Off-by-one trap: loop condition `i < n - 1` or two-pointer `left < right` might fail immediately.",
        ),
        TestCase(
            name="Already Sorted vs Reverse Sorted",
            description="Strictly descending order array.",
            sample_input="[5, 4, 3, 2, 1]",
            trap_warning="QuickSort pivot trap: worst-case O(N^2) stack depth if picking first/last element as pivot.",
        ),
        TestCase(
            name="Large Boundary Values (Overflow)",
            description="Integers nearing 32-bit limits.",
            sample_input="[1000000000, 1000000000, 1000000000]",
            trap_warning="Overflow trap: summing 32-bit signed integers causes negative overflow unless using long long / int64.",
        ),
    ],
    "linked_list": [
        TestCase(
            name="Single Node List",
            description="List with head -> null.",
            sample_input="[1]",
            trap_warning="Fast pointer null dereference trap: `fast->next->next` segfaults if `fast->next` is null.",
        ),
        TestCase(
            name="Two Node List (Cycle vs No Cycle)",
            description="List with 2 nodes.",
            sample_input="[1, 2]",
            trap_warning="Fast pointer overshoot trap: odd vs even length traversal boundary.",
        ),
        TestCase(
            name="Empty List (Null Head)",
            description="List pointer is nullptr.",
            sample_input="head = null",
            trap_warning="Null pointer exception on immediate `head->val` access.",
        ),
    ],
    "binary_search": [
        TestCase(
            name="Target Not Present (Smaller than all)",
            description="Search target is less than arr[0].",
            sample_input="arr = [2, 4, 6, 8], target = 1",
            trap_warning="Index underflow trap: high pointer falls below 0.",
        ),
        TestCase(
            name="Target Not Present (Greater than all)",
            description="Search target is greater than arr[n-1].",
            sample_input="arr = [2, 4, 6, 8], target = 10",
            trap_warning="Index out of bounds trap: low pointer exceeds n - 1.",
        ),
        TestCase(
            name="Two-Element Array Boundary",
            description="Array with 2 elements: arr = [1, 3], target = 3.",
            sample_input="arr = [1, 3], target = 3",
            trap_warning="Mid calculation rounding: if high = mid without +1, can cause infinite loop.",
        ),
    ],
    "string": [
        TestCase(
            name="All Identical Characters",
            description="String where every character is the same.",
            sample_input='"aaaaaa"',
            trap_warning="Sliding window trap: window size expansion and shrink condition boundary check.",
        ),
        TestCase(
            name="Non-Alphanumeric & Mixed Case",
            description="String with spaces, punctuation, and mixed case.",
            sample_input='"A man, a plan, a canal: Panama"',
            trap_warning="Two-pointer palindrome trap: skipping non-alphanumeric chars without boundary checks.",
        ),
    ],
    "tree_graph": [
        TestCase(
            name="Skewed Single-Branch Tree",
            description="Tree where every node has only a right child.",
            sample_input="[1, null, 2, null, 3, null, 4]",
            trap_warning="Recursion depth limit: O(N) recursion stack degenerates into linked list causing stack overflow.",
        ),
        TestCase(
            name="Disconnected Graph Components",
            description="Graph with isolated components.",
            sample_input="nodes = 5, edges = [[0, 1], [2, 3]]",
            trap_warning="Component traversal trap: running BFS/DFS from vertex 0 misses isolated nodes unless outer loop iterates 0 to V-1.",
        ),
    ],
}


def get_edge_cases_for_problem(topic: str, title: str) -> list[dict[str, Any]]:
    """
    Returns relevant edge-case test definitions based on problem topic and title.
    """
    t_lower = topic.lower()
    title_lower = title.lower()

    cases: list[TestCase] = []
    if "linked" in t_lower or "list" in title_lower:
        cases.extend(CANONICAL_EDGE_CASES["linked_list"])
    elif "binary search" in t_lower or "search" in title_lower:
        cases.extend(CANONICAL_EDGE_CASES["binary_search"])
    elif "string" in t_lower:
        cases.extend(CANONICAL_EDGE_CASES["string"])
    elif "tree" in t_lower or "graph" in t_lower:
        cases.extend(CANONICAL_EDGE_CASES["tree_graph"])
    else:
        cases.extend(CANONICAL_EDGE_CASES["array"])

    return [
        {
            "name": c.name,
            "description": c.description,
            "sample_input": c.sample_input,
            "trap_warning": c.trap_warning,
        }
        for c in cases
    ]


# Canonical Grill Me Question Bank (Deterministic High-Yield FAANG Interview Drills)
CANONICAL_GRILL_BANKS = {
    "triplet_duplicates": {
        "pattern": r"(3sum|triplet|two sum|2 sum|duplicate)",
        "question": "In your two-pointer or hashing implementation, how do you prevent generating duplicate answers when the input array contains repeated elements (e.g., `[-1, -1, 0, 1, 1]`)? Walk me through your exact pointer movement.",
        "trap_warning": "Failing to skip duplicates on `nums[left] == nums[left-1]` or fixated `nums[i] == nums[i-1]` leads to duplicate tuples in the output or O(N^3) set allocation overhead.",
        "gold_standard": "Sort the array first. Fix element i and skip if nums[i] == nums[i-1]. When the target sum matches, increment left and decrement right while continuously skipping consecutive identical elements: while (left < right && nums[left] == nums[left-1]) left++.",
        "keywords": ["skip", "duplicate", "left++", "right--", "sort", "while", "pointer", "identical"],
    },
    "all_negative_kadane": {
        "pattern": r"(kadane|max.*subarray|maximum subarray|negative)",
        "question": "What happens if every single element in the array is negative (e.g. `[-4, -2, -7, -1]`)? Does your algorithm return `-1` or `0`? Explain how your initialization and state resets handle the non-empty subarray constraint.",
        "trap_warning": "Initializing `max_sum = 0` incorrectly returns 0 when all elements are negative, violating the interview requirement that a non-empty subarray must be chosen.",
        "gold_standard": "Initialize max_sum = nums[0] (or INT_MIN) and current_sum = nums[0]. Iterate from index 1. Update current_sum = max(nums[i], current_sum + nums[i]) and max_sum = max(max_sum, current_sum). Never reset current_sum to 0 before updating max_sum.",
        "keywords": ["nums[0]", "int_min", "negative", "0", "reset", "-1", "non-empty", "max"],
    },
    "fast_slow_pointers": {
        "pattern": r"(linked.*list|cycle|middle.*node|pointer|slow|fast)",
        "question": "In Floyd's Tortoise and Hare algorithm, why must the while loop condition check `fast != nullptr && fast->next != nullptr`? What exact failure occurs if you only check `fast->next != nullptr` on an empty or even-length list?",
        "trap_warning": "On even-length lists or empty lists, `fast` becomes nullptr. Accessing `fast->next` when `fast` is nullptr triggers an immediate Segmentation Fault (SIGSEGV / NullPointerException).",
        "gold_standard": "Because fast advances two nodes at a time (fast = fast->next->next), fast will reach nullptr on even-length lists. Checking `fast != nullptr` first acts as a short-circuit guard before evaluating `fast->next != nullptr`.",
        "keywords": ["nullptr", "null", "segfault", "even", "guard", "short-circuit", "two steps", "fast->next"],
    },
    "binary_search_overflow": {
        "pattern": r"(binary search|lower bound|upper bound|search|overflow|mid)",
        "question": "When computing the midpoint, why is `mid = low + (high - low) / 2` strongly preferred over `(low + high) / 2` in C++ and Java? Additionally, why must the search space update use `mid + 1` or `mid - 1` rather than `mid`?",
        "trap_warning": "`low + high` can exceed 2^31 - 1 resulting in integer overflow (negative index). Updating boundaries with `low = mid` without +1 causes an infinite loop when high - low == 1.",
        "gold_standard": "`low + (high - low) / 2` mathematically prevents 32-bit signed integer overflow. Updating bounds with `mid + 1` and `mid - 1` guarantees the search space strictly shrinks by at least one element each iteration, preventing infinite loops.",
        "keywords": ["overflow", "integer", "infinite", "shrink", "+ 1", "- 1", "32-bit", "mid"],
    },
    "general_complexity": {
        "pattern": r".*",
        "question": "Walk me through the exact time and space complexity of your solution. Could this problem be solved with O(1) auxiliary space without degrading time complexity?",
        "trap_warning": "Candidates often confuse auxiliary space (extra memory used) with output/input space, or use hash maps when in-place two pointers could achieve O(1) extra space.",
        "gold_standard": "State best, average, and worst-case time complexity with Big-O notation. Differentiate between input space and auxiliary memory. If using O(N) space, consider whether sorting or in-place pointer indexing enables O(1) auxiliary space.",
        "keywords": ["o(1)", "o(n)", "space", "time", "auxiliary", "complexity", "in-place", "hash"],
    },
}


def find_canonical_grill(title: str, topic: str = "", extra: str = "") -> dict[str, Any]:
    """Finds the best matching canonical question bank entry."""
    combined = f"{title} {topic} {extra}".lower()
    for key, item in CANONICAL_GRILL_BANKS.items():
        if key == "general_complexity":
            continue
        if re.search(item["pattern"], combined, re.I):
            return item
    return CANONICAL_GRILL_BANKS["general_complexity"]


def generate_grill_question(
    title: str,
    topic: str = "",
    code: str = "",
    provider: str = "groq",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    timeout: float = 8.0,
) -> dict[str, Any]:
    """
    Generates a technical interview question for 'Grill Me' mode.
    Uses LLM if an API key is available; falls back to canonical question bank.
    """
    p_clean = (provider or "").lower().strip()
    prompt = (
        f"You are a strict, top-tier FAANG Technical Interviewer.\n"
        f"The candidate is solving '{title}' ({topic}).\n"
        f"Here is their implementation:\n```\n{code[:1500]}\n```\n\n"
        f"Task:\n"
        f"Ask ONE sharp, high-yield conceptual or edge-case interview question that tests whether they understand "
        f"critical edge cases (e.g. all negatives, duplicate collisions, integer overflow, odd/even boundaries) or "
        f"time/space complexity trade-offs.\n"
        f"Respond ONLY with a JSON object with this format:\n"
        f'{{"question": "your question here", "trap_warning": "the specific trap or edge case being tested", "gold_standard": "concise model answer"}}'
    )

    if p_clean in ("local", "striver-rtx5050", "striver"):
        try:
            from ytrag.local_llm import LocalTeachingAssistant
            assistant = LocalTeachingAssistant.get_instance()
            messages = [
                {"role": "system", "content": "You are an elite DSA interviewer. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
            raw_text, _, _ = assistant.generate_chat_completion(
                messages=messages,
                max_tokens=400,
                temperature=0.2,
            )
            cleaned = raw_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            if "question" in parsed:
                return {
                    "question": parsed.get("question", ""),
                    "trap_warning": parsed.get("trap_warning", ""),
                    "gold_standard": parsed.get("gold_standard", ""),
                    "source": "local",
                }
        except Exception as e:
            logger.warning("Local grill question generation failed: %s", e)

    api_key = api_key.strip()
    if api_key:
        try:
            resolved_base_url, resolved_model = get_provider_details(provider, base_url, model)
            payload = {
                "model": resolved_model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are an elite DSA interviewer. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            req = urllib.request.Request(
                f"{resolved_base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(data["choices"][0]["message"]["content"])
            return {
                "question": parsed.get("question", ""),
                "trap_warning": parsed.get("trap_warning", ""),
                "gold_standard": parsed.get("gold_standard", ""),
                "source": provider,
            }
        except Exception:
            pass  # Fall back to canonical

    # Deterministic Canonical Fallback
    bank = find_canonical_grill(title, topic)
    return {
        "question": bank["question"],
        "trap_warning": bank["trap_warning"],
        "gold_standard": bank["gold_standard"],
        "source": "canonical",
    }


def evaluate_grill_answer(
    question: str,
    trap_warning: str,
    user_answer: str,
    gold_standard: str = "",
    code: str = "",
    title: str = "",
    topic: str = "",
    provider: str = "groq",
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    timeout: float = 10.0,
) -> dict[str, Any]:
    """
    Evaluates the candidate's answer in 'Grill Me' mode.
    Scores 1-10 with detailed feedback and gold-standard comparison.
    """
    user_ans_clean = user_answer.strip()
    if not user_ans_clean:
        return {
            "score": 0,
            "verdict": "No Answer Provided",
            "feedback": "Please type your technical reasoning to receive interview evaluation.",
            "gold_standard": gold_standard or trap_warning,
            "source": "offline",
        }

    p_clean = (provider or "").lower().strip()
    prompt = (
        f"You are a strict FAANG Technical Interviewer evaluating a candidate's answer.\n"
        f"Interview Question Asked: {question}\n"
        f"Specific Trap Tested: {trap_warning}\n"
        f"Candidate's Answer: {user_ans_clean}\n\n"
        f"Task:\n"
        f"1. Score the candidate from 1 to 10 on accuracy, depth, and edge-case awareness.\n"
        f"2. Give a verdict: 'Outstanding' (9-10), 'Strong' (7-8), 'Partially Correct' (5-6), or 'Missed the Trap' (1-4).\n"
        f"3. Provide 2-3 sentences of constructive feedback.\n"
        f"4. Provide the gold standard FAANG answer.\n"
        f"Return ONLY a JSON object:\n"
        f'{{"score": 8, "verdict": "Strong", "feedback": "...", "gold_standard": "..."}}'
    )

    if p_clean in ("local", "striver-rtx5050", "striver"):
        try:
            from ytrag.local_llm import LocalTeachingAssistant
            assistant = LocalTeachingAssistant.get_instance()
            messages = [
                {"role": "system", "content": "You are an elite DSA interviewer. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
            raw_text, _, _ = assistant.generate_chat_completion(
                messages=messages,
                max_tokens=450,
                temperature=0.1,
            )
            cleaned = raw_text.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
            parsed = json.loads(cleaned)
            if "score" in parsed or "verdict" in parsed:
                return {
                    "score": int(parsed.get("score", 7)),
                    "verdict": parsed.get("verdict", "Strong"),
                    "feedback": parsed.get("feedback", ""),
                    "gold_standard": parsed.get("gold_standard", gold_standard),
                    "source": "local",
                }
        except Exception as e:
            logger.warning("Local grill evaluation failed: %s", e)

    api_key = api_key.strip()
    if api_key:
        try:
            resolved_base_url, resolved_model = get_provider_details(provider, base_url, model)
            payload = {
                "model": resolved_model,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "You are an elite DSA interviewer. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            req = urllib.request.Request(
                f"{resolved_base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(data["choices"][0]["message"]["content"])
            return {
                "score": int(parsed.get("score", 7)),
                "verdict": parsed.get("verdict", "Strong"),
                "feedback": parsed.get("feedback", ""),
                "gold_standard": parsed.get("gold_standard", gold_standard),
                "source": provider,
            }
        except Exception:
            pass  # Fall back to heuristic rule-based evaluation

    # Rule-Based Heuristic Evaluation (Offline)
    ans_lower = user_ans_clean.lower()
    if len(user_ans_clean) < 15:
        return {
            "score": 3,
            "verdict": "Missed the Trap",
            "feedback": "Your answer is too brief to demonstrate technical mastery in an interview setting. Walk through the pointer states or boundary conditions step-by-step.",
            "gold_standard": gold_standard or trap_warning,
            "source": "rule_based",
        }

    # Find matching bank for keyword heuristic
    bank = find_canonical_grill(title, topic, f"{question} {trap_warning}")
    matched_keywords = [kw for kw in bank.get("keywords", []) if kw in ans_lower]
    ratio = len(matched_keywords) / max(1, len(bank.get("keywords", [])))

    if ratio >= 0.4 or len(matched_keywords) >= 3:
        score = min(10, 7 + int(ratio * 3))
        verdict = "Outstanding" if score >= 9 else "Strong"
        feedback = f"Great intuition! You correctly addressed the core mechanism ({', '.join(matched_keywords[:3])}). You demonstrated solid edge-case mastery."
    elif ratio >= 0.2 or len(matched_keywords) >= 1:
        score = 6
        verdict = "Partially Correct"
        feedback = f"Good start, but you missed some critical edge case specifics. Notice how handling: '{trap_warning}' requires explicit boundary guards."
    else:
        score = 4
        verdict = "Missed the Trap"
        feedback = f"You did not address the central technical trap ({trap_warning}). Compare your reasoning against the gold standard answer."

    return {
        "score": score,
        "verdict": verdict,
        "feedback": feedback,
        "gold_standard": gold_standard or bank.get("gold_standard", trap_warning),
        "source": "rule_based",
    }
