---
title: Striver DSA Platform
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# YouTube Lecture RAG: Bias-Aware Timestamped Video Retrieval & Synthesis

A precision-engineered retrieval-augmented generation (RAG) system for technical YouTube lecture videos. Built to solve the **video timestamp precision problem**: taking student programming doubts (symptom descriptions, conceptual confusion, algorithm mechanics) and navigating directly to the exact second where the explanation begins.

---

## 1. Pipeline Architecture

```
[Student Query]
       │
       ├──► BM25 Lexical Channel (75s Windows, 15s Stride) ────────┐
       │                                                            ▼
       └──► Dense Semantic Channel (all-MiniLM-L6-v2 / BGE-small) ──► Reciprocal Rank Fusion (RRF, k=60)
                                                                            │
                                                                            ▼
                                                                 Top-K Retrieved Chunks
                                                                            │
                                                                            ▼
                                                                 Sub-Chunk Seek Refinement
                                                            (BM25 over 1-seg sliding window, pad=3s)
                                                                            │
                                                                            ▼
                                                                 Final Video Seek Timestamps
                                                                            │
                                                                            ▼
                                                                 Answer Synthesis
                                                            Default: Extractive Citation
                                                            Optional: LoRA Qwen-0.5B Assistant
```

### Core Components
1. **Dynamic Transcript Chunking (`ytrag/chunk.py`)**:
   - Spoken-audio transcripts partitioned into sliding contextual windows ($75s$ length, $15s$ stride).
2. **Hybrid Multi-Stage Retrieval (`ytrag/retriever.py`)**:
   - **Lexical Channel**: BM25Okapi over alphanumeric word tokens.
   - **Dense Channel**: Unit-normalized semantic cosine similarity using `sentence-transformers`.
   - **Rank Fusion**: Reciprocal Rank Fusion ($RRF(d) = \sum \frac{1}{60 + \text{rank}(d)}$).
3. **Sub-Chunk Seek Refinement (`ytrag/refine.py`)**:
   - Snaps the playback seek point from the coarse chunk start to the specific sentence introducing the queried concept.
4. **Answer Generation (`ytrag/extractive.py` & `ytrag/local_llm.py`)**:
   - **Production Default**: Deterministic extractive synthesis with timestamp citations (<0.05s latency, zero hallucinations).
   - **Experimental**: Fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` LoRA assistant (gated behind `USE_LOCAL_LLM=1`).

---

## 2. Pre-Registered Test Benchmark

Evaluated on the held-out test split ($n = 30$ queries) under a cryptographic pre-registration lock (`prereg.lock.json`, SHA-256 `2d43eff671ed`):

### Primary Metric: `seek` Hit Definition
*Hit iff same video and $\text{gt.start} - 20s \le \text{result.start} \le \text{gt.start} + 0.5 \times \text{len}(\text{gt})$.*

| System | Hit@1 (95% CI) | Hit@3 (95% CI) | Hit@5 (95% CI) | MRR@3 (95% CI) | Median \|Seek Error\| |
|---|---|---|---|---|---|
| **`bm25`** | 3.3% [0.6–16.7%] | 16.7% [7.3–33.6%] | 26.7% [14.2–44.4%] | 0.094 [0.022–0.183] | 26.0s ($n=13$) |
| **`hybrid`** | 10.0% [3.5–25.6%] | 20.0% [9.5–37.3%] | 26.7% [14.2–44.4%] | 0.150 [0.050–0.267] | 26.0s ($n=13$) |
| **`hybrid_refined`** | 6.7% [1.8–21.3%] | **30.0% [16.7–47.9%]** | **40.0% [24.6–57.7%]** | **0.161 [0.072–0.267]** | **11.0s ($n=18$)** |
| **`bge_refined`** | 3.3% [0.6–16.7%] | 26.7% [14.2–44.4%] | 40.0% [24.6–57.7%] | 0.133 [0.050–0.222] | **11.0s ($n=17$)** |

### Secondary Metric: `overlap` Hit Definition
*Hit iff same video and $\text{overlap} \ge \min(0.5 \times \text{len}(\text{gt}), 30s)$.*

| System | Hit@1 (95% CI) | Hit@3 (95% CI) | Hit@5 (95% CI) | MRR@3 (95% CI) |
|---|---|---|---|---|
| **`bm25`** | 13.3% [5.3–29.7%] | 26.7% [14.2–44.4%] | 43.3% [27.4–60.8%] | 0.200 [0.083–0.333] |
| **`hybrid`** | 16.7% [7.3–33.6%] | **40.0% [24.6–57.7%]** | **46.7% [30.2–63.9%]** | **0.272 [0.150–0.400]** |
| **`hybrid_refined`** | 13.3% [5.3–29.7%] | 30.0% [16.7–47.9%] | 36.7% [21.9–54.5%] | 0.200 [0.083–0.322] |
| **`bge_refined`** | 10.0% [3.5–25.6%] | 30.0% [16.7–47.9%] | 40.0% [24.6–57.7%] | 0.189 [0.083–0.300] |

### Pre-Registered Statistical Decision Rule
- **Primary Comparison**: `hybrid_refined` vs `hybrid` at $k=3$ under `seek` hit definition.
- **Pre-Registered Rule**: Claim improvement iff exact paired McNemar $p < 0.05$; otherwise report no detectable difference.
- **Outcome**: 3 queries found only by `hybrid`, 6 found only by `hybrid_refined`, 3 by both, 18 by neither. Exact McNemar **$p = 0.5078$**.
- **Conclusion**: **Null result.** The difference between `hybrid_refined` and `hybrid` does not achieve statistical significance on $n=30$ queries ($p \ge 0.05$). Refinement reduced median start error by $15s$ (from $26s$ to $11s$), but hit rate separation was within noise.
- **Embedder Comparison**: Dense encoder comparison between `all-MiniLM-L6-v2` (30.0% seek Hit@3) and `BAAI/bge-small-en-v1.5` (26.7% seek Hit@3) showed no detectable difference.

---

## 3. LLM Fine-Tuning & Evaluation: LoRA vs. Few-Shot vs. Extractive

We trained a LoRA adapter on `Qwen/Qwen2.5-0.5B-Instruct` (`q_proj, k_proj, v_proj, o_proj`, $r=8, \alpha=16$, lr=$3\times 10^{-4}$) on a leak-free 60-example synthetic dataset, and ran a paired comparative evaluation on the dev split ($n=22$ queries: 16 answerable, 6 unanswerable):

| Metric | Base (Zero-Shot) | Base (Few-Shot, 2 Exemplars) | LoRA Tuned Assistant (0.5B) | Extractive Default |
|---|---|---|---|---|
| **Valid JSON Schema (First Try)** | 68% [47–84%] | 45% [27–65%] | **86% [67–95%]** | **100%** |
| **Valid JSON ($\le 2$ Retries)** | 86% [67–95%] | 73% [52–87%] | **100% [85–100%]** | **100%** |
| **Integer Source Resolution ($s \in [1, k]$)** | 15% [4–42%] | **100% [70–100%]** | **100% [72–100%]** | **100%** |
| **Grounding Proxy ($\ge 50\%$ word overlap)** | 8% [1–33%] | 67% [35–88%] | 50% [24–76%] | **100%** |
| **answer_ok** (Strict Schema + Grounded) | 0/16 (0%) | **5/16 (31%)** | 1/16 (6%) | 3/16 (19%) |
| **False Abstain Rate** (Refused valid question) | **0/16 (0%)** | **0/16 (0%)** | 4/16 (25%) | **0/16 (0%)** |
| **Abstain OK** (Refused unanswerable question) | 0/6 (0%) | 0/6 (0%) | 0/6 (0%) | **2/6 (33%)** |
| **Inference Latency** | Heavy CPU | Heavy CPU | Heavy CPU | **<0.05s** |

### Key Findings:
1. **LoRA buys schema adherence, not domain knowledge**: Fine-tuning taught the 0.5B model to reliably output the JSON schema without conversational preambles (first-try validity: 86% vs 45%, $p = 0.022$). However, it did not improve question answering accuracy over few-shot prompting (1/16 vs 5/16).
2. **Citation indexing does not require tuning**: Two in-context multi-turn examples completely closed the source-resolution gap from 15% to 100%.
3. **The False Abstention Problem**: The LoRA model learned an over-cautious refusal behavior, falsely returning empty bullets on 25% of answerable questions.
4. **Why Extractive is Default**: The deterministic non-LLM sentence extraction baseline has zero hallucination risk, zero format failures, lower false abstentions, and runs in 50ms, making it the superior production choice. The local LoRA model remains gated behind `USE_LOCAL_LLM=1`.

---

## 4. Leakage Audit & Dataset Integrity

Fine-tuning records were strictly audited against the evaluation set using `leakage_check.py`:
- **Golden Query Leakage**: 0/22 dev queries, 0/30 test queries leaked (Jaccard $\ge 0.80$).
- **Answer Span Leakage**: 0/22 dev spans, 0/30 test spans leaked (shared 8-grams).
- **Train/Val Transcript Overlap**: 0/15 validation records share transcript text with training (disjoint video split).

---

## 5. Explicit Limitations

1. **Transcript-Derived Labels**: Ground-truth target spans were identified from transcript text; human blind video verification was not conducted. Reported hit rates reflect alignment with transcript-derived boundaries.
2. **Sample Size & Statistical Power**: With $n=22$ dev queries and $n=30$ test queries, confidence intervals span 20–30 percentage points. The sample size is insufficient to statistically separate close configurations.
3. **Dev-Selected Systems**: All candidate retriever configurations and refinement parameters were selected based on exploration on the dev split prior to locking the test evaluation.
4. **Held-Out Questions vs. Held-Out Lectures**: The main training set excluded golden questions and spans, but drew background windows from the same set of 8 lectures. Video-level holdout tooling (`video_holdout.py`) is provided for scaling to multi-course datasets.

---

## 6. Getting Started

### Installation
```bash
git clone https://github.com/your-username/yt-lecture-rag.git
cd yt-lecture-rag
python -m venv .venv
# Activate virtual environment
pip install -r requirements.txt
```

### Running the API & UI
```bash
# Start FastAPI backend (serves web UI at http://localhost:8000)
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

### Reproducing Evaluations
```bash
# 1. Run all unit tests (129 tests)
python -m pytest tests/

# 2. Verify leakage audit
python leakage_check.py --golden eval/golden.json --train training/data/train.jsonl --val training/data/val.jsonl --transcripts transcripts/

# 3. Check pre-registration lock status
python preregister.py status --lock prereg.lock.json

# 4. Run dev evaluation
python eval_harness.py --golden eval/golden.json --split dev --runs bm25=runs/run_bm25.json hybrid=runs/run_hybrid.json hybrid_refined=runs/run_hybrid_minilm_refined.json
```

---

## 7. License
MIT License.
