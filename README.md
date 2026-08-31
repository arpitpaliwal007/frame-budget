# Spend Frames Where They Matter

**Query-adaptive frame budgets for video-language models.** A controlled study on a
frozen 2B video-LLM asking two questions the literature keeps conflating:

1. Does query-aware keyframe selection actually beat uniform sampling **once you control
   for token budget**? (audit of the AKS / Q-Frame / FOCUS family)
2. Given the answer, is the bigger win choosing *which* frames — or choosing *how many*,
   **per query**? (our contribution: a confidence-gated cascade)

No training. Frozen model, frozen retriever, cached features. Runs end-to-end in one day
on a free Colab T4.

---

## The claim

> Prior work asks **which** frames to send. We show that on short-form video QA the
> selection gain is small and mostly vanishes by 16 frames — but the *cost* of a fixed
> budget is large, because most questions are already answered correctly at 4 frames.
> Routing frame budget per query on a confidence signal recovers full 16-frame accuracy at
> a fraction of the vision-token cost, and composes with any selector.

The deliverable is a **Pareto frontier of accuracy against inference cost**, plus the
oracle upper bound that says how much headroom is left.

## Why this is shaped like Applied Scientist work

| Habit | Where it shows up |
|---|---|
| Falsifiable hypotheses, written before results | `report/predictions.md`, committed first |
| Paired statistics, not eyeballed tables | McNemar on identical questions, bootstrap CIs |
| Effect size reported alongside p-values | `05_analyze.py` prints Δacc, gained/lost counts |
| Upper bounds, so "good" has a scale | soft-oracle selection + oracle routing |
| Cost measured, not assumed | `vlm.tokens_per_frame()` measures the slope empirically |
| Honest negatives | selection gains reported as n.s. where they are n.s. |
| Stratified error analysis | accuracy by question type and video length |
| Reproducibility under failure | append-only JSONL, resume-on-restart, frozen eval set |

## Results to fill in

| Config | Avg frames | Vision tokens | Accuracy [95% CI] | vs uniform-16 |
|---|---|---|---|---|
| uniform-4 | 4.0 | | | |
| uniform-16 | 16.0 | | | |
| mmr-16 | 16.0 | | | |
| **cascade (margin)** | | | | |
| oracle routing | | | | *upper bound* |

## Pipeline

```bash
export FB_ROOT=/content/drive/MyDrive/frame-budget    # persist across Colab disconnects
export FB_VIDEO_ROOT=/content/nextqa/videos

python scripts/01_build_evalset.py      # freeze a stratified 1200-question set
python scripts/02_extract_frames.py     # ffmpeg -> 1fps JPEGs, parallel, resumable
python scripts/03_cache_features.py     # SigLIP embeddings; ~4 min on a T4
python scripts/04_run_eval.py --selectors uniform --budgets 1 2 4 8 16 32
python scripts/04_run_eval.py --selectors mmr topk_sim random --budgets 2 4 8 16
python scripts/05_analyze.py            # stats, tables, results/pareto.png
```

The cascade costs **zero extra GPU time** — it replays cached predictions and
confidences from the fixed-budget runs and sweeps the threshold in numpy.

## Hardware notes that will save you an hour

- A **T4 is Turing (sm_75)**: no bf16, no FlashAttention-2. Use `dtype=float16` and
  `attn_implementation="sdpa"`. Passing `flash_attention_2` fails at load.
- Multiple-choice is scored by reading the **option-letter logits at the first generated
  position** — one forward pass, no decoding loop, fully deterministic. The only source of
  variance across runs is the frame subset, which is exactly the variable under study.
- `frame_selectors.py` is deliberately not named `selectors.py`; that name shadows a
  stdlib module `subprocess` imports and breaks `scipy` and `torch`.

## Honest positioning

Query-aware selection is a crowded area — AKS (CVPR'25), Q-Frame, FOCUS (ICLR'26),
A.I.R., LDDR, VideoRouter. This project **does not claim a new selector**. It claims
(a) a controlled, budget-matched audit of that family on a small open model, and
(b) an orthogonal axis — per-query budget allocation — that those methods leave fixed.
VideoRouter routes token-*allocation policies*; it does not gate on confidence to decide
*how many* frames a query gets.

## Limitations

- One model family and one benchmark. Short videos (~44s) — the regime where selection is
  known to help least. A long-video benchmark would likely favour selection more; that is
  the stated next experiment, not a claim.
- The 2-view agreement signal pays for two cheap passes; the cost accounting charges it
  for both. Do not quietly compare it against single-pass baselines.
- Soft oracle = best-of-8 random subsets, an estimate of achievable-selection headroom,
  not a true combinatorial optimum.
