# Research positioning (how to describe this project accurately)

Strong research and engineering work rarely stops at “I can call APIs.” **This repository is not a new algorithm paper.** Its defensible story is **measurement + data governance + reproducible experimental protocol** around standard post-training.

## What is *not* novel here

- **SFT, DPO, LoRA, LlamaFactory, lm-eval** — all established tooling.
- **Default datasets** (Alpaca-style SFT + Ultrafeedback prefs) — community baselines, not a new corpus.

Claiming otherwise without evidence will undermine credibility.

## What *is* a legitimate contribution angle

Think in terms of **systems for scientific iteration**:

1. **Manifest-first data pipeline** — every export is tied to HF revisions, byte hashes, filter versions, and environment metadata (`artifacts/data_manifest.json`). That is a **reproducibility instrument**, not a bash one-liner.
2. **Quantitative data QA** — `tools/data_quality_report.py` turns “we filtered stuff” into **dup rates, length tails, cheap repetition proxies, DPO pair diagnostics** (`artifacts/data_quality_report.json`).
3. **Documented preference construction** — `docs/PREFERENCE_AND_DATA.md` states exactly how multi-turn prefs collapse into DPO rows and what information is discarded.
4. **Pre-registered ablation grid** — `experiments/registry.yaml` + rendered `docs/ABLATION_REGISTRY.md` gives you a **hypothesis checklist** (β, data caps, CPT-then-SFT) instead of ad-hoc sweeps.
5. **Comparative evaluation harness** — `tools/compare_eval_runs.py` standardizes **before/after tables** from `lm-eval` JSON so gains are **auditable**, not screenshot-deep.
6. **Synthetic preference demo** — `tools/synth_preference_demo.py` supports **controlled** studies where human data noise would confound implementation work.

## One-line summary

> “I built a reproducible post-training stack where **data exports are versioned and quality-scored**, **preference construction is explicit**, and **evaluations are diffed automatically** — so iterative experiments behave like experiments, not one-off training runs.”

## What you still must do externally

- **Actually run** the ablations and commit `docs/BENCHMARK_TABLE.md` (see `docs/RESULTS.md`).
- Ideally attach **one figure**: e.g. bar chart of task averages vs baseline, with error bars if you repeat seeds.
- For a deeper follow-on project, pick **one axis** to deepen next (e.g. *data-centric* study on filtering / synthetic mix rates, or *alignment* study on off-policy DPO assumptions) and cite this repo as **infrastructure**.

## CPT scope

CPT in this repo is a **wiring + smoke** path plus documentation for scale (`docs/CPT_AND_PRETRAIN.md`). Useful framing: CPT’s place in the lifecycle, and how large-scale pretrain would attach — with a **minimal executable slice** in-tree.
