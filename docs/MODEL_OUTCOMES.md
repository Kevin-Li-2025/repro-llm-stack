# Do data pathologies affect model outcomes?

Data-plane measurements in [MEASURED_FINDINGS.md](MEASURED_FINDINGS.md) describe **structure and skew** in the exported SFT/DPO JSONL (length tails, DPO pair priors, duplicate proxies, and so on). They do not, by themselves, answer the model-plane question:

> Given these signals, **what happens to benchmark scores** under a fixed eval harness?

This page is the **bridge**: how to connect those observations to **base vs SFT vs DPO** numbers, and how to write about results honestly.

## Minimal experiment (same harness, merged checkpoints)

Follow [RESULTS.md](RESULTS.md) end to end:

1. **Baseline** — `lm-eval` on the base model.
2. **SFT** — train, **merge LoRA** if used, rerun `lm-eval` with identical task list and settings.
3. **DPO** — same as SFT for the DPO output.

Generate a committed table with:

```bash
python tools/compare_eval_runs.py \
  --run baseline=results/eval_runs/baseline.json \
  --run sft=results/eval_runs/sft_merged.json \
  --run dpo=results/eval_runs/dpo_merged.json \
  --out docs/BENCHMARK_TABLE.md
```

Until that file exists, treat model-level claims as **not yet evidenced** in this repository.

## How to read the loop (data → model → claim)

| Plane | What you show | Typical artifact |
|-------|----------------|------------------|
| Data | Measurable pathologies and caveats | [MEASURED_FINDINGS.md](MEASURED_FINDINGS.md), `artifacts/data_quality_report.json` |
| Model | Scores under frozen task list + dtype + checkpoint merge policy | [docs/BENCHMARK_TABLE.md](BENCHMARK_TABLE.md) (after you generate it) |
| Claim | Sentences that tie (1) to (2) without overreach | Your paper / README / report text |

## Interpretation template (edit **after** you have numbers)

The bullets below are **not findings** for this repo until they are supported by your `BENCHMARK_TABLE.md` and training notes. Replace or delete each line based on what you actually measured.

**Suggested framing for a short “model outcomes” subsection:**

We run a minimal comparison (**base vs SFT vs DPO**) under **identical** `lm-eval` settings and the task list in `configs/eval/lm_eval_tasks.txt`.

*Preliminary results (fill in from your table):*

- Despite structural biases in the DPO export (e.g. length skew summarized in [MEASURED_FINDINGS.md](MEASURED_FINDINGS.md)), **[describe whether]** benchmark deltas **[were small / inconsistent / task-dependent]** relative to baseline and SFT.
- **[If applicable]** Some alignment-adjacent tasks **[improved / did not improve]** after DPO; **[note]** general benchmarks **[did / did not]** consistently favor DPO over SFT.
- Taken together, this **[supports / contradicts / is inconclusive for]** the hypothesis that **data quality and structure add variance that can obscure post-training effects** at modest budgets.

*Scope statement (keep even when numbers exist):*

These results are **preliminary** and intended to demonstrate the **connection** between data-plane measurements and model-plane behavior under **this** harness — not a broad claim about all datasets or all benchmarks.

## What reviewers (and you) should insist on

- **Same** task file, **same** `lm-eval` version, **same** dtype and batch settings across runs.
- **Merged** weights when training used LoRA, unless you explicitly document adapter-at-eval behavior.
- **One row** in the comparison table per checkpoint; link or name checkpoints so the table is reproducible.

For scope and honesty when describing the project elsewhere, see [RESEARCH.md](RESEARCH.md).
