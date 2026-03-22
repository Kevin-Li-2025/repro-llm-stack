# Measured findings (data plane)

Quantitative summary of the **exported** SFT and DPO JSONL used by this repository (after recipe filters). These metrics support data-governance claims; they do not replace model-plane benchmarks.

## Summary

| Quantity | SFT (`repro_sft`) | DPO (`repro_dpo`) |
|----------|-------------------|-------------------|
| Rows | 2000 | 2000 |
| Primary text length (chars), mean | output **255.0** | chosen **1279.6** · rejected **1131.9** |
| Primary text length, p50 / p95 | **155.0** / **735.1** | Δ(chosen−rejected) p50 **58.5** |
| Approx. duplicate fraction (normalized SHA-256) | **0.0000** | **0.0000** |
| Other | trigram repetition (output), mean **0.0184** | P(chosen longer) **55.6%** · shared-prefix **0.75%** |

## Machine-readable data

- **JSON snapshot:** [`docs/metrics/data_plane_snapshot.json`](metrics/data_plane_snapshot.json) (same numbers as below; safe to cite programmatically).
- **Figure:** [`docs/figures/data_qa_overview.svg`](figures/data_qa_overview.svg)

## Regeneration

This file is produced from `artifacts/data_quality_report.json` after `prepare` and `data_quality_report`.

```bash
python tools/prepare_data.py --root . --max-sft 2000 --max-dpo 2000   # or full export
python tools/data_quality_report.py --root .
python tools/render_measured_findings.py --root .
python tools/plot_qa_figure.py --root .
```

Makefile: `make prepare && make quality && make findings`

## Scope and limitations

- Statistics describe **exported JSONL**, not complete Hugging Face dataset tables.
- With row caps, examples are the **first** passing rows in dataset order (not an i.i.d. subsample unless you change export logic).
- **recipe_id:** `default-v0`
- **git_commit** (recorded in manifest at export): `47f2c2608ba78eff869141c850a87363521bbf4a`

## SFT export (`repro_sft`)

- **rows:** 2000
- **mean output chars:** 254.97
- **p50 / p95 output chars:** 155.0 / 735.1
- **approx. duplicate fraction (normalized hash):** 0.0000
- **mean trigram repetition score (outputs):** 0.0184

### Interpretation

- Near-zero **duplicate fraction** under this hash indicates diverse superficial forms after normalization (not semantic deduplication).
- **Trigram repetition** flags templated or repetitive `output` text; investigate if the mean is unusually high for the domain.

## DPO export (`repro_dpo`, Ultrafeedback → flat pairs)

- **rows:** 2000
- **mean chosen / rejected chars:** 1279.60 / 1131.92
- **p50 length delta (chosen − rejected):** 58.5
- **fraction chosen longer than rejected:** 0.556
- **fraction with full shared prefix (up to shorter length):** 0.0075
- **approx. duplicate fraction:** 0.0000

### Interpretation

- **Chosen longer than rejected** above ~0.5 is common in helpfulness-style preferences but is **not** proof of quality; length can track hedging or verbosity.
- **Shared-prefix rate** near zero here is consistent with pairs that diverge early (e.g. different assistant continuations after the same user prompt).

## Provenance

- **`data_quality_report.created_at`:** `2026-03-22T15:14:40.454243+00:00`
- **Snapshot written to:** `docs/metrics/data_plane_snapshot.json`
