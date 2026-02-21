# Continuous integration and evaluation harness

## What CI proves today

The GitHub Actions workflow runs:

1. **Lint** — `ruff check tools/`
2. **Structure** — `tools/dry_run.py --strict`
3. **Data path** — capped `prepare_data` (downloads public HF shards)
4. **Data QA** — `tools/data_quality_report.py` on the capped export
5. **Eval harness smoke** — `lm_eval` on **`gpt2`** with a **tiny forward sample** (`--limit`) so we know the evaluation entrypoint is not rotted

The **`gpt2` smoke job is not a scientific claim about Qwen2.5-7B**. It is a **regression test** for `lm-eval` integration. Treat 7B numbers exactly as documented in `docs/RESULTS.md`.

## Artifacts

Each workflow run uploads **`lm-eval-smoke`** containing:

- `results/ci/lm_eval_gpt2.json`
- `results/ci/SUMMARY.md`

Download the artifact from the Actions tab if you need the raw JSON.

## Reproducing locally

```bash
pip install -e ".[eval]"
mkdir -p results/ci
lm_eval --model hf \
  --model_args pretrained=gpt2,dtype=float32 \
  --tasks arc_easy \
  --limit 40 \
  --batch_size 4 \
  --output_path results/ci/lm_eval_gpt2.json
python tools/summarize_lm_eval.py results/ci/lm_eval_gpt2.json --out results/ci/SUMMARY.md
```
