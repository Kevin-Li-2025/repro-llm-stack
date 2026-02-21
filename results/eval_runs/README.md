# Stored `lm_eval` JSON outputs

Copy the `artifacts/eval/lm_eval_*.json` files you care about into this directory with stable names (`baseline.json`, `sft_merged.json`, …) so `tools/compare_eval_runs.py` can build `docs/BENCHMARK_TABLE.md`.

These files can be large; committing them is optional. The important artifact for applications is usually the **generated Markdown table** plus the **exact commands** documented in `docs/RESULTS.md`.
