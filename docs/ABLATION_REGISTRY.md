# Ablation registry

This file is **generated** from `experiments/registry.yaml`. Edit the YAML, then run `python tools/experiments_render.py`.

| ID | Stage | What changes | Notes |
|----|-------|--------------|-------|
| E0-baseline | none | Base checkpoint before any adapters (merged full weights if comparing to LoRA runs). | MODEL_PATH=Qwen/Qwen2.5-7B ./scripts/eval/benchmarks.sh |
| E1-sft-default | sft | Default LoRA SFT recipe (configs/train/llamafactory_qwen25_7b_lora_sft.yaml). (learning_rate=0.0001, num_train_epochs=1.0) |  |
| E2-dpo-beta-0.05 | dpo | DPO with smaller beta (sharper preference signal). (pref_beta=0.05) |  |
| E3-dpo-beta-0.2 | dpo | DPO with larger beta (softer KL to reference). (pref_beta=0.2) |  |
| E4-data-cap-50pct | sft+dpo | Train on a random 50% subset of exported JSONL to test data scaling (implement via LlamaFactory max_samples or external shuffle). (data_fraction=0.5) |  |
| E5-cpt-smoke-then-sft | cpt+sft | Optional CPT smoke (WikiText slice) before SFT — expect small or mixed moves on knowledge-heavy tasks. | ./scripts/data/prepare_cpt_smoke.sh && ./scripts/train/cpt.sh then standard SFT |

## How to use

1. Run training + eval for each row you care about, saving `lm_eval` JSON under `results/eval_runs/`.
2. `python tools/compare_eval_runs.py --run ... --out docs/BENCHMARK_TABLE.md`
3. Paste deltas into your write-up; keep the JSON for auditability.

