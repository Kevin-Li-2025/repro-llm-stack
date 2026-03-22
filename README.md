# repro-llm-stack

[![CI](https://github.com/Kevin-Li-2025/repro-llm-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/Kevin-Li-2025/repro-llm-stack/actions/workflows/ci.yml)

## Why this matters

Post-training effects are often **small, noisy, and data-dependent**. If you cannot **freeze what data was used**, **quantify basic pathology** (duplicates, length skew, degenerate pairs), and **diff evaluations under identical harness settings**, you cannot say what you learned from a run. This project makes those steps **obligatory and scripted**, so “what changed?” is answerable.

## What we measured (data plane — in this repo today)

The default export (`recipes/default.yaml`) has been **materialized and scored** on a **2000-row prefix** of each stream (first passing examples in dataset order — see caveats in the doc):

| Signal | SFT (Alpaca export) | DPO (Ultrafeedback → flat pairs) |
|--------|---------------------|-----------------------------------|
| Mean output / response chars | **255** (p50 **155**, p95 **735**) | chosen **1280** vs rejected **1132** |
| Approx. duplicate rate (normalized hash) | **0** | **0** |
| DPO length prior | — | **55.6%** of pairs have chosen longer than rejected; **0.75%** share a full common prefix up to the shorter length |

Full tables, methodology, and interpretation: **[docs/MEASURED_FINDINGS.md](docs/MEASURED_FINDINGS.md)**.

![Mean character lengths for exported SFT outputs and DPO responses](docs/figures/data_qa_overview.svg)

Regenerate after changing recipes or caps:

`make prepare && make quality && make findings`

## What still belongs on the model plane (your GPUs)

**Downstream benchmark deltas** (base vs merged SFT vs merged DPO on `configs/eval/lm_eval_tasks.txt`) are not substituted by the table above. Produce and commit **`docs/BENCHMARK_TABLE.md`** via [docs/RESULTS.md](docs/RESULTS.md) once checkpoints exist. For how to **narrate** that step (data pathologies → model scores), see [docs/MODEL_OUTCOMES.md](docs/MODEL_OUTCOMES.md).

---

**Primary scope:** a reproducible **post-training** pipeline for a **~7B** causal LM — **data manifest → SFT → DPO → fixed `lm-eval` regression tasks → optional `vLLM`**.

**Extensions:** optional **continued pretraining (CPT) smoke** wiring (`stage: pt` in LlamaFactory) plus documentation for **large-scale CPT / pretrain** (separate compute path). This is **not** a “train 7B from scratch in one click” repository; see [docs/CPT_AND_PRETRAIN.md](docs/CPT_AND_PRETRAIN.md).

The training stack is pinned to **[LlamaFactory](https://github.com/hiyouga/LlamaFactory)** so configs, scripts, and docs stay coherent.

## Positioning

This is **not** a new algorithm repo. The defensible narrative is **experimentation infrastructure + data governance**:

| Lens | What exists here |
|------|------------------|
| **Research honesty** | [docs/RESEARCH.md](docs/RESEARCH.md) — how to describe contributions without overselling. |
| **Preference / data** | [docs/PREFERENCE_AND_DATA.md](docs/PREFERENCE_AND_DATA.md) — exact DPO construction + limitations + synthetic controls. |
| **Data quality metrics** | `tools/data_quality_report.py` → dup proxies, length tails, DPO pair diagnostics (`make quality`). |
| **Ablations** | [experiments/registry.yaml](experiments/registry.yaml) → [docs/ABLATION_REGISTRY.md](docs/ABLATION_REGISTRY.md) (`make experiments-render`). |
| **Eval regression** | CI runs an **`lm-eval` smoke job on `gpt2`** (harness health), *not* a 7B claim — [docs/CI_AND_HARNESS.md](docs/CI_AND_HARNESS.md). |

**You still must produce 7B (or your target) numbers locally** and commit/publish `docs/BENCHMARK_TABLE.md` — see [docs/RESULTS.md](docs/RESULTS.md).

## Default base model

- **`Qwen/Qwen2.5-7B`** — For another base (e.g. `mistralai/Mistral-7B-v0.3`), edit `model_name_or_path` and the chat **template** in `configs/train/`.

## Reproducibility guarantees

- **`scripts/data/prepare.sh`** → **`artifacts/data_manifest.json`**: resolved HF dataset SHAs, row counts, JSONL **SHA-256**, filter policy version, **`artifacts/data/dataset_info.json`**, and **`environment`** metadata (Python / packages / optional git SHA).
- **`scripts/eval/benchmarks.sh`** → always reads **`configs/eval/lm_eval_tasks.txt`** and writes **`artifacts/eval/SUMMARY.md`** next to the raw `lm_eval` JSON.

## End-to-end flow (post-training)

1. **Prepare** — `scripts/data/prepare.sh`
2. **SFT** — `scripts/train/sft.sh`
3. **DPO** — `scripts/train/dpo.sh`
4. **Eval** — `scripts/eval/benchmarks.sh`
5. **Serve** — `scripts/serve/vllm.sh` (after merging LoRA if needed)

## Optional CPT smoke flow (wiring + tiny corpus)

Used to validate **LlamaFactory pretrain / CPT** (`stage: pt`) on a **small public text slice** — not a substitute for web-scale CPT.

1. **Prepare CPT bundle** — `./scripts/data/prepare_cpt_smoke.sh` → `artifacts/cpt_data/`
2. **Train** — `./scripts/train/cpt.sh` → `configs/train/llamafactory_qwen25_7b_lora_cpt_smoke.yaml`

Read [docs/CPT_AND_PRETRAIN.md](docs/CPT_AND_PRETRAIN.md) before claiming production CPT results.

## Evidence: baselines, SFT vs DPO, and “how much better?”

**The repository cannot honestly ship hard benchmark deltas without your GPUs** — scores depend on merged checkpoints, data mix, and training budget. What *is* included:

- **[docs/RESULTS.md](docs/RESULTS.md)** — step-by-step protocol to capture **baseline → SFT → DPO** under identical `lm-eval` settings.
- **[docs/MODEL_OUTCOMES.md](docs/MODEL_OUTCOMES.md)** — closes the **data → model** loop: what to measure next and how to phrase outcomes without overselling.
- **`tools/compare_eval_runs.py`** — turns multiple `lm_eval` JSON files into a **single Markdown comparison table** (with per-task metric keys and deltas).

After you run evals, commit `docs/BENCHMARK_TABLE.md` (generated by the tool), or reuse the table in separate documentation or reports.

## Prerequisites

- Python **3.10+**
- NVIDIA GPU recommended for training and most evaluations
- Hugging Face token only if you use gated models

## Quickstart

```bash
cd repro-llm-stack
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[data]"
pip install -e ".[train]"
pip install -e ".[eval]"
```

### Makefile shortcuts

```bash
make dry-run
make prepare && make quality
make experiments-render
make sft && make dpo
make eval
make prepare-cpt-smoke && make cpt   # optional CPT smoke
make findings                        # MEASURED_FINDINGS.md + SVG (after prepare + quality)
make check
```

### Console entry points (after `pip install -e .`)

- `repro-prepare-data` / `repro-prepare-cpt-smoke`
- `repro-dry-run` / `repro-summarize-eval` / `repro-compare-eval`
- `repro-data-quality` / `repro-experiments-render` / `repro-synth-prefs-demo`
- `repro-render-findings` / `repro-plot-qa-figure`

### 1) Post-training data + manifest

```bash
./scripts/data/prepare.sh
```

### 2) Train SFT, then DPO

```bash
./scripts/train/sft.sh
./scripts/train/dpo.sh
```

### 3) Evaluate (set `MODEL_PATH` to a merged HF folder when using LoRA)

```bash
MODEL_PATH=Qwen/Qwen2.5-7B ./scripts/eval/benchmarks.sh
```

### 4) Serve (optional)

```bash
./scripts/serve/vllm.sh /path/to/merged_hf_model
```

## Repository layout

| Path | Purpose |
|------|---------|
| `configs/train/llamafactory_qwen25_7b_lora_{sft,dpo,cpt_smoke}.yaml` | Default training jobs |
| `configs/eval/lm_eval_tasks.txt` | Fixed regression tasks |
| `recipes/default.yaml` | SFT/DPO HF sources + filters |
| `recipes/cpt_smoke.yaml` | CPT smoke corpus recipe (WikiText) |
| `docs/CPT_AND_PRETRAIN.md` | CPT / pretrain scope + production checklist |
| `docs/RESEARCH.md` | Honest scope and novelty framing |
| `docs/PREFERENCE_AND_DATA.md` | Preference construction + synthetic controls |
| `docs/ABLATION_REGISTRY.md` | Rendered experiment grid |
| `docs/MEASURED_FINDINGS.md` | Committed data-plane measurements + interpretation |
| `docs/figures/data_qa_overview.svg` | Figure for mean lengths / DPO prior (regenerate with `make findings`) |
| `docs/CI_AND_HARNESS.md` | What CI proves / does not prove |
| `docs/RESULTS.md` | How to record baseline vs SFT vs DPO numbers |
| `docs/MODEL_OUTCOMES.md` | Narrative bridge from data QA to benchmark table (no fake scores) |
| `tools/data_quality_report.py` | Quantitative data QA |
| `experiments/registry.yaml` | Source of truth for ablations |
| `tools/compare_eval_runs.py` | Build comparison tables from `lm_eval` JSON |
| `results/eval_runs/README.md` | Where to park JSON outputs for comparisons |

## Project status

| Stage | In this repo |
|-------|----------------|
| SFT + DPO | Fully wired (default track) |
| Regression eval + summary | `lm_eval` + `SUMMARY.md` + comparison tool |
| Inference | `vLLM` wrapper |
| CPT | **Smoke path** (`prepare_cpt_smoke` + `stage: pt` YAML) + **docs** for large-scale CPT |
| Pretrain 7B from scratch | Out of scope — integrate Megatron/NeMo separately when funded |

## Utilities

```bash
python3 tools/dry_run.py --recipe recipes/default.yaml --strict
```

## Troubleshooting

- **`lm-eval` / `torch` install fails on Python 3.13+** — use **Python 3.10–3.12** (e.g. `brew install python@3.11`), then `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[eval]"`.
- **`No module named transformers` when running `lm_eval`** — install eval extras (`pip install -e ".[eval]"`), which include `transformers` and `torch` for the Hugging Face backend.
- **`prepare_data` exit code 2** — empty export; relax filters or use `--allow-empty` for debugging only.
- **`trust_remote_code` from `datasets`** — set `load_dataset_kwargs` in the recipe.
- **Template mismatch** — align LlamaFactory `template` with the model family.

## Freezing a release

Pin `sources.*.revision` in recipes to the SHAs recorded under `artifacts/*_manifest.json`.

## License

[MIT](LICENSE).
