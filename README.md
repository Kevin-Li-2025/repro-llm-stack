# repro-llm-stack

[![CI](https://github.com/Kevin-Li-2025/repro-llm-stack/actions/workflows/ci.yml/badge.svg)](https://github.com/Kevin-Li-2025/repro-llm-stack/actions/workflows/ci.yml)

Reproducible **end-to-end workflow** for post-training a **~7B** causal language model:

**versioned data manifest → supervised fine-tuning (SFT) → direct preference optimization (DPO) → fixed benchmark suite (`lm-eval`) → optional serving (`vLLM`)**.

The project standardizes on a **single training stack**—[LlamaFactory](https://github.com/hiyouga/LlamaFactory)—to keep configuration, scripts, and documentation consistent. Large-scale **continued pretraining (CPT)** or **pretraining from scratch** is treated as **phase two** (see `scripts/train/cpt.sh`); integrate NeMo, Megatron, or another CPT stack in a dedicated subtree when resources allow.

## Default base model

- **`Qwen/Qwen2.5-7B`** — To use another base (e.g. `mistralai/Mistral-7B-v0.3`), update `model_name_or_path` and the chat **template** in the LlamaFactory YAML files under `configs/train/`.

## Reproducibility guarantees

- **`scripts/data/prepare.sh`** produces **`artifacts/data_manifest.json`**: resolved Hugging Face dataset revisions (SHA), raw versus exported row counts, **SHA-256** checksums of exported JSONL, filter policy version, **`artifacts/data/dataset_info.json`** for LlamaFactory, plus **`environment`** metadata (Python and key package versions, optional git commit).
- **Evaluation** always consumes **`configs/eval/lm_eval_tasks.txt`** (seven tasks) and writes **`artifacts/eval/SUMMARY.md`** alongside the raw `lm_eval` JSON output.

## End-to-end flow

1. **Prepare** — download, filter, export JSONL, write manifest (`scripts/data/prepare.sh`).
2. **SFT** — LlamaFactory LoRA supervised fine-tuning (`scripts/train/sft.sh`).
3. **DPO** — LlamaFactory DPO on preference pairs (`scripts/train/dpo.sh`).
4. **Eval** — `lm_eval` on the fixed task list; summarize to Markdown (`scripts/eval/benchmarks.sh`).
5. **Serve** — optional `vLLM` after merging adapters (`scripts/serve/vllm.sh`).

## Prerequisites

- Python **3.10+**
- NVIDIA GPU recommended for training and most evaluation setups
- A [Hugging Face](https://huggingface.co/) token if you pull gated models (optional for the default public datasets)

## Quickstart

```bash
cd repro-llm-stack
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[data]"   # data preparation only
pip install -e ".[train]"  # LlamaFactory + PyTorch (install the CUDA wheel that matches your environment)
pip install -e ".[eval]"   # lm-evaluation-harness (`lm_eval` CLI)
```

### Makefile shortcuts

```bash
make dry-run    # validates recipe + required config files (--strict)
make prepare
make sft
make dpo
make eval
make check      # lint + dry-run
```

### Console entry points (after `pip install -e .`)

- `repro-prepare-data` — same as `python tools/prepare_data.py`
- `repro-dry-run` — same as `python tools/dry_run.py`
- `repro-summarize-eval` — same as `python tools/summarize_lm_eval.py`

### 1. Build datasets and manifest

```bash
./scripts/data/prepare.sh
# Optional smoke run: ./scripts/data/prepare.sh recipes/default.yaml --max-sft 1000 --max-dpo 1000
```

`tools/prepare_data.py` exits with **2** if either export would be empty (filters too tight). Use **`--allow-empty`** only for debugging.

### 2. Train (SFT, then DPO)

```bash
./scripts/train/sft.sh
./scripts/train/dpo.sh
# Extra LlamaFactory CLI flags: ./scripts/train/sft.sh path/to/custom.yaml -- ...
```

### 3. Evaluate

```bash
MODEL_PATH=Qwen/Qwen2.5-7B ./scripts/eval/benchmarks.sh
# Custom task list, then extra lm_eval flags:
# ./scripts/eval/benchmarks.sh configs/eval/lm_eval_tasks.txt --limit 0.1
# After merging LoRA into a full Hugging Face checkpoint, set MODEL_PATH to that directory.
```

### 4. Serve (optional)

vLLM is **not** pinned in `pyproject.toml` because wheels are CUDA- and driver-specific. Install vLLM for your stack, **merge adapters into full weights** using LlamaFactory’s export flow, then:

```bash
./scripts/serve/vllm.sh /path/to/merged_hf_model --tensor-parallel-size 1
```

## Repository layout

| Path | Purpose |
|------|---------|
| `configs/train/llamafactory_*.yaml` | Default Qwen2.5-7B LoRA SFT and DPO jobs |
| `configs/eval/lm_eval_tasks.txt` | Fixed regression task list |
| `recipes/default.yaml` | Dataset sources, filters, optional row caps, optional `load_dataset_kwargs` |
| `scripts/` | Thin CLI wrappers (prepare, train, eval, serve) |
| `tools/prepare_data.py` | HF download, filtering, JSONL export, manifest |
| `tools/summarize_lm_eval.py` | `lm_eval` JSON → `SUMMARY.md` |
| `Makefile` | Common automation targets |
| `.github/workflows/ci.yml` | Ruff + capped `prepare_data` smoke test |

## Project status

| Stage | Support in this repository |
|------|----------------------------|
| SFT + DPO | Implemented via LlamaFactory YAML and shell wrappers |
| Benchmarking | `lm_eval` with pinned task list and Markdown summary |
| Inference | Optional `vllm` wrapper; merge LoRA before serving |
| CPT / pretrain from scratch | Placeholder only (`scripts/train/cpt.sh`) |

## Utilities

```bash
python3 tools/dry_run.py --recipe recipes/default.yaml --strict
```

## Troubleshooting

- **`prepare_data` exits with code 2`** — No rows passed filters, or all SFT rows lacked `output`. Widen `filters` in the recipe or pass `--allow-empty` for debugging.
- **`trust_remote_code` errors from `datasets`** — Set `load_dataset_kwargs.trust_remote_code: true` in your recipe (see comment in `recipes/default.yaml`).
- **Template mismatch after swapping the base model** — Align LlamaFactory `template` with the model family (Qwen vs Mistral, etc.).

## Freezing a release

Set `sources.*.revision` in your recipe YAML to the **dataset commit SHA** recorded in `artifacts/data_manifest.json`, then commit the recipe change and tag a release.

## License

This project is released under the [MIT License](LICENSE).
