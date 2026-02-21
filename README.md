# repro-llm-stack

Reproducible **end-to-end workflow** for post-training a **~7B** causal language model:

**versioned data manifest → supervised fine-tuning (SFT) → direct preference optimization (DPO) → fixed benchmark suite (`lm-eval`) → optional serving (`vLLM`)**.

The project standardizes on a **single training stack**—[LlamaFactory](https://github.com/hiyouga/LlamaFactory)—to keep configuration, scripts, and documentation consistent. Large-scale **continued pretraining (CPT)** or **pretraining from scratch** is treated as **phase two** (see `scripts/train/cpt.sh`); integrate NeMo, Megatron, or another CPT stack in a dedicated subtree when resources allow.

## Default base model

- **`Qwen/Qwen2.5-7B`** — To use another base (e.g. `mistralai/Mistral-7B-v0.3`), update `model_name_or_path` and the chat **template** in the LlamaFactory YAML files under `configs/train/`.

## Reproducibility guarantees

- **`scripts/data/prepare.sh`** produces **`artifacts/data_manifest.json`**: resolved Hugging Face dataset revisions (SHA), raw versus exported row counts, **SHA-256** checksums of exported JSONL, filter policy version, and **`artifacts/data/dataset_info.json`** for LlamaFactory.
- **Evaluation** always consumes **`configs/eval/lm_eval_tasks.txt`** (seven tasks) and writes **`artifacts/eval/SUMMARY.md`** alongside the raw `lm_eval` JSON output.

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

### 1. Build datasets and manifest

```bash
./scripts/data/prepare.sh
# Optional smoke run: ./scripts/data/prepare.sh recipes/default.yaml --max-sft 1000 --max-dpo 1000
```

### 2. Train (SFT, then DPO)

```bash
./scripts/train/sft.sh
./scripts/train/dpo.sh
```

### 3. Evaluate

```bash
MODEL_PATH=Qwen/Qwen2.5-7B ./scripts/eval/benchmarks.sh
# After merging LoRA into a full Hugging Face checkpoint, set MODEL_PATH to that directory.
```

### 4. Serve (optional)

vLLM is **not** pinned in `pyproject.toml` because wheels are CUDA- and driver-specific. Install vLLM for your stack, **merge adapters into full weights** using LlamaFactory’s export flow, then:

```bash
./scripts/serve/vllm.sh /path/to/merged_hf_model
```

## Repository layout

| Path | Purpose |
|------|---------|
| `configs/train/llamafactory_*.yaml` | Default Qwen2.5-7B LoRA SFT and DPO jobs |
| `configs/eval/lm_eval_tasks.txt` | Fixed regression task list |
| `recipes/default.yaml` | Dataset sources, filters, optional row caps |
| `scripts/` | Thin CLI wrappers (prepare, train, eval, serve) |
| `tools/prepare_data.py` | HF download, filtering, JSONL export, manifest |
| `tools/summarize_lm_eval.py` | `lm_eval` JSON → `SUMMARY.md` |

## Project status

| Stage | Support in this repository |
|------|----------------------------|
| SFT + DPO | Implemented via LlamaFactory YAML and shell wrappers |
| Benchmarking | `lm_eval` with pinned task list and Markdown summary |
| Inference | Optional `vllm` wrapper; merge LoRA before serving |
| CPT / pretrain from scratch | Placeholder only (`scripts/train/cpt.sh`) |

## Utilities

```bash
python3 tools/dry_run.py --recipe recipes/default.yaml
```

## Freezing a release

Set `sources.*.revision` in your recipe YAML to the **dataset commit SHA** recorded in `artifacts/data_manifest.json`, then commit the recipe change and tag a release.

## License

This project is released under the [MIT License](LICENSE).
