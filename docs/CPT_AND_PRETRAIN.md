# Continued pretraining (CPT) and pretraining from scratch

This repository is **primarily a post-training stack** (SFT + DPO + evaluation). CPT and full pretraining are **supported as extensions**, with different engineering and compute requirements.

## Terminology

| Term | Meaning |
|------|---------|
| **Post-training** | SFT / preference optimization on top of a released base checkpoint. |
| **CPT (continued pretraining)** | Additional **unsupervised** training on raw text/code to shift the base distribution before SFT. |
| **Pretraining from scratch** | Training all weights from random init — typically **multi-GPU-weeks** for a 7B-class model. |

## What ships in this repo today

- **Production post-training path** — `scripts/data/prepare.sh`, `scripts/train/sft.sh`, `scripts/train/dpo.sh`.
- **CPT smoke path** — demonstrates that LlamaFactory `stage: pt` is wired to a **local JSONL corpus** with a proper `dataset_info.json`:
  - `./scripts/data/prepare_cpt_smoke.sh` → `artifacts/cpt_data/*`
  - `./scripts/train/cpt.sh` → `configs/train/llamafactory_qwen25_7b_lora_cpt_smoke.yaml`

The smoke corpus defaults to a **WikiText-103 (raw)** slice. It is useful for **pipeline validation**, not for claiming large benchmark gains.

## Compute expectations (order-of-magnitude)

| Workload | Typical scale |
|----------|----------------|
| LoRA SFT + DPO (default recipe) | 1–8× high-memory GPUs depending on sequence length and method. |
| LoRA CPT on a **small** curated corpus | Similar to SFT in cost if token count is comparable. |
| **Full** CPT on web-scale data | Multi-node, data-pipeline-heavy; consider Megatron-Core, NeMo, or DeepSpeed. |
| 7B pretrain from scratch | Large cluster, weeks — out of scope for this template repository. |

## Production CPT checklist

1. **Corpus** — license-clean, heavily deduplicated, language- or domain-balanced; store manifests (hashes, versions) like `artifacts/data_manifest.json`.
2. **Tokenizer / base alignment** — CPT almost always uses the **same tokenizer** as the base model.
3. **Objective** — causal LM next-token loss; monitor held-out perplexity and downstream probes.
4. **Training stack** — once data exceeds ~single-node throughput, plan a **dedicated** NeMo/Megatron (or similar) subtree instead of stretching the smoke YAML.

## References

- LlamaFactory training stages: [hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)
- LlamaFactory dataset format for pretraining (`text` column): see upstream `data/README.md`.
