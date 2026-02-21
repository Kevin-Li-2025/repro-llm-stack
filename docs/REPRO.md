# Reproducibility contract

## What we pin

- **OS / image:** Docker image digest after first successful build (record in your run log).
- **Python:** version in `pyproject.toml` classifiers / CI matrix.
- **CUDA / PyTorch:** match the wheel index you install from; record `torch.version.cuda` and `torch.__version__`.
- **HF ecosystem:** `transformers`, `datasets`, `accelerate`, plus **LlamaFactory** version / git tag.
- **Eval:** `lm-evaluation-harness` git tag or pip version.
- **Seeds:** set on data shuffling, init, dropout — record in training config.

## Hardware reporting

For each published result, record:

- GPU model, count, interconnect (NVLink/PCIe), approximate **TFLOPs-s** or at least **GPU-hours**
- peak memory per GPU
- effective throughput (tokens/s/GPU) from trainer logs

## Artifacts

Each run should emit under `artifacts/`:

- `data_manifest.json` — dataset revisions, row counts, checksums
- `train_config.yaml` — resolved config (Hydra-style dump if you use Hydra)
- `eval/` — raw lm-eval outputs + a small `SUMMARY.md` you paste into releases

## “Same results?” expectations

LLM training is **not bitwise reproducible** across:

- different GPU models
- different framework minor versions
- different cuDNN / flash-attn builds

Target **statistical reproducibility**: eval metrics within small tolerance across two seeds / two nodes when everything else is fixed.
