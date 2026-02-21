#!/usr/bin/env python3
"""Print resolved pipeline steps from a recipe (no downloads, no training)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_recipe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", type=Path, default=Path("recipes/default.yaml"))
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    recipe_path = args.recipe if args.recipe.is_absolute() else root / args.recipe
    recipe = load_recipe(recipe_path)

    plan = {
        "stack": "LlamaFactory (SFT + DPO) + lm-evaluation-harness + vLLM",
        "base_model_default": "Qwen/Qwen2.5-7B",
        "recipe_id": recipe.get("recipe_id"),
        "steps": [
            {
                "name": "prepare_data",
                "script": "scripts/data/prepare.sh",
                "outputs": [
                    "artifacts/data/sft.jsonl",
                    "artifacts/data/dpo.jsonl",
                    "artifacts/data/dataset_info.json",
                    "artifacts/data_manifest.json",
                ],
            },
            {
                "name": "sft",
                "script": "scripts/train/sft.sh",
                "config": "configs/train/llamafactory_qwen25_7b_lora_sft.yaml",
            },
            {
                "name": "dpo",
                "script": "scripts/train/dpo.sh",
                "config": "configs/train/llamafactory_qwen25_7b_lora_dpo.yaml",
            },
            {
                "name": "eval",
                "script": "scripts/eval/benchmarks.sh",
                "tasks": "configs/eval/lm_eval_tasks.txt",
            },
            {"name": "serve", "script": "scripts/serve/vllm.sh"},
        ],
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
