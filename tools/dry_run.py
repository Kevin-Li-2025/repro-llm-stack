#!/usr/bin/env python3
"""Print resolved pipeline steps from a recipe (no downloads, no training)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_recipe(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Recipe must be a YAML mapping")
    return data


def main() -> int:
    p = argparse.ArgumentParser(description="Print pipeline plan JSON for a recipe")
    p.add_argument("--recipe", type=Path, default=Path("recipes/default.yaml"))
    p.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root for existence checks (default: parent of tools/)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if expected config files are missing",
    )
    args = p.parse_args()

    root = args.root or Path(__file__).resolve().parents[1]
    recipe_path = args.recipe if args.recipe.is_absolute() else root / args.recipe
    if not recipe_path.is_file():
        print(f"Recipe not found: {recipe_path}", file=sys.stderr)
        return 1

    try:
        recipe = load_recipe(recipe_path)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    if recipe.get("schema_version") != 1:
        print("Unsupported schema_version (expected 1).", file=sys.stderr)
        return 1
    if not recipe.get("recipe_id"):
        print("Missing recipe_id.", file=sys.stderr)
        return 1

    cfg_sft = root / "configs/train/llamafactory_qwen25_7b_lora_sft.yaml"
    cfg_dpo = root / "configs/train/llamafactory_qwen25_7b_lora_dpo.yaml"
    cfg_cpt = root / "configs/train/llamafactory_qwen25_7b_lora_cpt_smoke.yaml"
    recipe_cpt = root / "recipes/cpt_smoke.yaml"
    cfg_tasks = root / "configs/eval/lm_eval_tasks.txt"
    missing = [
        str(p.relative_to(root))
        for p in (cfg_sft, cfg_dpo, cfg_cpt, recipe_cpt, cfg_tasks)
        if not p.is_file()
    ]
    if missing and args.strict:
        print(f"Missing expected files: {', '.join(missing)}", file=sys.stderr)
        return 2

    plan: dict = {
        "stack": "LlamaFactory (SFT + DPO) + lm-evaluation-harness + vLLM",
        "base_model_default": "Qwen/Qwen2.5-7B",
        "recipe_id": recipe.get("recipe_id"),
        "checks": {
            "missing_files": missing,
        },
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
            {
                "name": "cpt_smoke_prepare",
                "script": "scripts/data/prepare_cpt_smoke.sh",
                "docs": "docs/CPT_AND_PRETRAIN.md",
                "outputs": [
                    "artifacts/cpt_data/cpt_smoke.jsonl",
                    "artifacts/cpt_data/dataset_info.json",
                    "artifacts/cpt_data_manifest.json",
                ],
            },
            {
                "name": "cpt_smoke_train",
                "script": "scripts/train/cpt.sh",
                "config": "configs/train/llamafactory_qwen25_7b_lora_cpt_smoke.yaml",
            },
        ],
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
