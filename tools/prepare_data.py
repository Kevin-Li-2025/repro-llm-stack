#!/usr/bin/env python3
"""Download HF sources, apply filters, export JSONL + LlamaFactory dataset_info + manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from datasets import load_dataset
    from huggingface_hub import HfApi
except ImportError as e:  # pragma: no cover - import guard for minimal installs
    print("Missing dependency. Install with: pip install -e '.[data]'", file=sys.stderr)
    raise e


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolved_revision(repo_id: str, revision: str | None) -> str:
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision or "main")
    return info.sha or (revision or "main")


def _last_assistant_content(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return str(m.get("content") or "")
    return ""


def _sft_text_len(row: dict[str, Any]) -> int:
    parts = [
        str(row.get("instruction") or ""),
        str(row.get("input") or ""),
        str(row.get("output") or ""),
    ]
    return len("\n".join(parts).strip())


def _dpo_text_len(row: dict[str, Any]) -> int:
    parts = [
        str(row.get("instruction") or ""),
        str(row.get("input") or ""),
        str(row.get("chosen") or ""),
        str(row.get("rejected") or ""),
    ]
    return len("\n".join(parts).strip())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dataset_info() -> dict[str, Any]:
    return {
        "repro_sft": {
            "file_name": "sft.jsonl",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        },
        "repro_dpo": {
            "file_name": "dpo.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", type=Path, default=Path("recipes/default.yaml"))
    p.add_argument("--root", type=Path, default=None, help="Repo root (default: parent of tools/)")
    p.add_argument("--max-sft", type=int, default=None, help="Cap exported SFT rows (debug/smoke)")
    p.add_argument("--max-dpo", type=int, default=None, help="Cap exported DPO rows (debug/smoke)")
    args = p.parse_args()

    root = args.root or Path(__file__).resolve().parents[1]
    recipe_path = args.recipe if args.recipe.is_absolute() else root / args.recipe
    with recipe_path.open("r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f)

    out_dir = root / "artifacts" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft.jsonl"
    dpo_path = out_dir / "dpo.jsonl"
    info_path = out_dir / "dataset_info.json"
    manifest_path = root / "artifacts" / "data_manifest.json"

    filters = recipe.get("filters") or {}
    min_c = int(filters.get("min_chars", 1))
    max_c = int(filters.get("max_chars", 10**9))
    policy_ver = str(recipe.get("filter_policy_version", "v1"))

    sources = recipe.get("sources") or {}
    sft_src = sources.get("sft") or {}
    dpo_src = sources.get("dpo") or {}

    limits = recipe.get("limits") or {}
    max_sft = args.max_sft if args.max_sft is not None else limits.get("max_sft_rows")
    max_dpo = args.max_dpo if args.max_dpo is not None else limits.get("max_dpo_rows")
    if max_sft is not None:
        max_sft = int(max_sft)
    if max_dpo is not None:
        max_dpo = int(max_dpo)

    # --- SFT (Alpaca schema) ---
    sft_path_hf = str(sft_src.get("hf_path", "tatsu-lab/alpaca"))
    sft_split = str(sft_src.get("split", "train"))
    sft_rev_arg = sft_src.get("revision")
    sft_rev_arg = str(sft_rev_arg) if sft_rev_arg not in (None, "") else None
    sft_revision_sha = _resolved_revision(sft_path_hf, sft_rev_arg)

    sft_raw = load_dataset(sft_path_hf, split=sft_split, revision=sft_revision_sha)
    sft_rows: list[dict[str, Any]] = []
    for ex in sft_raw:
        row = {
            "instruction": str(ex.get("instruction") or ""),
            "input": str(ex.get("input") or ""),
            "output": str(ex.get("output") or ""),
        }
        n = _sft_text_len(row)
        if n < min_c or n > max_c:
            continue
        sft_rows.append(row)
        if max_sft is not None and len(sft_rows) >= max_sft:
            break

    # --- DPO (UltraFeedback binarized prefs) ---
    dpo_path_hf = str(dpo_src.get("hf_path", "HuggingFaceH4/ultrafeedback_binarized"))
    dpo_split = str(dpo_src.get("split", "train_prefs"))
    dpo_rev_arg = dpo_src.get("revision")
    dpo_rev_arg = str(dpo_rev_arg) if dpo_rev_arg not in (None, "") else None
    dpo_revision_sha = _resolved_revision(dpo_path_hf, dpo_rev_arg)

    dpo_raw = load_dataset(dpo_path_hf, split=dpo_split, revision=dpo_revision_sha)
    dpo_rows: list[dict[str, Any]] = []
    for ex in dpo_raw:
        prompt = str(ex.get("prompt") or "").strip()
        chosen_t = _last_assistant_content(ex.get("chosen"))
        rejected_t = _last_assistant_content(ex.get("rejected"))
        row = {
            "instruction": prompt,
            "input": "",
            "chosen": chosen_t,
            "rejected": rejected_t,
        }
        n = _dpo_text_len(row)
        if n < min_c or n > max_c:
            continue
        if not chosen_t or not rejected_t:
            continue
        dpo_rows.append(row)
        if max_dpo is not None and len(dpo_rows) >= max_dpo:
            break

    _write_jsonl(sft_path, sft_rows)
    _write_jsonl(dpo_path, dpo_rows)

    dataset_info = _build_dataset_info()
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recipe_id": recipe.get("recipe_id"),
        "recipe_path": str(recipe_path),
        "filter_policy_version": policy_ver,
        "filters": {"min_chars": min_c, "max_chars": max_c},
        "sources": [
            {
                "role": "sft",
                "hf_path": sft_path_hf,
                "split": sft_split,
                "revision_requested": sft_rev_arg,
                "revision_resolved_sha": sft_revision_sha,
                "raw_rows_reported": sft_raw.num_rows,
                "exported_rows": len(sft_rows),
            },
            {
                "role": "dpo",
                "hf_path": dpo_path_hf,
                "split": dpo_split,
                "revision_requested": dpo_rev_arg,
                "revision_resolved_sha": dpo_revision_sha,
                "raw_rows_reported": dpo_raw.num_rows,
                "exported_rows": len(dpo_rows),
            },
        ],
        "exports": {
            "sft_jsonl": {
                "path": str(sft_path.relative_to(root)),
                "sha256": _sha256_file(sft_path),
                "bytes": sft_path.stat().st_size,
                "rows": len(sft_rows),
            },
            "dpo_jsonl": {
                "path": str(dpo_path.relative_to(root)),
                "sha256": _sha256_file(dpo_path),
                "bytes": dpo_path.stat().st_size,
                "rows": len(dpo_rows),
            },
            "llamafactory_dataset_info": str(info_path.relative_to(root)),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {sft_path} ({len(sft_rows)} rows)")
    print(f"Wrote {dpo_path} ({len(dpo_rows)} rows)")
    print(f"Wrote {info_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
