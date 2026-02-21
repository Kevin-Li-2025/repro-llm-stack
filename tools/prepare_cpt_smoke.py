#!/usr/bin/env python3
"""Build a tiny continued-pretraining (CPT) corpus for LlamaFactory `stage: pt` smoke tests."""

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
except ImportError as e:  # pragma: no cover
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


def main() -> int:
    p = argparse.ArgumentParser(description="Export CPT smoke JSONL + dataset_info for LlamaFactory pretrain")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--recipe", type=Path, default=Path("recipes/cpt_smoke.yaml"))
    p.add_argument("--max-docs", type=int, default=20_000, help="Max non-empty documents to export")
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--max-chars", type=int, default=8000, help="Truncate documents longer than this")
    args = p.parse_args()

    root = args.root or Path(__file__).resolve().parents[1]
    recipe_path = args.recipe if args.recipe.is_absolute() else root / args.recipe
    if not recipe_path.is_file():
        print(f"CPT recipe not found: {recipe_path}", file=sys.stderr)
        return 1

    with recipe_path.open("r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    if not isinstance(recipe, dict) or recipe.get("schema_version") != 1:
        print(f"Invalid CPT recipe: {recipe_path}", file=sys.stderr)
        return 1

    src = recipe.get("source") or {}
    hf_path = str(src.get("hf_path", "wikitext"))
    hf_config = src.get("config")
    split = str(src.get("split", "train"))
    rev = src.get("revision")
    rev_s = str(rev) if rev not in (None, "") else None
    revision_sha = _resolved_revision(hf_path, rev_s)

    out_dir = root / "artifacts" / "cpt_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "cpt_smoke.jsonl"
    info_path = out_dir / "dataset_info.json"
    manifest_path = root / "artifacts" / "cpt_data_manifest.json"

    load_kw: dict[str, Any] = {"trust_remote_code": False}
    extra = recipe.get("load_dataset_kwargs")
    if isinstance(extra, dict):
        load_kw.update(extra)

    if hf_config is not None:
        ds = load_dataset(
            hf_path,
            str(hf_config),
            split=split,
            revision=revision_sha,
            **load_kw,
        )
    else:
        ds = load_dataset(hf_path, split=split, revision=revision_sha, **load_kw)

    rows_out = 0
    with jsonl_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            text = str(ex.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            if len(text) > args.max_chars:
                text = text[: args.max_chars]
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            rows_out += 1
            if rows_out >= args.max_docs:
                break

    if rows_out == 0:
        print("CPT smoke export produced zero rows.", file=sys.stderr)
        return 2

    dataset_info = {
        "repro_cpt": {
            "file_name": "cpt_smoke.jsonl",
            "columns": {"prompt": "text"},
        }
    }
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recipe_path": str(recipe_path),
        "source": {
            "hf_path": hf_path,
            "config": hf_config,
            "split": split,
            "revision_resolved_sha": revision_sha,
            "raw_rows_reported": getattr(ds, "num_rows", None),
            "exported_rows": rows_out,
        },
        "exports": {
            "cpt_jsonl": {
                "path": str(jsonl_path.relative_to(root)),
                "sha256": _sha256_file(jsonl_path),
                "bytes": jsonl_path.stat().st_size,
                "rows": rows_out,
            },
            "llamafactory_dataset_info": str(info_path.relative_to(root)),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {jsonl_path} ({rows_out} rows)")
    print(f"Wrote {info_path}")
    print(f"Wrote {manifest_path}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
