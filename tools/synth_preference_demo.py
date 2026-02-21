#!/usr/bin/env python3
"""
Synthetic preference pairs for methodology / debugging (NOT a replacement for real human prefs).

Use this to sanity-check DPO wiring, loss scales, or to run controlled micro-experiments where you
want an explicit margin between chosen/rejected.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _pair(seed: int, i: int) -> dict[str, str]:
    rng = random.Random(seed ^ i)
    topic = rng.choice(["algebra", "cooking", "git", "health"])
    instruction = f"Give a concise, correct answer about {topic} (question #{i})."
    chosen = (
        "Answer: I'll give step-by-step reasoning, then the final result.\n"
        "1) Identify knowns.\n2) Apply the right rule.\n3) Double-check units.\n"
        "Final: [correct, specific solution]."
    )
    rejected = (
        "Answer: I can't help with that.\n"
        "(This response is intentionally evasive and does not address the question.)"
    )
    return {"instruction": instruction, "input": "", "chosen": chosen, "rejected": rejected}


def main() -> int:
    p = argparse.ArgumentParser(description="Write synthetic DPO-style JSONL for controlled tests")
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()
    root = args.root or Path(__file__).resolve().parents[1]
    out_dir = root / "artifacts" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dpo_synth_demo.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(_pair(args.seed, i), ensure_ascii=False) + "\n")

    info = {
        "repro_dpo_synth": {
            "file_name": "dpo_synth_demo.jsonl",
            "formatting": "alpaca",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    }
    info_path = out_dir / "dataset_info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {path} ({args.n} rows)")
    print(f"Wrote {info_path}")
    print("Register this bundle by pointing LlamaFactory dataset_dir at artifacts/synthetic", file=sys.stderr)
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
