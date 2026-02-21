#!/usr/bin/env python3
"""Render experiments/registry.yaml into a reviewer-friendly Markdown grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=Path("experiments/registry.yaml"))
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("docs/ABLATION_REGISTRY.md"))
    args = p.parse_args()

    root = args.root or Path(__file__).resolve().parents[1]
    reg_path = args.registry if args.registry.is_absolute() else root / args.registry
    if not reg_path.is_file():
        print(f"Missing {reg_path}", file=sys.stderr)
        return 1

    with reg_path.open("r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    exps = reg.get("experiments") or []

    lines = [
        "# Ablation registry",
        "",
        "This file is **generated** from `experiments/registry.yaml`. "
        "Edit the YAML, then run `python tools/experiments_render.py`.",
        "",
        "| ID | Stage | What changes | Notes |",
        "|----|-------|--------------|-------|",
    ]
    for e in exps:
        eid = e.get("id", "")
        stage = e.get("train_stage", "")
        desc = str(e.get("description", "")).replace("|", "\\|")
        toggles = e.get("toggles")
        notes = str(e.get("notes", "")).replace("|", "\\|")
        toggle_s = ""
        if isinstance(toggles, dict):
            toggle_s = ", ".join(f"{k}={v}" for k, v in toggles.items())
        change = desc if not toggle_s else f"{desc} ({toggle_s})"
        lines.append(f"| {eid} | {stage} | {change} | {notes} |")

    lines.extend(
        [
            "",
            "## How to use",
            "",
            "1. Run training + eval for each row you care about, saving `lm_eval` JSON under `results/eval_runs/`.",
            "2. `python tools/compare_eval_runs.py --run ... --out docs/BENCHMARK_TABLE.md`",
            "3. Paste deltas into your write-up; keep the JSON for auditability.",
            "",
        ]
    )

    out = args.out if args.out.is_absolute() else root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
