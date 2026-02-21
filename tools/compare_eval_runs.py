#!/usr/bin/env python3
"""Merge multiple lm_eval JSON outputs into one Markdown comparison table."""

from __future__ import annotations

import argparse
import json
import math
import numbers
import sys
from pathlib import Path
from typing import Any


def _is_scalar_number(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return False
    if isinstance(v, numbers.Real):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return False
        return True
    return False


def _pick_metric(metrics: dict[str, Any]) -> tuple[str, float] | None:
    preferred = ["acc,none", "acc_norm,none", "exact_match,strict-match", "acc"]
    for k in preferred:
        if k in metrics and _is_scalar_number(metrics[k]):
            return k, float(metrics[k])
    for k, v in metrics.items():
        if _is_scalar_number(v):
            return k, float(v)
    return None


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    raw = payload.get("results") or {}
    out: dict[str, dict[str, Any]] = {}
    for task, metrics in raw.items():
        if not isinstance(metrics, dict):
            continue
        picked = _pick_metric(metrics)
        if picked is None:
            continue
        out[task] = {"metric": picked[0], "value": picked[1]}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compare lm_eval JSON runs (same tasks)")
    p.add_argument(
        "--run",
        action="append",
        metavar="NAME=PATH",
        required=True,
        help="Repeatable. Example: --run baseline=artifacts/eval/base.json",
    )
    p.add_argument("--out", type=Path, default=None, help="Markdown output path (default: stdout)")
    args = p.parse_args()

    runs: list[tuple[str, Path]] = []
    for item in args.run:
        if "=" not in item:
            print(f"Invalid --run (expected NAME=PATH): {item}", file=sys.stderr)
            return 1
        name, path_s = item.split("=", 1)
        name = name.strip()
        path = Path(path_s.strip())
        if not name or not path.is_file():
            print(f"Missing file for run '{name}': {path}", file=sys.stderr)
            return 1
        runs.append((name, path))

    per_run: dict[str, dict[str, dict[str, Any]]] = {}
    all_tasks: set[str] = set()
    for name, path in runs:
        res = _load_results(path)
        per_run[name] = res
        all_tasks.update(res.keys())

    if not all_tasks:
        print("No tasks found in inputs.", file=sys.stderr)
        return 2

    tasks_sorted = sorted(all_tasks)
    header = ["task"] + [name for name, _ in runs] + ["delta_last_minus_first"]
    lines = [
        "# Evaluation comparison",
        "",
        "Metrics are chosen per task in priority order: `acc,none`, `acc_norm,none`, "
        "`exact_match,strict-match`, then first numeric field.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]

    first_name = runs[0][0]
    last_name = runs[-1][0]
    for task in tasks_sorted:
        row = [task]
        vals: list[float | None] = []
        for name, _ in runs:
            cell = per_run[name].get(task)
            if cell is None:
                row.append("—")
                vals.append(None)
            else:
                v = cell["value"]
                row.append(f"{v:.4f} (`{cell['metric']}`)")
                vals.append(v)
        delta = "—"
        if vals[0] is not None and vals[-1] is not None:
            delta = f"{vals[-1] - vals[0]:+.4f}"
        row.append(delta)
        lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            f"_Delta column: `{last_name}` minus `{first_name}` (adjust ordering with `--run`)._",
            "",
        ]
    )

    text = "\n".join(lines) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
