#!/usr/bin/env python3
"""Turn lm-evaluation-harness JSON output into a small Markdown table."""

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


def _format_metric(v: Any) -> str:
    if _is_scalar_number(v):
        x = float(v)
        if x.is_integer():
            return str(int(x))
        return f"{x:.4f}"
    return str(v)


def _pick_metric(metrics: dict[str, Any]) -> tuple[str, Any] | None:
    preferred_keys = [
        "acc,none",
        "acc_norm,none",
        "exact_match,strict-match",
        "acc",
    ]
    for k in preferred_keys:
        if k in metrics:
            return k, metrics[k]
    for k, v in metrics.items():
        if _is_scalar_number(v):
            return k, v
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize lm_eval JSON into SUMMARY.md")
    p.add_argument("results_json", type=Path, help="Path written by lm_eval --output_path")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Markdown path (default: <results_dir>/SUMMARY.md)",
    )
    args = p.parse_args()

    path = args.results_json
    if not path.is_file():
        print(f"Results file not found: {path}", file=sys.stderr)
        return 1

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    model_args = cfg.get("model_args_string") or cfg.get("model_args")

    results = payload.get("results") or {}
    rows: list[tuple[str, str, str]] = []
    for task in sorted(results.keys()):
        metrics = results[task]
        if not isinstance(metrics, dict):
            continue
        picked = _pick_metric(metrics)
        if picked is None:
            continue
        rows.append((task, picked[0], _format_metric(picked[1])))

    out = args.out if args.out is not None else path.parent / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation summary",
        "",
        f"- **Source:** `{path}`",
    ]
    if model_args:
        lines.append(f"- **Model args:** `{model_args}`")
    lines.extend(
        [
            "",
            "| Task | Metric | Value |",
            "|------|--------|-------|",
        ]
    )
    for task, mname, val in rows:
        lines.append(f"| {task} | `{mname}` | {val} |")
    if not rows:
        lines.extend(["", "_No scalar metrics found in results payload._", ""])
    else:
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
