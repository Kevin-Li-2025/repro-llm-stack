#!/usr/bin/env python3
"""Turn lm-evaluation-harness JSON output into a small Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_json", type=Path, help="lm_eval --output_path JSON")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Default: artifacts/eval/SUMMARY.md next to results or cwd artifacts/eval",
    )
    args = p.parse_args()

    path = args.results_json
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    results = payload.get("results") or {}
    rows: list[tuple[str, str]] = []
    for task in sorted(results.keys()):
        metrics = results[task]
        if not isinstance(metrics, dict):
            continue
        # Prefer common scalar metrics if present
        preferred_keys = [
            "acc,none",
            "acc_norm,none",
            "exact_match,strict-match",
            "acc",
        ]
        picked = None
        for k in preferred_keys:
            if k in metrics:
                picked = (k, metrics[k])
                break
        if picked is None:
            # fall back: first float-like value
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    picked = (k, v)
                    break
        if picked is None:
            continue
        rows.append((task, f"{picked[1]:.4f}" if isinstance(picked[1], float) else str(picked[1])))

    out = args.out
    if out is None:
        out = path.parent / "SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Eval summary",
        "",
        f"- source: `{path}`",
        "",
        "| task | metric |",
        "|------|--------|",
    ]
    for task, val in rows:
        lines.append(f"| {task} | {val} |")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
