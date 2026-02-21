#!/usr/bin/env python3
"""Write a simple SVG bar chart from artifacts/data_quality_report.json (no extra deps)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=None)
    args = p.parse_args()
    root = args.root or Path(__file__).resolve().parents[1]
    qa_path = root / "artifacts" / "data_quality_report.json"
    out_dir = root / "docs" / "figures"
    out_path = out_dir / "data_qa_overview.svg"

    if not qa_path.is_file():
        print(f"Missing {qa_path}", file=sys.stderr)
        return 1

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    sft = qa.get("sft") or {}
    dpo = qa.get("dpo") or {}

    sft_mean = float(sft.get("output_char_len", {}).get("mean") or 0)
    ch_mean = float(dpo.get("chosen_char_len", {}).get("mean") or 0)
    rej_mean = float(dpo.get("rejected_char_len", {}).get("mean") or 0)
    frac = float(dpo.get("fraction_chosen_longer_than_rejected") or 0)

    labels = ["SFT output μ", "DPO chosen μ", "DPO rejected μ"]
    values = [sft_mean, ch_mean, rej_mean]
    vmax = max(values) * 1.12 if values else 100

    w, h = 640, 300
    margin_l, margin_b = 72, 56
    chart_w = w - margin_l - 24
    bw = chart_w / len(values)
    bars: list[str] = []
    for i, v in enumerate(values):
        x = margin_l + i * bw + 10
        bh = (h - margin_b - 48) * (v / vmax) if vmax else 0
        y = h - margin_b - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 20:.1f}" height="{max(bh, 1):.1f}" fill="#2563eb" rx="4"/>'
        )
        bars.append(
            f'<text x="{x + (bw - 20) / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif">{v:.0f}</text>'
        )
        bars.append(
            f'<text x="{x + (bw - 20) / 2:.1f}" y="{h - margin_b + 18}" text-anchor="middle" font-size="10" font-family="system-ui,sans-serif">{labels[i]}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="100%" height="100%" fill="#fafafa"/>
  <text x="{margin_l}" y="28" font-size="15" font-weight="600" font-family="system-ui,sans-serif">Exported data — mean character lengths</text>
  <text x="{margin_l}" y="46" font-size="11" fill="#444" font-family="system-ui,sans-serif">Source: artifacts/data_quality_report.json</text>
  {"".join(bars)}
  <text x="{margin_l}" y="{h - 8}" font-size="11" font-family="system-ui,sans-serif">DPO: P(chosen longer than rejected) = {frac * 100:.1f}%</text>
</svg>
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
