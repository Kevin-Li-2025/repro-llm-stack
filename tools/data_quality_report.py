#!/usr/bin/env python3
"""Quantitative data QA for exported SFT / DPO JSONL (complements manifests)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _percentiles(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def pct(p: float) -> float:
        if n == 1:
            return float(xs_sorted[0])
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return float(xs_sorted[int(k)])
        return float(xs_sorted[f] + (xs_sorted[c] - xs_sorted[f]) * (k - f))

    return {"p50": pct(0.5), "p90": pct(0.9), "p95": pct(0.95), "p99": pct(0.99)}


def _norm_key(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return t


def _duplicate_fraction(keys: list[str]) -> tuple[float, int]:
    if not keys:
        return 0.0, 0
    unique = len(set(keys))
    return 1.0 - (unique / len(keys)), unique


def _trigram_repetition_score(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < 3:
        return 0.0
    tri = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not tri:
        return 0.0
    c = Counter(tri)
    repeated = sum(1 for t in tri if c[t] > 1)
    return repeated / len(tri)


def _sft_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lens_out: list[float] = []
    lens_all: list[float] = []
    keys: list[str] = []
    rep_scores: list[float] = []
    for r in rows:
        ins = str(r.get("instruction") or "")
        inp = str(r.get("input") or "")
        out = str(r.get("output") or "")
        lens_out.append(float(len(out)))
        lens_all.append(float(len(ins) + len(inp) + len(out)))
        keys.append(hashlib.sha256(_norm_key(ins + "\n" + inp + "\n" + out).encode()).hexdigest())
        rep_scores.append(_trigram_repetition_score(out))
    dup_frac, uniq = _duplicate_fraction(keys)
    return {
        "rows": len(rows),
        "output_char_len": _percentiles(lens_out) | {"mean": float(statistics.mean(lens_out)) if lens_out else 0.0},
        "total_char_len": _percentiles(lens_all) | {"mean": float(statistics.mean(lens_all)) if lens_all else 0.0},
        "approx_duplicate_fraction_sha256_norm_text": dup_frac,
        "approx_unique_norm_hashes": uniq,
        "output_trigram_repetition_mean": float(statistics.mean(rep_scores)) if rep_scores else 0.0,
    }


def _dpo_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys: list[str] = []
    clen: list[float] = []
    rlen: list[float] = []
    diff: list[float] = []
    chosen_longer = 0
    prefix_hits = 0
    for r in rows:
        p = str(r.get("instruction") or "")
        c = str(r.get("chosen") or "")
        rej = str(r.get("rejected") or "")
        keys.append(hashlib.sha256(_norm_key(p + "\n" + c + "\n" + rej).encode()).hexdigest())
        clen.append(float(len(c)))
        rlen.append(float(len(rej)))
        diff.append(len(c) - len(rej))
        if len(c) > len(rej):
            chosen_longer += 1
        lp = min(len(c), len(rej))
        if lp > 0 and c[:lp] == rej[:lp]:
            prefix_hits += 1
    dup_frac, uniq = _duplicate_fraction(keys)
    return {
        "rows": len(rows),
        "chosen_char_len": _percentiles(clen) | {"mean": float(statistics.mean(clen)) if clen else 0.0},
        "rejected_char_len": _percentiles(rlen) | {"mean": float(statistics.mean(rlen)) if rlen else 0.0},
        "len_diff_chosen_minus_rejected": _percentiles([float(x) for x in diff])
        | {"mean": float(statistics.mean(diff)) if diff else 0.0},
        "fraction_chosen_longer_than_rejected": (chosen_longer / len(rows)) if rows else 0.0,
        "fraction_shared_prefix_all_chars_of_shorter": (prefix_hits / len(rows)) if rows else 0.0,
        "approx_duplicate_fraction_sha256_norm_text": dup_frac,
        "approx_unique_norm_hashes": uniq,
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Data quality report",
        "",
        f"- generated: `{payload.get('created_at')}`",
        f"- sft_path: `{payload.get('paths', {}).get('sft')}`",
        f"- dpo_path: `{payload.get('paths', {}).get('dpo')}`",
        "",
        "## SFT",
        "",
        "```json",
        json.dumps(payload.get("sft") or {}, indent=2),
        "```",
        "",
        "## DPO / preferences",
        "",
        "```json",
        json.dumps(payload.get("dpo") or {}, indent=2),
        "```",
        "",
        "## How to read this",
        "",
        "- **approx_duplicate_fraction** uses SHA-256 of whitespace-normalized, lowercased full example text. "
        "It catches exact near-duplicates after trivial normalization, not semantic dedup.",
        "- **trigram_repetition** on SFT `output` is a cheap proxy for template loops / copy-paste artifacts.",
        "- **shared_prefix** on DPO measures how often chosen/rejected share the same beginning up to the length "
        "of the shorter string — values near 1.0 can indicate pairs that only diverge late (often desirable for "
        "DPO), while very low values may indicate formatting mismatches.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="Compute data quality stats for artifacts/data/*.jsonl")
    p.add_argument("--root", type=Path, default=None)
    args = p.parse_args()
    root = args.root or Path(__file__).resolve().parents[1]
    sft_path = root / "artifacts" / "data" / "sft.jsonl"
    dpo_path = root / "artifacts" / "data" / "dpo.jsonl"
    if not sft_path.is_file() or not dpo_path.is_file():
        print("Missing artifacts/data/sft.jsonl or dpo.jsonl — run prepare first.", file=sys.stderr)
        return 1

    sft_rows = _read_jsonl(sft_path)
    dpo_rows = _read_jsonl(dpo_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "sft": str(sft_path.relative_to(root)),
            "dpo": str(dpo_path.relative_to(root)),
        },
        "sft": _sft_report(sft_rows),
        "dpo": _dpo_report(dpo_rows),
    }

    out_dir = root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data_quality_report.json"
    md_path = out_dir / "data_quality_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_render_md(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
