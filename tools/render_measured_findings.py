#!/usr/bin/env python3
"""Generate docs/MEASURED_FINDINGS.md from artifacts/data_quality_report.json (+ optional manifest)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _write_snapshot(
    root: Path,
    qa: dict[str, Any],
    manifest: dict[str, Any],
    sft: dict[str, Any],
    dpo: dict[str, Any],
) -> Path:
    env = manifest.get("environment") if isinstance(manifest.get("environment"), dict) else {}
    rid = manifest.get("recipe_id")
    snap: dict[str, Any] = {
        "schema_version": 1,
        "description": "Aggregated data-plane metrics only (no raw examples). Committed alongside MEASURED_FINDINGS.md.",
        "created_at": qa.get("created_at"),
        "export": {
            "recipe_id": rid,
            "git_commit_at_export": env.get("git_commit"),
            "sft_jsonl": "artifacts/data/sft.jsonl",
            "dpo_jsonl": "artifacts/data/dpo.jsonl",
            "sft_rows": sft.get("rows"),
            "dpo_rows": dpo.get("rows"),
        },
        "sft": sft,
        "dpo": dpo,
    }
    out_dir = root / "docs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_plane_snapshot.json"
    out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=None)
    args = p.parse_args()
    root = args.root or Path(__file__).resolve().parents[1]
    qa_path = root / "artifacts" / "data_quality_report.json"
    man_path = root / "artifacts" / "data_manifest.json"
    out_path = root / "docs" / "MEASURED_FINDINGS.md"

    if not qa_path.is_file():
        print(f"Missing {qa_path}; run prepare + data_quality_report first.", file=sys.stderr)
        return 1

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}

    sft = qa.get("sft") or {}
    dpo = qa.get("dpo") or {}
    rid = manifest.get("recipe_id", "—")
    git = (manifest.get("environment") or {}).get("git_commit", "—")

    snap_path = _write_snapshot(root, qa, manifest, sft, dpo)

    o = sft.get("output_char_len") or {}
    ch = dpo.get("chosen_char_len") or {}
    rj = dpo.get("rejected_char_len") or {}
    dlen = dpo.get("len_diff_chosen_minus_rejected") or {}

    lines = [
        "# Measured findings (data plane)",
        "",
        "Quantitative summary of the **exported** SFT and DPO JSONL used by this repository (after recipe filters). "
        "These metrics support data-governance claims; they do not replace model-plane benchmarks.",
        "",
        "## Summary",
        "",
        "| Quantity | SFT (`repro_sft`) | DPO (`repro_dpo`) |",
        "|----------|-------------------|-------------------|",
        f"| Rows | {sft.get('rows')} | {dpo.get('rows')} |",
        f"| Primary text length (chars), mean | output **{o.get('mean', 0):.1f}** | chosen **{ch.get('mean', 0):.1f}** · rejected **{rj.get('mean', 0):.1f}** |",
        f"| Primary text length, p50 / p95 | **{o.get('p50', 0):.1f}** / **{o.get('p95', 0):.1f}** | Δ(chosen−rejected) p50 **{dlen.get('p50', 0):.1f}** |",
        f"| Approx. duplicate fraction (normalized SHA-256) | **{sft.get('approx_duplicate_fraction_sha256_norm_text', 0):.4f}** | **{dpo.get('approx_duplicate_fraction_sha256_norm_text', 0):.4f}** |",
        f"| Other | trigram repetition (output), mean **{sft.get('output_trigram_repetition_mean', 0):.4f}** | "
        f"P(chosen longer) **{100 * float(dpo.get('fraction_chosen_longer_than_rejected') or 0):.1f}%** · "
        f"shared-prefix **{100 * float(dpo.get('fraction_shared_prefix_all_chars_of_shorter') or 0):.2f}%** |",
        "",
        "## Machine-readable data",
        "",
        "- **JSON snapshot:** [`docs/metrics/data_plane_snapshot.json`](metrics/data_plane_snapshot.json) (same numbers as below; safe to cite programmatically).",
        "- **Figure:** [`docs/figures/data_qa_overview.svg`](figures/data_qa_overview.svg)",
        "",
        "## Regeneration",
        "",
        "This file is produced from `artifacts/data_quality_report.json` after `prepare` and `data_quality_report`.",
        "",
        "```bash",
        "python tools/prepare_data.py --root . --max-sft 2000 --max-dpo 2000   # or full export",
        "python tools/data_quality_report.py --root .",
        "python tools/render_measured_findings.py --root .",
        "python tools/plot_qa_figure.py --root .",
        "```",
        "",
        "Makefile: `make prepare && make quality && make findings`",
        "",
        "## Scope and limitations",
        "",
        "- Statistics describe **exported JSONL**, not complete Hugging Face dataset tables.",
        "- With row caps, examples are the **first** passing rows in dataset order (not an i.i.d. subsample unless you change export logic).",
        f"- **recipe_id:** `{rid}`",
        f"- **git_commit** (recorded in manifest at export): `{git}`",
        "",
        "## SFT export (`repro_sft`)",
        "",
        f"- **rows:** {sft.get('rows')}",
        f"- **mean output chars:** {sft.get('output_char_len', {}).get('mean', 0):.2f}",
        f"- **p50 / p95 output chars:** {sft.get('output_char_len', {}).get('p50', 0):.1f} / {sft.get('output_char_len', {}).get('p95', 0):.1f}",
        f"- **approx. duplicate fraction (normalized hash):** {sft.get('approx_duplicate_fraction_sha256_norm_text', 0):.4f}",
        f"- **mean trigram repetition score (outputs):** {sft.get('output_trigram_repetition_mean', 0):.4f}",
        "",
        "### Interpretation",
        "",
        "- Near-zero **duplicate fraction** under this hash indicates diverse superficial forms after normalization (not semantic deduplication).",
        "- **Trigram repetition** flags templated or repetitive `output` text; investigate if the mean is unusually high for the domain.",
        "",
        "## DPO export (`repro_dpo`, Ultrafeedback → flat pairs)",
        "",
        f"- **rows:** {dpo.get('rows')}",
        f"- **mean chosen / rejected chars:** {dpo.get('chosen_char_len', {}).get('mean', 0):.2f} / {dpo.get('rejected_char_len', {}).get('mean', 0):.2f}",
        f"- **p50 length delta (chosen − rejected):** {dpo.get('len_diff_chosen_minus_rejected', {}).get('p50', 0):.1f}",
        f"- **fraction chosen longer than rejected:** {dpo.get('fraction_chosen_longer_than_rejected', 0):.3f}",
        f"- **fraction with full shared prefix (up to shorter length):** {dpo.get('fraction_shared_prefix_all_chars_of_shorter', 0):.4f}",
        f"- **approx. duplicate fraction:** {dpo.get('approx_duplicate_fraction_sha256_norm_text', 0):.4f}",
        "",
        "### Interpretation",
        "",
        "- **Chosen longer than rejected** above ~0.5 is common in helpfulness-style preferences but is **not** proof of quality; length can track hedging or verbosity.",
        "- **Shared-prefix rate** near zero here is consistent with pairs that diverge early (e.g. different assistant continuations after the same user prompt).",
        "",
        "## Provenance",
        "",
        f"- **`data_quality_report.created_at`:** `{qa.get('created_at', '—')}`",
        f"- **Snapshot written to:** `{snap_path.relative_to(root)}`",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {snap_path}")
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
