"""
Compare dense-only retrieval (RAG-style first stage) vs dense + cross-encoder reranking on BEIR.

Baseline: single bi-encoder, cosine top-k (standard dense retrieval as used in vanilla RAG pipelines).
Improved: same bi-encoder pool (top ``rerank_pool``), reranked with a cross-encoder; tail ranks
preserved from bi-encoder scores so Recall@large remains defined.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BEIR_DATASETS = {
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
}


def _doc_passage(corpus: dict[str, dict[str, str]], doc_id: str) -> str:
    d = corpus[doc_id]
    title = (d.get("title") or "").strip()
    text = (d.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def _subsample_queries(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    max_queries: int | None,
    seed: int,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    if max_queries is None or max_queries >= len(qrels):
        qids = sorted(qrels.keys())
        return {q: queries[q] for q in qids if q in queries}, qrels
    rng = random.Random(seed)
    qids = [q for q in sorted(qrels.keys()) if q in queries]
    rng.shuffle(qids)
    picked = set(qids[:max_queries])
    new_qrels = {q: qrels[q] for q in picked}
    new_queries = {q: queries[q] for q in picked}
    return new_queries, new_qrels


def _dense_search(
    bi_model_name: str,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    first_stage_top_k: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    from beir.retrieval.evaluation import EvaluateRetrieval
    from beir.retrieval.models import SentenceBERT
    from beir.retrieval.search.dense import DenseRetrievalExactSearch

    model = SentenceBERT(bi_model_name)
    search = DenseRetrievalExactSearch(model, batch_size=batch_size, corpus_chunk_size=50000)
    # max(k_values) is the bi-encoder depth per query (vanilla dense RAG retrieval cap).
    k_values = sorted({1, 3, 5, 10, 100, max(first_stage_top_k, 10)})
    retriever = EvaluateRetrieval(search, k_values=k_values, score_function="cos_sim")
    return retriever.retrieve(corpus, queries)


def _merge_rerank(
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    bi_results: dict[str, dict[str, float]],
    cross_encoder_name: str,
    rerank_pool: int,
    ce_batch_size: int,
) -> dict[str, dict[str, float]]:
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(cross_encoder_name)
    merged: dict[str, dict[str, float]] = {}

    for qid, qtext in queries.items():
        ranked = sorted(bi_results[qid].items(), key=lambda x: x[1], reverse=True)
        head = ranked[:rerank_pool]
        tail = ranked[rerank_pool:]

        pairs = [(qtext, _doc_passage(corpus, did)) for did, _ in head]
        if not pairs:
            merged[qid] = dict(ranked)
            continue

        raw = ce.predict(pairs, batch_size=ce_batch_size, show_progress_bar=False)
        # Keep cross-encoder heads strictly above any bi-encoder tail.
        ce_part = {did: 1000.0 + float(s) for (did, _), s in zip(head, raw, strict=True)}
        tail_part = {did: float(score) for did, score in tail}
        merged[qid] = {**tail_part, **ce_part}

    return merged


def _k_values_for_eval(first_stage_top_k: int) -> list[int]:
    candidates = [1, 3, 5, 10, 100, 1000, first_stage_top_k]
    return sorted({k for k in candidates if k <= first_stage_top_k})


def _evaluate(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, Any]:
    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)
    return {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision,
    }


def _download_dataset(name: str, data_root: Path) -> Path:
    from beir import util

    if name not in BEIR_DATASETS:
        raise SystemExit(f"Unknown dataset {name!r}. Choose from: {', '.join(sorted(BEIR_DATASETS))}")
    url = BEIR_DATASETS[name]
    out = util.download_and_unzip(url, str(data_root))
    return Path(out)


def _run_one_dataset(
    dataset: str,
    data_root: Path,
    bi_model: str,
    cross_encoder: str,
    first_stage_top_k: int,
    rerank_pool: int,
    max_queries: int | None,
    seed: int,
    bi_batch_size: int,
    ce_batch_size: int,
) -> dict[str, Any]:
    from beir.datasets.data_loader import GenericDataLoader

    ds_path = _download_dataset(dataset, data_root)
    loader = GenericDataLoader(str(ds_path))
    corpus, queries, qrels = loader.load(split="test")
    queries, qrels = _subsample_queries(queries, qrels, max_queries, seed)

    logger.info(
        "Dataset %s: corpus=%d queries_eval=%d (after subsample)",
        dataset,
        len(corpus),
        len(queries),
    )

    bi_results = _dense_search(
        bi_model,
        corpus,
        queries,
        first_stage_top_k,
        bi_batch_size,
    )
    reranked = _merge_rerank(corpus, queries, bi_results, cross_encoder, rerank_pool, ce_batch_size)

    k_values = _k_values_for_eval(first_stage_top_k)
    metrics_bi = _evaluate(qrels, bi_results, k_values)
    metrics_ce = _evaluate(qrels, reranked, k_values)

    return {
        "dataset": dataset,
        "n_queries": len(queries),
        "n_corpus": len(corpus),
        "bi_encoder": bi_model,
        "cross_encoder": cross_encoder,
        "first_stage_top_k": first_stage_top_k,
        "rerank_pool": rerank_pool,
        "max_queries": max_queries,
        "seed": seed,
        "baseline_dense_only": metrics_bi,
        "dense_plus_cross_encoder_rerank": metrics_ce,
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## BEIR retrieval: dense-only vs dense + cross-encoder rerank",
        "",
        "Primary readout (ranking quality at the cut RAG usually consumes): **NDCG@10**. "
        "Secondary: MAP@10, Recall@10.",
        "",
        "| Dataset | Queries | NDCG@10 | MAP@10 | R@10 | NDCG@10 Δ |",
        "|---------|--------:|--------:|-------:|-----:|----------:|",
    ]
    for r in rows:
        d = r["dataset"]
        nq = r["n_queries"]
        b_ndcg = r["baseline_dense_only"]["ndcg"].get("NDCG@10", 0.0)
        c_ndcg = r["dense_plus_cross_encoder_rerank"]["ndcg"].get("NDCG@10", 0.0)
        b_map = r["baseline_dense_only"]["map"].get("MAP@10", 0.0)
        c_map = r["dense_plus_cross_encoder_rerank"]["map"].get("MAP@10", 0.0)
        b_r = r["baseline_dense_only"]["recall"].get("Recall@10", 0.0)
        c_r = r["dense_plus_cross_encoder_rerank"]["recall"].get("Recall@10", 0.0)
        d_ndcg = c_ndcg - b_ndcg
        lines.append(
            f"| {d} | {nq} | {b_ndcg:.4f} → {c_ndcg:.4f} | "
            f"{b_map:.4f} → {c_map:.4f} | {b_r:.4f} → {c_r:.4f} | {d_ndcg:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."), help="Repository root (for default paths)")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Download/cache BEIR zips here (default: <root>/artifacts/beir_datasets)",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["nfcorpus", "scifact"],
        choices=sorted(BEIR_DATASETS.keys()),
        help="BEIR datasets to evaluate",
    )
    p.add_argument(
        "--bi-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model id for dense retrieval",
    )
    p.add_argument(
        "--cross-encoder",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="CrossEncoder model id for reranking",
    )
    p.add_argument(
        "--first-stage-top-k",
        type=int,
        default=1000,
        help="Bi-encoder candidates kept per query before merge (BEIR standard top-k cap)",
    )
    p.add_argument(
        "--rerank-pool",
        type=int,
        default=100,
        help="Top-M from bi-encoder to score with the cross-encoder",
    )
    p.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Subsample queries for smoke tests (deterministic with --seed)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bi-batch-size", type=int, default=128)
    p.add_argument("--ce-batch-size", type=int, default=32)
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full metrics JSON (default: <root>/artifacts/retrieval/beir_compare_<stamp>.json)",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Write NDCG@10 summary table (default: <root>/docs/BENCHMARK_RETRIEVAL_BEIR.md)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--allow-regression",
        action="store_true",
        help="Do not exit with error if +CE rerank fails to beat dense NDCG@10 on any dataset",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    root = args.root.resolve()
    data_dir = (args.data_dir or (root / "artifacts" / "beir_datasets")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    stamp = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = args.out_json or (root / "artifacts" / "retrieval" / f"beir_compare_{stamp}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md = args.out_md or (root / "docs" / "BENCHMARK_RETRIEVAL_BEIR.md")

    rows: list[dict[str, Any]] = []
    for name in args.datasets:
        row = _run_one_dataset(
            name,
            data_dir,
            args.bi_encoder,
            args.cross_encoder,
            args.first_stage_top_k,
            args.rerank_pool,
            args.max_queries,
            args.seed,
            args.bi_batch_size,
            args.ce_batch_size,
        )
        rows.append(row)

    payload = {
        "protocol": {
            "baseline": "Single-stage dense retrieval (bi-encoder cosine similarity), same as typical RAG first stage.",
            "improved": (
                f"Same bi-encoder, then cross-encoder rerank of top-{args.rerank_pool} with score merge "
                "so reranked heads beat bi-encoder tail (standard two-stage retrieval)."
            ),
            "beir_paper": "https://arxiv.org/abs/2104.08663",
        },
        "runs": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out_json)

    md = _markdown_table(rows)
    header = (
        "<!-- Auto-generated by tools/beir_retrieval_compare.py — edit the tool or re-run, not by hand. -->\n\n"
    )
    out_md.write_text(header + md, encoding="utf-8")
    logger.info("Wrote %s", out_md)

    # Fail if any dataset does not improve NDCG@10 (strict success criterion for this harness).
    failed: list[str] = []
    for r in rows:
        b = r["baseline_dense_only"]["ndcg"].get("NDCG@10", 0.0)
        c = r["dense_plus_cross_encoder_rerank"]["ndcg"].get("NDCG@10", 0.0)
        if c <= b + 1e-6:
            failed.append(f"{r['dataset']}: NDCG@10 {c:.5f} <= dense {b:.5f}")

    if failed and not args.allow_regression:
        logger.error("Improvement check failed:\n  %s", "\n  ".join(failed))
        return 2
    if failed and args.allow_regression:
        logger.warning("Improvement check failed (ignored): %s", failed)
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
