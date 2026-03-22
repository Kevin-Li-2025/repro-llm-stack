# RAG retrieval benchmark (public BEIR)

This repository compares two retrieval stacks on the **BEIR** benchmark ([Thakur et al., 2021](https://arxiv.org/abs/2104.08663)) using the same bi-encoder, corpus, and qrels — only the **ranking** step changes.

## What “vanilla RAG retrieval” means here

In most RAG systems, the first stage is **dense retrieval**: embed the query and passages, score by cosine similarity, take top‑*k* chunks. That is exactly the **baseline** in our harness: one **bi-encoder** (`sentence-transformers/all-MiniLM-L6-v2`) and cosine similarity, `top_k=1000` per query.

## What “better on retrieval quality” means

The **improved** system is standard **two-stage retrieval**:

1. Same bi-encoder as the baseline (unchanged recall pool up to `top_k`).
2. **Cross-encoder reranking** on the top **`rerank_pool`** (default 100) candidates: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. **Score merge**: cross-encoder scores get a large offset so reranked documents always outrank the remaining bi-encoder tail, preserving deep candidates for Recall@100 / @1000.

This targets the same axis you asked for earlier: **higher precision at the top of the ranked list** (typically **NDCG@10**, **MAP@10**) without claiming a new foundation model.

## Metrics

BEIR evaluation uses `pytrec_eval` via `beir.retrieval.evaluation.EvaluateRetrieval`. We report at least **NDCG@10**; the generated table also includes **MAP@10** and **Recall@10**.

## How to reproduce

```bash
cd repro-llm-stack
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[retrieval]"
```

Full comparison (default: `nfcorpus` + `scifact`, all qrels queries — **long**):

```bash
repro-beir-retrieval-compare --root .
```

Smaller smoke (faster; higher variance):

```bash
repro-beir-retrieval-compare --root . --datasets nfcorpus --max-queries 64 --seed 0
```

Outputs:

- **JSON** (full metrics): `artifacts/retrieval/beir_compare_<UTC>.json`
- **Summary table**: `docs/BENCHMARK_RETRIEVAL_BEIR.md` (overwrite)

By default the tool **exits with code 2** if cross-encoder reranking does not improve **NDCG@10** on any selected dataset (guards empty claims in CI or release scripts). Use `--allow-regression` when experimenting with other model pairs.

## Honest limitations

- This measures **retrieval**, not end-to-end RAG answer quality (no LLM generation in the loop).
- Stronger bi-encoders or hybrid sparse+dense can change the margin; the point is a **controlled** A/B on a **public** leaderboard-style benchmark.
- Subsampling queries with `--max-queries` is useful for dev; paper-grade numbers should use the **full** qrels split.

## Relation to `lm-eval` in this repo

`scripts/eval/benchmarks.sh` remains the **language model** regression harness. Retrieval benchmarks are **orthogonal** and live under this document plus `tools/beir_retrieval_compare.py`.
