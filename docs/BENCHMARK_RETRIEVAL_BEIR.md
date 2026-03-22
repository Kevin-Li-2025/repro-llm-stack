<!-- Regenerate: `pip install -e ".[retrieval]" && repro-beir-retrieval-compare --root .` (see docs/RAG_RETRIEVAL_BENCHMARK.md) -->

## BEIR retrieval: dense-only vs dense + cross-encoder rerank

Pinned run: **NFCorpus** `test`, **64 queries** (seed **0**, `--max-queries 64`), models `sentence-transformers/all-MiniLM-L6-v2` + `cross-encoder/ms-marco-MiniLM-L-6-v2`. JSON: `artifacts/retrieval/beir_compare_20260322T152417Z.json` (local only; path recorded for provenance).

Primary readout (ranking quality at the cut RAG usually consumes): **NDCG@10**. Secondary: MAP@10, Recall@10.

| Dataset | Queries | NDCG@10 | MAP@10 | R@10 | NDCG@10 Δ |
|---------|--------:|--------:|-------:|-----:|----------:|
| nfcorpus | 64 | 0.3309 → 0.3635 | 0.1249 → 0.1412 | 0.1627 → 0.1667 | +0.0326 |

For **full** official query counts, omit `--max-queries` (slow; downloads + encodes full corpus per dataset).
