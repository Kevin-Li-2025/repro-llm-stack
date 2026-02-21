.PHONY: help dry-run prepare prepare-cpt-smoke sft dpo cpt eval lint check

help:
	@echo "repro-llm-stack — common targets"
	@echo "  make dry-run           — print pipeline plan (JSON)"
	@echo "  make prepare           — SFT/DPO JSONL + manifest (pip install -e \".[data]\")"
	@echo "  make prepare-cpt-smoke — tiny CPT corpus for LlamaFactory pt smoke tests"
	@echo "  make sft / dpo / cpt   — LlamaFactory train wrappers (cpt needs prepare-cpt-smoke)"
	@echo "  make eval              — lm_eval + SUMMARY.md (pip install -e \".[eval]\")"
	@echo "  make lint              — ruff check tools/"
	@echo "  make check             — dry-run --strict + lint"

dry-run:
	python3 tools/dry_run.py --recipe recipes/default.yaml --strict

prepare:
	./scripts/data/prepare.sh

prepare-cpt-smoke:
	./scripts/data/prepare_cpt_smoke.sh

sft:
	./scripts/train/sft.sh

dpo:
	./scripts/train/dpo.sh

cpt:
	./scripts/train/cpt.sh

eval:
	./scripts/eval/benchmarks.sh

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check tools/ || python3 -m ruff check tools/

check: lint dry-run
