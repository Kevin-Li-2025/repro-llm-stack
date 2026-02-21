.PHONY: help dry-run prepare sft dpo eval lint check

help:
	@echo "repro-llm-stack — common targets"
	@echo "  make dry-run    — print pipeline plan (JSON)"
	@echo "  make prepare    — export JSONL + manifest (needs: pip install -e \".[data]\")"
	@echo "  make sft / dpo  — LlamaFactory train wrappers"
	@echo "  make eval       — lm_eval + SUMMARY.md (needs: pip install -e \".[eval]\")"
	@echo "  make lint       — ruff check tools/"
	@echo "  make check      — dry-run --strict + lint"

dry-run:
	python3 tools/dry_run.py --recipe recipes/default.yaml --strict

prepare:
	./scripts/data/prepare.sh

sft:
	./scripts/train/sft.sh

dpo:
	./scripts/train/dpo.sh

eval:
	./scripts/eval/benchmarks.sh

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check tools/ || python3 -m ruff check tools/

check: lint dry-run
