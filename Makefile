.PHONY: help dry-run prepare quality findings synth-demo experiments-render prepare-cpt-smoke sft dpo cpt eval lint check

help:
	@echo "repro-llm-stack — common targets"
	@echo "  make dry-run             — print pipeline plan (JSON)"
	@echo "  make prepare             — SFT/DPO JSONL + manifest"
	@echo "  make quality             — data QA report (needs prepare first)"
	@echo "  make findings            — MEASURED_FINDINGS.md + docs/figures/data_qa_overview.svg"
	@echo "  make synth-demo          — synthetic DPO JSONL for controlled tests"
	@echo "  make experiments-render  — regenerate docs/ABLATION_REGISTRY.md"
	@echo "  make prepare-cpt-smoke   — tiny CPT corpus"
	@echo "  make sft / dpo / cpt     — LlamaFactory train wrappers"
	@echo "  make eval                — lm_eval + SUMMARY.md"
	@echo "  make lint / check        — ruff + dry-run --strict"

dry-run:
	python3 tools/dry_run.py --recipe recipes/default.yaml --strict

prepare:
	./scripts/data/prepare.sh

quality:
	./scripts/data/quality_report.sh

findings:
	python3 tools/render_measured_findings.py --root .
	python3 tools/plot_qa_figure.py --root .

synth-demo:
	python3 tools/synth_preference_demo.py --root .

experiments-render:
	python3 tools/experiments_render.py --root .

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
