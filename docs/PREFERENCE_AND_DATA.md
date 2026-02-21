# Preference data construction (DPO track)

This repository exports **Ultrafeedback (binarized)** preference rows into LlamaFactory’s **alpaca-style ranking** JSONL. This page documents the **exact mapping**, what information is **lost**, and how to extend or replace it.

## Source

- **Hub:** `HuggingfaceH4/ultrafeedback_binarized`
- **Split:** `train_prefs` (not `train`, which does not exist for this dataset)
- **Schema (typical row):** `prompt` (string), `chosen` / `rejected` (chat message lists with `role` / `content`)

## Transformation (`tools/prepare_data.py`)

1. **Prompt** — `instruction := prompt.strip()`
2. **Chosen / rejected text** — walk the message list from the end; take the **last** message with `role == "assistant"`; stringify `content` (supports string or multimodal-style list chunks).
3. **Filters**
   - Drop rows with empty prompt or empty chosen/rejected.
   - Drop rows where **chosen text equals rejected text** (degenerate pair).
   - Apply global `filters.min_chars` / `max_chars` on the concatenation of `instruction`, `chosen`, and `rejected`.

## What this buys you / what it costs

| Benefit | Cost |
|--------|------|
| Stable JSONL + hashes for reproducibility | Loses full multi-turn context beyond the final assistant turn |
| Simple LlamaFactory `ranking: true` format | Does not preserve original Bradley–Terry scores as training targets |
| Fast to audit row counts | Not a substitute for **domain-specific** preference collection |

## Quality signals

After `prepare`, run:

```bash
python tools/data_quality_report.py --root .
```

Key DPO fields in `artifacts/data_quality_report.json`:

- **`fraction_chosen_longer_than_rejected`** — crude prior; many (not all) helpful datasets skew longer for better answers.
- **`fraction_shared_prefix_all_chars_of_shorter`** — high values often mean the pair shares a long common prefix and diverges late (typical for “same prompt, two continuations”).
- **`approx_duplicate_fraction_*`** — catches copy/paste / normalization duplicates, not paraphrases.

## Synthetic pairs (controlled experiments)

For **method debugging** or **micro-benchmarks** where you need an explicit margin between policies:

```bash
python tools/synth_preference_demo.py --n 512
```

This writes `artifacts/synthetic/dpo_synth_demo.jsonl` plus `dataset_info.json`. **Do not** claim human-alignment results from this file — it exists to isolate **optimization / implementation** effects.

## Replacing the preference source

1. Add a new recipe file under `recipes/` pointing at your HF dataset or local files.
2. Implement a dedicated exporter (fork `prepare_data.py` or add a `tools/prepare_preferences_custom.py`) that outputs the same four columns: `instruction`, `input`, `chosen`, `rejected`.
3. Register in `artifacts/data/dataset_info.json` with `"ranking": true`.

## Why “data > model” still matters here

Even without a novel loss, **where preferences come from**, **how pairs are filtered**, and **how duplicates are controlled** often dominate DPO outcomes. This repo makes those choices **explicit and measurable** rather than hiding them inside notebook cells.
