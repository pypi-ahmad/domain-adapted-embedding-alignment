# 02 - Data Preparation and Schema

## What it is
Data preparation loads multiple real datasets and normalizes them into a shared schema:
- `documents.parquet`
- `queries.parquet`
- `pairs.parquet`

## Why it is used
Training and evaluation stages depend on consistent fields and deterministic generation. Without schema stability, downstream metrics and comparisons become unreliable.

## How it appears in code
- Pipeline stage: `src/domain_adapted_embedding_alignment/pipelines/prepare_data.py`
- Dataset-specific loaders: `src/domain_adapted_embedding_alignment/data/loaders.py`
  - `load_msmarco`
  - `load_medical_medmentions`
  - `load_legal_lexglue`
  - `load_cyber_mitre`
- Record contracts: `src/domain_adapted_embedding_alignment/schemas.py`

## Practical explanation from run outputs
From `artifacts/reports/data_preparation_report.json`:
- documents_total: `168448`
- queries_total: `98907`
- pairs_total: `200000`
- document mix:
  - general: `153637`
  - medical: `4392`
  - legal: `7999`
  - cybersecurity: `2420`

These values are the canonical evidence for this run's training and benchmark scope.

## Beginner checkpoint
- Learn which file stores documents vs queries vs pairs.
- Learn where domain labels live and why they matter for analysis.

## Advanced checkpoint
- Explain how changing `DEA_MSMARCO_MAX_ROWS` or `DEA_FINAL_PAIR_TARGET` changes model behavior and evaluation confidence.
- Identify where data lineage is persisted (`data_preparation_report.json`).

## Common pitfalls
- Missing MITRE CTI source breaks cyber ingestion.
- Inconsistent environment overrides can silently shift dataset composition.

