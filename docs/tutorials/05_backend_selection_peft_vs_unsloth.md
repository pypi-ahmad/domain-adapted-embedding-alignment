# 05 - Backend Selection: PEFT vs Unsloth

## What it is
A backend resolution system that decides whether training runs with `peft` or `unsloth`.

## Why it is used
Different hosts have different capabilities (CUDA, installed packages). Backend routing keeps behavior explicit and reproducible, including fallback semantics.

## How it appears in code
- Resolver: `src/domain_adapted_embedding_alignment/training/backend_selection.py`
- Runtime capability checks: `unsloth_runtime_status()`
- Strict mode behavior: `resolve_training_backend(..., strict_backend=True)`
- Trainer execution entry: `src/domain_adapted_embedding_alignment/pipelines/train_model.py`

## Practical explanation from run outputs
From `artifacts/reports/training_report.json` runtime metadata:
- `backend_requested`: `unsloth`
- `backend_used`: `unsloth`
- `fallback_reason`: `null`
- `unsloth_available`: `true`

This confirms no fallback occurred in the completed run.

## Beginner checkpoint
- Understand `auto` vs explicit backend selection.
- Understand what strict backend mode is protecting.

## Advanced checkpoint
- Design a deployment policy for mixed environments (GPU and CPU hosts).
- Define alerting rules when requested backend and used backend diverge.

## Verification hooks
- Unit tests: `tests/test_backend_selection.py`
- Runtime report: `artifacts/reports/training_report.json`

