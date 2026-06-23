# 04 - Contrastive Training and LoRA

## What it is
Training updates embedding behavior with adapter-based fine-tuning (LoRA) using contrastive objectives.

## Why it is used
LoRA reduces memory footprint and update cost while preserving a reusable base model. Contrastive objectives align query vectors with relevant documents and separate hard negatives.

## How it appears in code
- PEFT-style trainer: `src/domain_adapted_embedding_alignment/training/trainer.py`
- Unsloth path with SentenceTransformers trainer: `src/domain_adapted_embedding_alignment/training/unsloth_trainer.py`
- Training entrypoint: `src/domain_adapted_embedding_alignment/pipelines/train_model.py`
- Adapter-aware embedding load path: `src/domain_adapted_embedding_alignment/retrieval/backend_factory.py`

## Practical explanation from run outputs
From `artifacts/reports/training_report.json`:
- `total_steps`: 700
- `train_examples`: 159698
- `val_examples`: 2000
- `best_val_loss`: 0.06946182250976562
- `runtime.backend_used`: unsloth

From `artifacts/logs/01_train.log`:
- Unsloth initialized on RTX 4060 Laptop GPU
- `task_type` set to `FEATURE_EXTRACTION`

## Beginner checkpoint
- Understand what an adapter is.
- Understand why train/validation are both required.

## Advanced checkpoint
- Explain the tradeoff between eval frequency and runtime.
- Explain how `MultipleNegativesRankingLoss` interacts with hard negatives in batch construction.

## Common pitfalls
- Missing or incompatible adapter path breaks tuned backend loading.
- Confusing base-model performance with adapter performance if adapter load silently fails.

