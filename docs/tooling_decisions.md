# Tooling Decisions: PEFT, Unsloth, and TRL

This note explains why the runtime uses PEFT/Unsloth for embedding alignment and why TRL is documentation-only for this project version.

## 1) PEFT

### Definition
PEFT (Parameter-Efficient Fine-Tuning) updates a small subset of parameters (for example LoRA adapters) instead of full model weights.

### Why it is used here
- Embedding alignment must run under local compute constraints.
- Adapter-based tuning keeps memory and storage overhead practical.
- It enables direct baseline-vs-tuned comparison on the same base model family.

### Where it appears in code
- LoRA-centric training path: `src/domain_adapted_embedding_alignment/training/trainer.py`
- Adapter-aware loading for retrieval: `src/domain_adapted_embedding_alignment/retrieval/backend_factory.py`

### Practical effect to measure
- Retrieval deltas (`recall@k`, `MRR`, `NDCG`, `MAP`)
- Similarity ranking improvement
- Latency vs quality tradeoff relative to baseline

## 2) Unsloth

### Definition
Unsloth is an acceleration framework for efficient fine-tuning, with embedding-specific guidance centered on SentenceTransformer workflows.

### Why it is used here
- Provides an accelerated optional backend for embedding training.
- Enables strict backend routing with explicit fallback semantics.
- Preserves the same objective while improving practical runtime throughput on compatible GPUs.

### Where it appears in code
- Backend resolution rules: `src/domain_adapted_embedding_alignment/training/backend_selection.py`
- Unsloth trainer implementation: `src/domain_adapted_embedding_alignment/training/unsloth_trainer.py`
- Runtime metadata persistence: `src/domain_adapted_embedding_alignment/pipelines/train_model.py`

### Practical effect to measure
- Train runtime and throughput against PEFT path
- Stability of downstream retrieval quality and latency
- Backend request/use/fallback traceability in `training_report.json`

## 3) TRL

### Definition
TRL provides post-training trainers for language model alignment workflows (for example SFT, DPO, PPO/GRPO family).

### Why it is not in runtime for this project
- This repository targets encoder-style contrastive embedding alignment.
- Runtime objective is not causal language-model post-training.
- Adding TRL trainers here would increase complexity without solving the current objective.

### Where it is covered
- Documentation and optional dependency path for future scope expansion.
- Tool impact report records TRL as `documentation_only`.

### Practical effect to measure
- No direct runtime metric impact in this version by design.
- Reduced objective mismatch and maintenance surface.

## Official Sources

- Unsloth embedding fine-tuning docs: https://unsloth.ai/docs/basics/embedding-finetuning
- Unsloth requirements: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements
- PEFT type references (`FEATURE_EXTRACTION`): https://huggingface.co/docs/peft/main/package_reference/peft_types
- PEFT overview: https://huggingface.co/docs/peft/index
- TRL overview: https://huggingface.co/docs/trl
- TRL SFT trainer docs: https://huggingface.co/docs/trl/en/sft_trainer
- TRL DPO trainer docs: https://huggingface.co/docs/trl/en/dpo_trainer
- Sentence Transformers losses: https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html

