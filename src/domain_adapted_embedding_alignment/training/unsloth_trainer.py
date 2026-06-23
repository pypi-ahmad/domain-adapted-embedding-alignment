"""Optional Unsloth-backed embedding training via SentenceTransformers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger

from domain_adapted_embedding_alignment.settings import Settings


class UnslothContrastiveTrainer:
    """Train embeddings with Unsloth acceleration when available.

    Notes:
    - This path follows Unsloth's embedding guidance built around FastSentenceTransformer.
    - The objective is still contrastive (triplet-formatted data with multiple-negatives loss).
    """

    def __init__(self, settings: Settings, pairs_path: Path) -> None:
        self.settings = settings
        self.pairs_path = pairs_path

        frame = pl.read_parquet(str(pairs_path))
        self.train_rows = frame.filter(pl.col("split") == "train")
        self.val_rows = frame.filter(pl.col("split") == "validation")
        if self.settings.unsloth_val_max_rows > 0:
            self.val_rows = self.val_rows.head(self.settings.unsloth_val_max_rows)

    def _build_model(self):
        from unsloth import FastSentenceTransformer

        model: Any
        use_4bit = bool(self.settings.unsloth_load_in_4bit)
        load_kwargs = {
            "model_name": self.settings.trainable_model_name,
            "max_seq_length": self.settings.unsloth_max_seq_length,
            "load_in_4bit": use_4bit,
            # Unsloth expects exactly one precision mode to be enabled.
            "load_in_16bit": not use_4bit,
        }
        try:
            model = FastSentenceTransformer.from_pretrained(**load_kwargs)
        except TypeError:
            # Keep compatibility with older/newer signatures.
            load_kwargs.pop("load_in_16bit", None)
            load_kwargs.pop("load_in_4bit", None)
            model = FastSentenceTransformer.from_pretrained(**load_kwargs)

        if isinstance(model, tuple):
            model = model[0]

        if hasattr(FastSentenceTransformer, "get_peft_model"):
            lora_kwargs = {
                "r": self.settings.lora_rank,
                "lora_alpha": self.settings.lora_alpha,
                "lora_dropout": self.settings.lora_dropout,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "use_gradient_checkpointing": self.settings.gradient_checkpointing,
            }
            try:
                model = FastSentenceTransformer.get_peft_model(model, **lora_kwargs)
            except TypeError:
                # Some versions require fewer args.
                lora_kwargs.pop("use_gradient_checkpointing", None)
                model = FastSentenceTransformer.get_peft_model(model, **lora_kwargs)

        return model

    def train(self) -> dict:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
        from sentence_transformers.evaluation import TripletEvaluator

        model = self._build_model()

        train_dataset = Dataset.from_dict(
            {
                "anchor": self.train_rows["query"].to_list(),
                "positive": self.train_rows["positive_text"].to_list(),
                "negative": self.train_rows["hard_negative_text"].to_list(),
            }
        )
        val_anchors = self.val_rows["query"].to_list()
        val_positives = self.val_rows["positive_text"].to_list()
        val_negatives = self.val_rows["hard_negative_text"].to_list()
        val_dataset = Dataset.from_dict(
            {
                "anchor": val_anchors,
                "positive": val_positives,
                "negative": val_negatives,
            }
        )

        output_dir = self.settings.model_dir / "checkpoints" / "unsloth_trainer"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_kwargs = {
            "output_dir": str(output_dir),
            "learning_rate": self.settings.learning_rate,
            "weight_decay": self.settings.weight_decay,
            "warmup_ratio": self.settings.warmup_ratio,
            "max_steps": self.settings.train_max_steps,
            "num_train_epochs": float(self.settings.train_epochs),
            "per_device_train_batch_size": self.settings.train_batch_size,
            "per_device_eval_batch_size": self.settings.eval_batch_size,
            "gradient_accumulation_steps": self.settings.grad_accum_steps,
            "eval_strategy": "steps",
            "eval_steps": self.settings.eval_every_steps,
            "save_strategy": "steps",
            "save_steps": self.settings.save_every_steps,
            "logging_steps": max(1, self.settings.eval_every_steps // 2),
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": torch.cuda.is_available(),
            "report_to": [],
        }
        try:
            train_args = SentenceTransformerTrainingArguments(**training_kwargs)
        except TypeError:
            # Compatibility path for older APIs using evaluation_strategy.
            training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")
            train_args = SentenceTransformerTrainingArguments(**training_kwargs)

        evaluator = TripletEvaluator(
            anchors=val_anchors,
            positives=val_positives,
            negatives=val_negatives,
            name="validation_triplets",
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            loss=losses.MultipleNegativesRankingLoss(model=model),
            evaluator=evaluator,
        )

        train_output = trainer.train()
        eval_output = trainer.evaluate()
        history_rows = list(trainer.state.log_history)

        # Save in two stable locations used by downstream pipelines.
        final_dir = self.settings.model_dir / "final_adapter"
        best_dir = self.settings.model_dir / "best_adapter"
        for path in [final_dir, best_dir]:
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(path))

        train_losses: list[float] = []
        train_steps: list[int] = []
        val_losses: list[float] = []
        val_steps: list[int] = []
        for row in history_rows:
            if "loss" in row and "step" in row:
                train_steps.append(int(row["step"]))
                train_losses.append(float(row["loss"]))
            if "eval_loss" in row and "step" in row:
                val_steps.append(int(row["step"]))
                val_losses.append(float(row["eval_loss"]))

        best_val_loss = min(val_losses) if val_losses else float(eval_output.get("eval_loss", 0.0))
        history_payload = {
            "train_steps": train_steps,
            "train_losses": train_losses,
            "val_steps": val_steps,
            "val_losses": val_losses,
            "val_pair_accuracy": [],
            "best_val_loss": float(best_val_loss),
            "total_steps": int(trainer.state.global_step),
            "train_examples": int(len(train_dataset)),
            "val_examples": int(len(val_dataset)),
            "train_runtime_seconds": float(train_output.metrics.get("train_runtime", 0.0)),
            "eval_runtime_seconds": float(eval_output.get("eval_runtime", 0.0)),
        }

        history_path = self.settings.eval_dir / "training_history.json"
        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
        logger.info("Unsloth training completed with {} steps", trainer.state.global_step)
        return history_payload
