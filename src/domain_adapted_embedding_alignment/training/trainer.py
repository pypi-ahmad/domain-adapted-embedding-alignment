"""Contrastive training loop with Multiple-Negatives + hard-negative InfoNCE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from domain_adapted_embedding_alignment.settings import Settings
from domain_adapted_embedding_alignment.training.data import ContrastivePairDataset
from domain_adapted_embedding_alignment.training.model import ContrastiveEmbeddingEncoder


@dataclass(slots=True)
class TrainingHistory:
    train_steps: list[int]
    train_losses: list[float]
    val_steps: list[int]
    val_losses: list[float]
    val_pair_accuracy: list[float]


class ContrastiveTrainer:
    """Encapsulates local fine-tuning and validation for embedding alignment."""

    def __init__(self, settings: Settings, pairs_path: Path) -> None:
        self.settings = settings
        self.pairs_path = pairs_path

        self.train_dataset = ContrastivePairDataset(str(pairs_path), split="train")
        self.val_dataset = ContrastivePairDataset(str(pairs_path), split="validation")

        self.model = ContrastiveEmbeddingEncoder(
            model_name=settings.trainable_model_name,
            max_length=settings.max_doc_length,
            lora_rank=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_dropout=settings.lora_dropout,
            gradient_checkpointing=settings.gradient_checkpointing,
        )

    @staticmethod
    def _collate(batch):
        return {
            "query": [item.query for item in batch],
            "positive": [item.positive_text for item in batch],
            "hard_negative": [item.hard_negative_text for item in batch],
        }

    def _compute_loss(self, query_emb: torch.Tensor, pos_emb: torch.Tensor, neg_emb: torch.Tensor) -> torch.Tensor:
        # Multiple negatives ranking loss (in-batch positives).
        logits = (query_emb @ pos_emb.T) / self.settings.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        inbatch_loss = F.cross_entropy(logits, labels)

        # Pairwise InfoNCE with explicit hard negatives.
        pos_scores = (query_emb * pos_emb).sum(dim=1, keepdim=True) / self.settings.temperature
        neg_scores = (query_emb * neg_emb).sum(dim=1, keepdim=True) / self.settings.temperature
        pair_logits = torch.cat([pos_scores, neg_scores], dim=1)
        pair_labels = torch.zeros(pair_logits.shape[0], dtype=torch.long, device=pair_logits.device)
        pair_loss = F.cross_entropy(pair_logits, pair_labels)

        return 0.5 * inbatch_loss + 0.5 * pair_loss

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.settings.eval_batch_size,
            shuffle=False,
            collate_fn=self._collate,
        )

        losses: list[float] = []
        accuracies: list[float] = []

        for batch in val_loader:
            query_emb = self.model.encode_texts(batch["query"])
            pos_emb = self.model.encode_texts(batch["positive"])
            neg_emb = self.model.encode_texts(batch["hard_negative"])

            loss = self._compute_loss(query_emb, pos_emb, neg_emb)
            losses.append(float(loss.item()))

            pos_scores = (query_emb * pos_emb).sum(dim=1)
            neg_scores = (query_emb * neg_emb).sum(dim=1)
            accuracies.append(float((pos_scores > neg_scores).float().mean().item()))

        self.model.train()
        val_loss = float(np.mean(losses)) if losses else 0.0
        val_acc = float(np.mean(accuracies)) if accuracies else 0.0
        return val_loss, val_acc

    def train(self) -> dict:
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.settings.train_batch_size,
            shuffle=True,
            collate_fn=self._collate,
            drop_last=True,
        )

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.settings.learning_rate, weight_decay=self.settings.weight_decay)

        total_steps = min(
            self.settings.train_max_steps,
            max(1, len(train_loader) * self.settings.train_epochs),
        )
        warmup_steps = max(1, int(total_steps * self.settings.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        history = TrainingHistory(
            train_steps=[],
            train_losses=[],
            val_steps=[],
            val_losses=[],
            val_pair_accuracy=[],
        )

        global_step = 0
        running_losses: list[float] = []
        best_val_loss = float("inf")

        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.settings.train_epochs):
            logger.info("Training epoch {}/{}", epoch + 1, self.settings.train_epochs)

            progress = tqdm(train_loader, desc=f"epoch_{epoch+1}")
            for batch in progress:
                query_emb = self.model.encode_texts(batch["query"])
                pos_emb = self.model.encode_texts(batch["positive"])
                neg_emb = self.model.encode_texts(batch["hard_negative"])

                loss = self._compute_loss(query_emb, pos_emb, neg_emb)
                loss = loss / self.settings.grad_accum_steps
                loss.backward()

                running_losses.append(float(loss.item()) * self.settings.grad_accum_steps)

                if (global_step + 1) % self.settings.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                progress.set_postfix(loss=f"{np.mean(running_losses[-20:]):.4f}")

                if global_step % self.settings.eval_every_steps == 0:
                    val_loss, val_acc = self._validate()
                    mean_train_loss = float(np.mean(running_losses[-self.settings.eval_every_steps :]))

                    history.train_steps.append(global_step)
                    history.train_losses.append(mean_train_loss)
                    history.val_steps.append(global_step)
                    history.val_losses.append(val_loss)
                    history.val_pair_accuracy.append(val_acc)

                    logger.info(
                        "step={} train_loss={:.4f} val_loss={:.4f} val_pair_accuracy={:.4f}",
                        global_step,
                        mean_train_loss,
                        val_loss,
                        val_acc,
                    )

                    checkpoint_dir = self.settings.model_dir / "checkpoints" / f"step_{global_step}"
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_adapter(str(checkpoint_dir))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dir = self.settings.model_dir / "best_adapter"
                        best_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_adapter(str(best_dir))

                if global_step >= self.settings.train_max_steps:
                    break

            if global_step >= self.settings.train_max_steps:
                break

        # Save final adapter.
        final_dir = self.settings.model_dir / "final_adapter"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_adapter(str(final_dir))

        history_payload = {
            "train_steps": history.train_steps,
            "train_losses": history.train_losses,
            "val_steps": history.val_steps,
            "val_losses": history.val_losses,
            "val_pair_accuracy": history.val_pair_accuracy,
            "best_val_loss": best_val_loss,
            "total_steps": global_step,
            "train_examples": len(self.train_dataset),
            "val_examples": len(self.val_dataset),
        }

        history_path = self.settings.eval_dir / "training_history.json"
        history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

        return history_payload
