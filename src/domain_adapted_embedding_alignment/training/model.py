"""Trainable Qwen embedding encoder with LoRA adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


@dataclass(slots=True)
class EncoderBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class ContrastiveEmbeddingEncoder(nn.Module):
    """Qwen embedding model wrapper with PEFT LoRA adapters.

    This module exposes `encode_texts` so the same model can be used for:
    - training,
    - retrieval evaluation,
    - and final inference export.
    """

    def __init__(
        self,
        model_name: str,
        max_length: int,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        torch_dtype = torch.float32
        if torch.cuda.is_available():
            # 4-bit loading on GPU keeps memory usage within 6GB profile budgets.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            torch_dtype = torch.float16

        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        if self.device_name == "cpu":
            self.model.to(torch.device("cpu"))

        logger.info("Initialized ContrastiveEmbeddingEncoder on {}", self.device_name)

    def _tokenize(self, texts: list[str]) -> EncoderBatch:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        return EncoderBatch(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        batch = self._tokenize(texts)
        outputs = self.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        hidden_state = outputs.last_hidden_state
        mask = batch.attention_mask.unsqueeze(-1)
        pooled = (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings = F.normalize(pooled, p=2, dim=1)
        return embeddings

    def save_adapter(self, output_dir: str) -> None:
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
