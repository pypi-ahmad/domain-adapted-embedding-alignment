"""PyTorch datasets and collators for contrastive embedding training."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from torch.utils.data import Dataset


@dataclass(slots=True)
class ContrastiveExample:
    """Single contrastive sample with one positive and one hard negative."""

    query: str
    positive_text: str
    hard_negative_text: str


class ContrastivePairDataset(Dataset):
    """Dataset backed by pair parquet artifacts."""

    def __init__(self, pairs_path: str, split: str) -> None:
        frame = (
            pl.scan_parquet(pairs_path)
            .filter(pl.col("split") == split)
            .select(["query", "positive_text", "hard_negative_text"])
            .collect(streaming=True)
        )
        self.queries = frame["query"].to_list()
        self.positives = frame["positive_text"].to_list()
        self.hard_negatives = frame["hard_negative_text"].to_list()

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> ContrastiveExample:
        return ContrastiveExample(
            query=str(self.queries[index]),
            positive_text=str(self.positives[index]),
            hard_negative_text=str(self.hard_negatives[index]),
        )
