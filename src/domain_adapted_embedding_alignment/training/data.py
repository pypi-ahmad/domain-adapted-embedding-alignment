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
        frame = pl.read_parquet(pairs_path)
        frame = frame.filter(pl.col("split") == split)
        self.rows = frame.to_dicts()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> ContrastiveExample:
        row = self.rows[index]
        return ContrastiveExample(
            query=str(row["query"]),
            positive_text=str(row["positive_text"]),
            hard_negative_text=str(row["hard_negative_text"]),
        )
