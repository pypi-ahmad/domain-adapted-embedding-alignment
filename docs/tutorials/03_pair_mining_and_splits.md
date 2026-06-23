# 03 - Pair Mining and Splits

## What it is
Pair mining transforms query relevance seeds into contrastive training rows containing:
- `query`
- `positive_text`
- `negative_text`
- `hard_negative_text`
- deterministic `split`

## Why it is used
Contrastive learning quality depends on how informative negatives are. Easy negatives teach little; hard negatives force useful representation boundaries.

## How it appears in code
- Pair builder: `src/domain_adapted_embedding_alignment/data/pair_builder.py`
- Deterministic split assignment: `_split_from_query_id`
- Domain quota policy: `_domain_quota`
- Hard negative mining: `_pick_hard_negative`

## Practical explanation from real implementation
The current strategy samples candidate negatives and scores lexical overlap with query tokens, then caches choices per query id for reproducibility.

This approach is intentionally runtime-efficient at `200000` pair scale and avoids expensive full-corpus BM25 rescoring for every row.

## Validation pointers
- Pair totals: `artifacts/reports/data_preparation_report.json`
- Split correctness tests: `tests/test_pair_builder.py`

## Beginner checkpoint
- Understand why split assignment is based on query id hash.
- Understand difference between random negative and hard negative.

## Advanced checkpoint
- Evaluate how lexical-overlap hard negatives may bias against semantic-hard negatives.
- Explain when to replace this strategy with semantic mining or ANN pre-mining.

