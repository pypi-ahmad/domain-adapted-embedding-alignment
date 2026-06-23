"""Build positive/negative/hard-negative training pairs.

Design goals:
- Preserve real query->positive grounding from source datasets.
- Add difficult negatives that are lexically/semantically close.
- Keep balanced domain coverage to avoid generic-domain dominance.
- Keep full-size pair generation feasible on local hardware.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from dataclasses import asdict
from typing import Iterable

import polars as pl
from loguru import logger

from domain_adapted_embedding_alignment.schemas import DocumentRecord, EvalQuery, PairRecord


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\-]+", (text or "").lower())


def _split_from_query_id(query_id: str) -> str:
    """Deterministic split assignment to prevent accidental leakage."""
    digest = hashlib.sha1(query_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _domain_quota(target_size: int) -> dict[str, int]:
    """Balanced quotas with extra weight for MS MARCO general supervision."""
    return {
        "general": int(target_size * 0.55),
        "medical": int(target_size * 0.15),
        "legal": int(target_size * 0.15),
        "cybersecurity": int(target_size * 0.15),
    }


def build_pairs(
    documents: list[DocumentRecord],
    queries: list[EvalQuery],
    target_size: int,
    seed: int,
) -> list[PairRecord]:
    """Build pair records with negative and hard-negative examples.

    Hard negatives are selected via deterministic lexical-overlap sampling and
    cached per query id for reproducibility and speed.
    """
    rng = random.Random(seed)

    docs_by_id = {doc.doc_id: doc for doc in documents}
    docs_by_domain: dict[str, list[DocumentRecord]] = defaultdict(list)
    for doc in documents:
        docs_by_domain[doc.domain].append(doc)

    all_docs = list(documents)
    domain_doc_ids = {
        domain: [doc.doc_id for doc in domain_docs]
        for domain, domain_docs in docs_by_domain.items()
    }
    domain_quota = _domain_quota(target_size)
    hard_negative_cache: dict[str, str] = {}
    token_cache: dict[str, set[str]] = {}

    def _token_set(doc_id: str) -> set[str]:
        cached = token_cache.get(doc_id)
        if cached is not None:
            return cached
        tokens = set(_tokenize(docs_by_id[doc_id].text))
        token_cache[doc_id] = tokens
        return tokens

    def _pick_negative_doc(relevant_ids: set[str], domain: str, label: str) -> DocumentRecord:
        """Pick a random negative while avoiding identical domain+label when possible."""
        for _ in range(32):
            candidate = rng.choice(all_docs)
            if candidate.doc_id in relevant_ids:
                continue
            if candidate.domain == domain and candidate.label == label:
                continue
            return candidate

        for candidate in all_docs:
            if candidate.doc_id in relevant_ids:
                continue
            if candidate.domain == domain and candidate.label == label:
                continue
            return candidate
        raise RuntimeError("No valid negative document candidate could be found.")

    def _pick_hard_negative(query: EvalQuery, relevant_ids: set[str]) -> DocumentRecord:
        """Pick in-domain hard negatives using lexical overlap over sampled candidates."""
        cached_doc_id = hard_negative_cache.get(query.query_id)
        if cached_doc_id and cached_doc_id not in relevant_ids:
            return docs_by_id[cached_doc_id]

        candidates = [doc_id for doc_id in domain_doc_ids.get(query.domain, []) if doc_id not in relevant_ids]
        if not candidates:
            candidates = [doc.doc_id for doc in all_docs if doc.doc_id not in relevant_ids]
        if not candidates:
            raise RuntimeError(f"No hard-negative candidates available for query_id={query.query_id}")

        sample_size = min(192, len(candidates))
        sampled_ids = candidates if sample_size == len(candidates) else rng.sample(candidates, sample_size)

        q_tokens = set(_tokenize(query.query))
        best_doc_id = sampled_ids[0]
        best_overlap = -1
        for doc_id in sampled_ids:
            overlap = len(q_tokens & _token_set(doc_id))
            if overlap > best_overlap:
                best_overlap = overlap
                best_doc_id = doc_id

        hard_negative_cache[query.query_id] = best_doc_id
        return docs_by_id[best_doc_id]

    # Group queries by domain so we can enforce quotas.
    queries_by_domain: dict[str, list[EvalQuery]] = defaultdict(list)
    for query in queries:
        queries_by_domain[query.domain].append(query)

    pairs: list[PairRecord] = []

    for domain, quota in domain_quota.items():
        domain_queries = queries_by_domain.get(domain, [])
        if not domain_queries:
            logger.warning("No queries found for domain='{}'", domain)
            continue

        produced = 0
        loop_idx = 0
        while produced < quota:
            query = domain_queries[loop_idx % len(domain_queries)]
            loop_idx += 1

            relevant_ids = set(query.relevant_doc_ids)
            if not relevant_ids:
                continue
            pos_doc_id = next(iter(relevant_ids))
            if pos_doc_id not in docs_by_id:
                continue

            pos_doc = docs_by_id[pos_doc_id]

            negative_doc = _pick_negative_doc(relevant_ids, domain, pos_doc.label)
            hard_negative_doc = _pick_hard_negative(query, relevant_ids)

            pair_id = f"pair_{domain}_{query.query_id}_{produced:07d}"
            pair = PairRecord(
                pair_id=pair_id,
                split=_split_from_query_id(query.query_id),
                domain=domain,
                source=pos_doc.source,
                query=query.query,
                positive_doc_id=pos_doc.doc_id,
                positive_text=pos_doc.text,
                negative_doc_id=negative_doc.doc_id,
                negative_text=negative_doc.text,
                hard_negative_doc_id=hard_negative_doc.doc_id,
                hard_negative_text=hard_negative_doc.text,
                relevance=1,
            )
            pairs.append(pair)
            produced += 1

    rng.shuffle(pairs)
    logger.info("Built {} pair records", len(pairs))
    return pairs


def persist_documents(documents: Iterable[DocumentRecord], path: str) -> None:
    """Persist canonical document corpus to parquet."""
    frame = pl.DataFrame([doc.to_dict() for doc in documents])
    frame.write_parquet(path)


def persist_queries(queries: Iterable[EvalQuery], path: str) -> None:
    """Persist query seeds and relevance mappings to parquet."""
    frame = pl.DataFrame([asdict(query) for query in queries])
    frame.write_parquet(path)


def persist_pairs(pairs: Iterable[PairRecord], path: str) -> None:
    """Persist training/evaluation pairs to parquet."""
    frame = pl.DataFrame([pair.to_dict() for pair in pairs])
    frame.write_parquet(path)
