"""LLM-as-a-judge evaluation using local Ollama models."""

from __future__ import annotations

import json
from dataclasses import dataclass

import ollama
from loguru import logger


@dataclass(slots=True)
class JudgeScore:
    relevance: float
    usefulness: float
    semantic_relevance: float
    reasoning: str


def judge_retrieval_context(
    model_name: str,
    query: str,
    context: str,
    retrieved_summary: str,
) -> JudgeScore:
    """Ask an LLM judge to score retrieval quality.

    Returned scores are in [1, 5]. Downstream metrics convert to [0, 1].
    """
    prompt = f"""
You are evaluating retrieval quality for a domain-specific search system.

Query:
{query}

Retrieved context:
{context[:5000]}

Retrieved summary:
{retrieved_summary[:1500]}

Score each criterion from 1 to 5:
- relevance
- usefulness
- semantic_relevance

Return JSON only:
{{
  "relevance": 1,
  "usefulness": 1,
  "semantic_relevance": 1,
  "reasoning": "one concise sentence"
}}
"""

    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "num_predict": 250},
        )
        payload = json.loads(response["message"]["content"])
        return JudgeScore(
            relevance=float(payload.get("relevance", 3)),
            usefulness=float(payload.get("usefulness", 3)),
            semantic_relevance=float(payload.get("semantic_relevance", 3)),
            reasoning=str(payload.get("reasoning", "No reasoning provided.")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge model '{}' failed: {}", model_name, exc)
        return JudgeScore(
            relevance=3.0,
            usefulness=3.0,
            semantic_relevance=3.0,
            reasoning="Judge fallback score used because model call failed.",
        )
