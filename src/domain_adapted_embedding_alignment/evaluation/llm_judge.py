"""LLM-as-a-judge evaluation using local Ollama models."""

from __future__ import annotations

import json
from dataclasses import dataclass

import ollama
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError


@dataclass(slots=True)
class JudgeScore:
    relevance: float
    usefulness: float
    semantic_relevance: float
    reasoning: str


class _JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevance: float = Field(ge=1.0, le=5.0)
    usefulness: float = Field(ge=1.0, le=5.0)
    semantic_relevance: float = Field(ge=1.0, le=5.0)
    reasoning: str = Field(min_length=1, max_length=512)


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevance": {"type": "number", "minimum": 1, "maximum": 5},
        "usefulness": {"type": "number", "minimum": 1, "maximum": 5},
        "semantic_relevance": {"type": "number", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string"},
    },
    "required": ["relevance", "usefulness", "semantic_relevance", "reasoning"],
    "additionalProperties": False,
}


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
            format=_JUDGE_SCHEMA,
            options={"temperature": 0.0, "num_predict": 250},
        )
        content = response["message"]["content"]
        if isinstance(content, str):
            payload = _JudgeResponse.model_validate_json(content)
        else:
            payload = _JudgeResponse.model_validate(content)
        return JudgeScore(
            relevance=float(payload.relevance),
            usefulness=float(payload.usefulness),
            semantic_relevance=float(payload.semantic_relevance),
            reasoning=str(payload.reasoning),
        )
    except (ValidationError, KeyError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Judge model '{}' returned invalid schema payload: {}", model_name, exc)
        return JudgeScore(
            relevance=3.0,
            usefulness=3.0,
            semantic_relevance=3.0,
            reasoning="Judge fallback score used because schema validation failed.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge model '{}' failed: {}", model_name, exc)
        return JudgeScore(
            relevance=3.0,
            usefulness=3.0,
            semantic_relevance=3.0,
            reasoning="Judge fallback score used because model call failed.",
        )
