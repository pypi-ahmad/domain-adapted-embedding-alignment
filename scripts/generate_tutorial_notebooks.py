#!/usr/bin/env python3
"""Generate tutorial-first notebooks with consistent educational structure."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"


NOTEBOOK_SPECS = [
    (
        "01_problem_analysis_and_embedding_theory.ipynb",
        "Notebook 01 - Problem Analysis and Embedding Theory Foundations",
        "Problem framing for specialized retrieval and embedding alignment.",
    ),
    (
        "02_data_preparation_and_pair_mining.ipynb",
        "Notebook 02 - Data Preparation, Relevance Labels, and Hard Negative Mining",
        "How query-document supervision is constructed for domain adaptation.",
    ),
    (
        "03_baseline_retrieval_systems.ipynb",
        "Notebook 03 - Baseline Retrieval Systems (BM25, Dense, Hybrid)",
        "How to establish trustworthy baselines before model adaptation.",
    ),
    (
        "04_contrastive_learning_and_finetuning.ipynb",
        "Notebook 04 - Contrastive Learning, PEFT, and Optional Unsloth Acceleration",
        "How embedding alignment training is implemented and tuned.",
    ),
    (
        "05_evaluation_and_error_analysis.ipynb",
        "Notebook 05 - Retrieval Evaluation, Judge Metrics, and Error Analysis",
        "How to interpret retrieval quality and failure modes with real metrics.",
    ),
    (
        "06_visualization_embedding_space.ipynb",
        "Notebook 06 - Embedding Space Visualization (PCA, t-SNE, UMAP)",
        "How vector-space geometry changes before vs after alignment.",
    ),
    (
        "07_rag_integration_chroma_pinecone.ipynb",
        "Notebook 07 - RAG Integration with ChromaDB and Pinecone",
        "How tuned embeddings change practical RAG retrieval outcomes.",
    ),
    (
        "08_graphrag_integration_and_inference.ipynb",
        "Notebook 08 - GraphRAG Integration and End-to-End Inference",
        "How local/global graph retrieval behaves with tuned embeddings.",
    ),
]


TECHNIQUE_TEMPLATE = """## What is This Technique?

This notebook focuses on **{focus}** in the domain-adapted embedding alignment pipeline.

## Definition and Core Concepts

- **Lexical similarity**: text overlap at the token/string level.
- **Semantic similarity**: meaning-level closeness, even when vocabulary differs.
- **Embedding alignment**: training embeddings so domain-equivalent concepts are closer in vector space.
- **Retrieval quality first principle**: generation quality in RAG is bounded by retrieval quality.

## Why Was This Technique Developed?

Traditional retrieval pipelines often underperform in medical, legal, and cybersecurity settings because domain terms are highly variable (synonyms, abbreviations, jargon, and evolving vocabulary).
This technique was developed to reduce those failures using representation learning and domain-aware supervision.

## What Limitations of Traditional RAG Does It Solve?

- Reduces misses caused by synonym drift (for example, `termination` vs `rescission`).
- Reduces false negatives when domain terms do not lexically overlap.
- Improves retrieval grounding quality so downstream answers have stronger factual context.

## Architecture and Workflow Diagram Explanation

```mermaid
flowchart LR
    A[Domain Query] --> B[Retriever]
    B --> C[Top-k Context]
    C --> D[RAG/GraphRAG Consumer]
    E[Contrastive Training Data] --> F[Embedding Alignment]
    F --> B
```

The key idea is that the retriever is the control point: better embedding geometry improves candidate ranking quality before any generator consumes context.

## Component-by-Component Breakdown

1. Data ingestion and relevance mapping.
2. Positive, negative, and hard-negative supervision construction.
3. Baseline retrieval measurement.
4. Embedding alignment training (PEFT default, Unsloth optional).
5. Evaluation across quality, ranking, and latency.
6. RAG and GraphRAG integration for practical impact validation.

## When Should It Be Used in Real-World Systems?

Use this technique when:
- Domain vocabulary is specialized and rapidly evolving.
- BM25-only retrieval misses semantically relevant context.
- Your RAG/GraphRAG quality is bottlenecked by retrieval recall and ranking.

## Advantages and Disadvantages

**Advantages**
- Better in-domain recall and semantic matching.
- Stronger retrieval grounding for downstream reasoning.
- Parameter-efficient adaptation with PEFT.

**Disadvantages**
- Requires carefully curated relevance supervision.
- Hard-negative quality strongly influences outcomes.
- Adds training/evaluation complexity compared with static embeddings.

## Comparison Against Standard RAG and Implemented RAG Variants

- **Standard RAG (baseline embeddings)**: simpler but more lexical-mismatch failures.
- **Aligned dense retrieval**: improved semantic recall and ranking precision in-domain.
- **Hybrid retrieval**: robust baseline; tuned dense often improves semantic edge cases.
- **GraphRAG**: better for multi-hop entity exploration; still depends on initial retrieval quality.

## Implementation Details and Design Decisions Used in This Project

- Primary trainable model: `Qwen/Qwen3-Embedding-0.6B`.
- Baseline model: `qwen3-embedding:4b`.
- Training objective: contrastive alignment with hard negatives.
- Runtime policy:
  - PEFT: primary production backend.
  - Unsloth: optional acceleration backend when runtime supports it.
  - TRL: documented and evaluated for fit, intentionally excluded from runtime in this project version.
"""


TOOLING_COVERAGE = """## Unsloth, PEFT, and TRL Coverage

### PEFT
- **Definition**: parameter-efficient adaptation (LoRA adapters instead of full model updates).
- **Why used**: best fit for contrastive embedding alignment under local compute constraints.
- **Where used**: primary training backend and adapter-based tuned retrieval path.
- **What changed**: training outputs are lightweight adapters (`best_adapter`, `final_adapter`), enabling reproducible local tuning.
- **Post-run effect to analyze**: retrieval-quality uplift and latency tradeoffs compared to baseline.

### Unsloth
- **Definition**: acceleration framework for efficient fine-tuning; embedding docs center on `FastSentenceTransformer`.
- **Why used**: optional speed/VRAM optimization when compatible runtime exists.
- **Where used**: optional training backend selected via `--training-backend unsloth` or `auto` when available.
- **What changed**: backend selection now records requested backend, used backend, and fallback reason.
- **Post-run effect to analyze**: training runtime, throughput, and quality parity/improvement vs PEFT.

### TRL
- **Definition**: post-training library with trainers like SFT, DPO, PPO, and related alignment methods.
- **Why not used in runtime here**: current project target is encoder-style contrastive embedding alignment, not causal-LM post-training.
- **Where covered**: documented in tutorial and decisions file to avoid misuse.
- **What changed**: clearer scope boundaries and lower implementation complexity.
- **Post-run effect**: no direct runtime metric impact in this version (intentional design decision).
"""


STEP_BY_STEP = """## Step-by-Step Practical Workflow

1. Confirm prerequisites and environment settings.
2. Execute the relevant pipeline stage using `dea-cli`.
3. Load generated artifacts (`artifacts/reports`, `artifacts/evaluation`, `artifacts/figures`).
4. Compare baseline vs tuned quality and latency metrics.
5. Record failure modes, tradeoffs, and deployment recommendations.
"""


POST_RUN_ANALYSIS = """## Post-Run Analysis of Actual Outputs and Metrics

Use this section **after real end-to-end execution**. The cell below reads generated artifacts and computes practical comparisons.
"""


ANALYSIS_CODE = """from pathlib import Path
import json

project_root = Path("..").resolve()
reports_dir = project_root / "artifacts" / "reports"
eval_dir = project_root / "artifacts" / "evaluation"
fig_dir = project_root / "artifacts" / "figures"

paths = {
    "training_report": reports_dir / "training_report.json",
    "final_report": reports_dir / "final_report.json",
    "retrieval_evaluation": eval_dir / "retrieval_evaluation.json",
    "rag_benchmark": eval_dir / "rag_benchmark.json",
    "graphrag_benchmark": eval_dir / "graphrag_benchmark.json",
    "training_history": eval_dir / "training_history.json",
}

for name, path in paths.items():
    print(f"{name:20s} exists={path.exists()} -> {path}")

def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

training = _load(paths["training_report"])
retrieval = _load(paths["retrieval_evaluation"])
rag = _load(paths["rag_benchmark"])
graphrag = _load(paths["graphrag_benchmark"])
final_report = _load(paths["final_report"])

if training:
    runtime = training.get("runtime", {})
    print("\\n[Training Backend]")
    print("requested:", runtime.get("backend_requested"))
    print("used     :", runtime.get("backend_used"))
    print("fallback :", runtime.get("fallback_reason"))

if retrieval:
    systems = retrieval.get("systems", {})
    baseline = systems.get("baseline_dense_qwen4b", {})
    tuned = systems.get("tuned_dense_qwen0_6b_lora", {})
    b_metrics = baseline.get("retrieval_metrics", {})
    t_metrics = tuned.get("retrieval_metrics", {})
    b_latency = baseline.get("latency_metrics", {})
    t_latency = tuned.get("latency_metrics", {})

    print("\\n[Retrieval Quality Delta: Tuned - Baseline]")
    for key in ["recall@10", "precision@10", "mrr", "ndcg@10", "map@10"]:
        print(f"{key:12s} {t_metrics.get(key, 0.0) - b_metrics.get(key, 0.0):+.4f}")

    print("\\n[Latency Delta ms: Tuned - Baseline]")
    for key in ["mean_ms", "p50_ms", "p95_ms"]:
        print(f"{key:12s} {t_latency.get(key, 0.0) - b_latency.get(key, 0.0):+.2f}")

    print("\\n[Generation-Quality Proxy via LLM Judge]")
    print("baseline:", baseline.get("judge_metrics", {}))
    print("tuned   :", tuned.get("judge_metrics", {}))

if rag:
    print("\\n[RAG Benchmarks]")
    print(json.dumps(rag, indent=2))

if graphrag:
    print("\\n[GraphRAG Benchmarks]")
    print(json.dumps(graphrag, indent=2))

if final_report:
    print("\\n[Inference Snapshot]")
    inf = final_report.get("inference_examples", {})
    results = inf.get("results", []) if isinstance(inf, dict) else inf
    for item in results[:3]:
        print("-", item.get("query", ""))
        for hit in item.get("ranked_results", [])[:3]:
            print("   ", hit.get("rank"), hit.get("domain"), f"{hit.get('score', 0.0):.4f}", hit.get("preview", "")[:80])
"""


INTERPRETATION = """## How to Interpret the Observed Results

1. **Retrieval quality**: improved Recall@k/MRR/NDCG indicates better ranking quality and fewer missed relevant documents.
2. **Latency**: compare mean and p95; improvements are useful only if quality remains acceptable.
3. **Judge metrics**: treat as qualitative proxies for context usefulness/relevance, not absolute truth.
4. **RAG and GraphRAG impact**: verify hit-rate changes and failure patterns across medical/legal/cyber domains.

## Why Specific Outputs Were Produced

- Gains usually come from improved domain-term alignment and better hard-negative separation.
- Residual failures often come from ambiguous queries, noisy negatives, or insufficient domain coverage.
- Latency differences depend on backend choice, model size, quantization, and device.

## Detailed Observations, Lessons Learned, and Practical Takeaways

- Keep baseline measurements fixed before tuning.
- Validate hard-negative quality continuously.
- Choose backend by objective:
  - PEFT for stable production alignment.
  - Unsloth when compatible acceleration is available.
  - TRL only when moving to causal-LM post-training objectives.

## Final Conclusion (Using Actual Measured Results)

After execution, summarize:
1. whether tuned embeddings improved retrieval quality materially,
2. whether latency remained operationally acceptable,
3. whether RAG/GraphRAG grounding quality improved in practice,
4. whether backend choice (PEFT vs optional Unsloth) delivered net value for your deployment constraints.
"""


def make_notebook(title: str, focus: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# {title}"),
        nbf.v4.new_markdown_cell(
            "## Learning Objectives\n\n"
            "By the end of this notebook, you should be able to explain the technique, run it, and interpret "
            "its impact on retrieval quality, latency, and downstream RAG behavior."
        ),
        nbf.v4.new_markdown_cell(TECHNIQUE_TEMPLATE.format(focus=focus)),
        nbf.v4.new_markdown_cell(TOOLING_COVERAGE),
        nbf.v4.new_markdown_cell(STEP_BY_STEP),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "project_root = Path('..').resolve()\n"
            "print('Project root:', project_root)\n"
            "print('Run stages with dea-cli as needed for this notebook.')"
        ),
        nbf.v4.new_markdown_cell(POST_RUN_ANALYSIS),
        nbf.v4.new_code_cell(ANALYSIS_CODE),
        nbf.v4.new_markdown_cell(INTERPRETATION),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for filename, title, focus in NOTEBOOK_SPECS:
        nb = make_notebook(title=title, focus=focus)
        path = NOTEBOOK_DIR / filename
        path.write_text(nbf.writes(nb), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
