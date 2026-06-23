# 12 - Reports, Validation, and Auditability

## What it is
Final-stage reporting and validation layer that confirms run completeness and captures tooling impact.

## Why it is used
A run is not operationally complete until required artifacts exist, are non-empty, and runtime metadata is coherent.

## How it appears in code
- Final report assembly: `src/domain_adapted_embedding_alignment/pipelines/build_final_report.py`
- Completion checks: `build_completion_checklist(...)` in `src/domain_adapted_embedding_alignment/pipelines/finalize_reports.py`
- Tool impact summary: `build_tool_impact_report(...)` in `src/domain_adapted_embedding_alignment/pipelines/finalize_reports.py`

## Practical explanation from run outputs
From `artifacts/reports/completion_checklist.json`:
- `all_required_artifacts_ok=true`
- `all_runtime_checks_ok=true`
- `project_completion_ok=true`

From `artifacts/reports/tool_impact_report.json`:
- `unsloth.used=true`
- `peft.used=false`
- `trl.used_in_runtime=false`

From `artifacts/reports/final_report.json`:
- Combined payload includes data prep, training, evaluation, RAG, GraphRAG, visualizations, inference, completion, and tool impact.

## Audit checklist
1. Confirm all required artifacts exist and are non-empty.
2. Confirm runtime backend metadata exists in training report.
3. Confirm stage logs exist under `artifacts/logs/`.
4. Confirm tests passed (`artifacts/logs/09_pytest.log`).
5. Confirm notebook execution outputs exist (`notebooks/*.executed.ipynb`).

## Practical outcome
This project run satisfies the completion gate and can be treated as an auditable end-to-end execution.

