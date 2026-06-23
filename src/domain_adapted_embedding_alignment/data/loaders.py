"""Load real corpora and construct query-positive seeds.

All loaders use public real datasets only:
- microsoft/ms_marco
- bigbio/medmentions
- coastalcph/lex_glue (scotus + eurlex)
- MITRE ATT&CK STIX bundles (local mirror from reference repo)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from loguru import logger

from domain_adapted_embedding_alignment.schemas import DocumentRecord, EvalQuery


LEGAL_QUERY_TEMPLATES: dict[str, str] = {
    "Criminal Procedure": "Find legal precedents and interpretations for criminal procedure in case law.",
    "Civil Rights": "Retrieve legal analysis on civil rights protections and judicial rulings.",
    "First Amendment": "Find legal decisions related to First Amendment interpretation.",
    "Due Process": "Retrieve legal doctrine and examples for due process disputes.",
    "Privacy": "Find legal rulings discussing privacy and constitutional protections.",
    "Attorneys": "Retrieve legal content on attorney conduct and legal profession rules.",
    "Unions": "Find legal decisions around labor unions and employment law.",
    "Economic Activity": "Retrieve legal materials on business regulation and economic activity.",
    "Judicial Power": "Find legal rulings about limits and scope of judicial power.",
    "Federalism": "Retrieve legal decisions on federal-state power balance.",
    "Interstate Relations": "Find legal cases involving disputes across U.S. states.",
    "Federal Taxation": "Retrieve legal interpretations of federal taxation issues.",
    "Miscellaneous": "Find legal rulings covering miscellaneous constitutional issues.",
    "civil law": "Retrieve legal documents covering civil law obligations and disputes.",
    "criminal law": "Find legal materials related to criminal law provisions.",
    "energy policy": "Retrieve EU legal acts focused on energy policy.",
    "environmental policy": "Find EU legal regulations related to environmental policy.",
    "consumer protection": "Retrieve EU consumer protection legislation and guidance.",
    "external relations": "Find legal documents on EU external relations policy.",
    "human rights": "Retrieve EU legal material on human rights protections.",
    "asylum and immigration policy": "Find legal documents on EU asylum and immigration policy.",
    "public health": "Retrieve EU legal directives related to public health.",
    "transport policy": "Find legal regulations and directives on transport policy.",
    "competition policy": "Retrieve legal materials on EU competition and antitrust policy.",
}


def _stable_doc_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _extract_medmentions_passages(row: dict[str, Any]) -> tuple[str, str]:
    """Extract title and abstract from MedMentions passage format."""
    title = ""
    abstract = ""
    for passage in row.get("passages", []):
        text = _first_text(passage.get("text", "")).strip()
        passage_type = _first_text(passage.get("type", "")).lower()
        if "title" in passage_type and not title:
            title = text
        elif "abstract" in passage_type and not abstract:
            abstract = text
        elif not abstract and text:
            abstract = text
    return title, abstract


def load_msmarco(
    max_rows: int,
    seed: int,
) -> tuple[list[DocumentRecord], list[EvalQuery]]:
    """Load MS MARCO and return documents plus query-positive mappings."""
    logger.info("Loading microsoft/ms_marco v1.1 train split")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")
    ds = ds.shuffle(seed=seed).select(range(min(max_rows, len(ds))))

    documents: list[DocumentRecord] = []
    queries: list[EvalQuery] = []
    seen_doc_ids: set[str] = set()

    for row in ds:
        query_text = str(row["query"]).strip()
        query_id = f"msmarco_q_{int(row['query_id'])}"
        passages = row["passages"]
        passage_texts = passages.get("passage_text", [])
        selected_flags = passages.get("is_selected", [])

        selected_idx = [idx for idx, flag in enumerate(selected_flags) if int(flag) == 1]
        non_selected_idx = [idx for idx, flag in enumerate(selected_flags) if int(flag) == 0]

        if not selected_idx:
            continue

        pos_text = str(passage_texts[selected_idx[0]]).strip()
        if len(pos_text) < 32:
            continue

        pos_doc_id = _stable_doc_id("msmarco_doc", pos_text)
        if pos_doc_id not in seen_doc_ids:
            seen_doc_ids.add(pos_doc_id)
            documents.append(
                DocumentRecord(
                    doc_id=pos_doc_id,
                    domain="general",
                    source="ms_marco",
                    title="",
                    label="selected_passage",
                    text=pos_text,
                    metadata={"query_id": query_id},
                )
            )

        # Also index one lexical near-negative from the same query context.
        if non_selected_idx:
            neg_text = str(passage_texts[non_selected_idx[0]]).strip()
            if len(neg_text) >= 32:
                neg_doc_id = _stable_doc_id("msmarco_doc", neg_text)
                if neg_doc_id not in seen_doc_ids:
                    seen_doc_ids.add(neg_doc_id)
                    documents.append(
                        DocumentRecord(
                            doc_id=neg_doc_id,
                            domain="general",
                            source="ms_marco",
                            title="",
                            label="non_selected_passage",
                            text=neg_text,
                            metadata={"query_id": query_id},
                        )
                    )

        answers = row.get("answers", [])
        reference_answer = str(answers[0]).strip() if answers else pos_text[:220]

        queries.append(
            EvalQuery(
                query_id=query_id,
                domain="general",
                query=query_text,
                relevant_doc_ids=[pos_doc_id],
                reference_answer=reference_answer,
            )
        )

    logger.info("MS MARCO loaded: {} docs | {} query seeds", len(documents), len(queries))
    return documents, queries


def load_medical_medmentions(
    max_records: int,
    seed: int,
) -> tuple[list[DocumentRecord], list[EvalQuery]]:
    """Load MedMentions and build domain-specific biomedical query seeds."""
    logger.info("Loading bigbio/medmentions across train/validation/test")
    # `trust_remote_code` is deprecated in recent `datasets` versions.
    # The cached/native dataset can still be loaded directly.
    dataset = load_dataset("bigbio/medmentions")

    merged_rows: list[dict[str, Any]] = []
    for split_name in ["train", "validation", "test"]:
        merged_rows.extend(list(dataset[split_name]))

    # Deterministic sampling by slicing after shuffle.
    import numpy as np

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(merged_rows))
    order = order[: min(max_records, len(order))]

    documents: list[DocumentRecord] = []
    queries: list[EvalQuery] = []

    for idx in order:
        row = merged_rows[int(idx)]
        title, abstract = _extract_medmentions_passages(row)
        text = "\n\n".join([part for part in [title, abstract] if part]).strip()
        if len(text) < 64:
            continue

        doc_id = _stable_doc_id("med_doc", text)
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                domain="medical",
                source="bigbio/medmentions",
                title=title,
                label="biomedical_abstract",
                text=text,
                metadata={"pmid": str(row.get("pmid", ""))},
            )
        )

        entities = row.get("entities", [])
        entity_texts: list[str] = []
        for entity in entities:
            mention = _first_text(entity.get("text", "")).strip()
            if len(mention) >= 3:
                entity_texts.append(mention)
        entity_texts = list(dict.fromkeys(entity_texts))[:2]

        if not entity_texts:
            continue

        for e_idx, entity_text in enumerate(entity_texts):
            query = f"What biomedical evidence is available about {entity_text}?"
            queries.append(
                EvalQuery(
                    query_id=f"med_q_{doc_id}_{e_idx}",
                    domain="medical",
                    query=query,
                    relevant_doc_ids=[doc_id],
                    reference_answer=abstract[:280] if abstract else text[:280],
                )
            )

    logger.info("Medical loaded: {} docs | {} query seeds", len(documents), len(queries))
    return documents, queries


def load_legal_lexglue(
    max_records: int,
    seed: int,
) -> tuple[list[DocumentRecord], list[EvalQuery]]:
    """Load LexGLUE SCOTUS + EURLEX with label-grounded legal queries."""
    logger.info("Loading coastalcph/lex_glue subsets: scotus + eurlex")

    import numpy as np

    documents: list[DocumentRecord] = []
    queries: list[EvalQuery] = []

    # SCOTUS
    scotus = load_dataset("coastalcph/lex_glue", "scotus", split="train")
    scotus = scotus.shuffle(seed=seed).select(range(min(max_records // 2, len(scotus))))
    scotus_label_names = {
        0: "Criminal Procedure",
        1: "Civil Rights",
        2: "First Amendment",
        3: "Due Process",
        4: "Privacy",
        5: "Attorneys",
        6: "Unions",
        7: "Economic Activity",
        8: "Judicial Power",
        9: "Federalism",
        10: "Interstate Relations",
        11: "Federal Taxation",
        12: "Miscellaneous",
    }

    for idx, row in enumerate(scotus):
        text = re.sub(r"\s+", " ", str(row["text"]).strip())
        if len(text) < 160:
            continue

        label = int(row["label"])
        label_name = scotus_label_names.get(label, "Miscellaneous")
        doc_id = f"legal_scotus_{idx:06d}"

        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                domain="legal",
                source="lex_glue/scotus",
                title="",
                label=label_name,
                text=text,
                metadata={"label_id": label},
            )
        )

        query = LEGAL_QUERY_TEMPLATES.get(
            label_name,
            f"Find legal documents discussing {label_name.lower()} in U.S. case law.",
        )
        queries.append(
            EvalQuery(
                query_id=f"legal_q_scotus_{idx:06d}",
                domain="legal",
                query=query,
                relevant_doc_ids=[doc_id],
                reference_answer=text[:280],
            )
        )

    # EURLEX
    eurlex = load_dataset("coastalcph/lex_glue", "eurlex", split="train")
    eurlex = eurlex.shuffle(seed=seed).select(range(min(max_records // 2, len(eurlex))))

    rng = np.random.default_rng(seed)
    for idx, row in enumerate(eurlex):
        text = re.sub(r"\s+", " ", str(row["text"]).strip())
        if len(text) < 160:
            continue

        labels = row.get("labels", [])
        if not labels:
            continue
        label = int(labels[0])
        label_name = str(label)

        # A compact map for frequently used legal domains.
        label_map = {
            13: "environmental policy",
            15: "consumer protection",
            21: "human rights",
            23: "asylum and immigration policy",
            25: "public health",
            29: "competition policy",
            31: "social policy",
            40: "european union law",
            41: "internal market and customs union",
            53: "economic, monetary and fiscal policy",
            60: "eu finance, budget and monetary system",
            70: "general political and institutional framework",
        }
        label_name = label_map.get(label, label_name)

        doc_id = f"legal_eurlex_{idx:06d}"
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                domain="legal",
                source="lex_glue/eurlex",
                title="",
                label=label_name,
                text=text,
                metadata={"label_id": label},
            )
        )

        query = LEGAL_QUERY_TEMPLATES.get(
            label_name,
            f"Retrieve EU legal directives and regulations about {label_name}.",
        )
        # Add a small lexical variant to create realistic legal term mismatch.
        if rng.random() < 0.35:
            query = query.replace("legal", "regulatory")

        queries.append(
            EvalQuery(
                query_id=f"legal_q_eurlex_{idx:06d}",
                domain="legal",
                query=query,
                relevant_doc_ids=[doc_id],
                reference_answer=text[:280],
            )
        )

    logger.info("Legal loaded: {} docs | {} query seeds", len(documents), len(queries))
    return documents, queries


def load_cyber_mitre(
    max_records: int,
    mitre_root: Path,
) -> tuple[list[DocumentRecord], list[EvalQuery]]:
    """Load ATT&CK STIX objects from local MITRE CTI repository."""
    supported_files = [
        mitre_root / "enterprise-attack" / "enterprise-attack.json",
        mitre_root / "mobile-attack" / "mobile-attack.json",
        mitre_root / "ics-attack" / "ics-attack.json",
    ]

    objects: list[dict[str, Any]] = []
    for file_path in supported_files:
        if not file_path.exists():
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        objects.extend(payload.get("objects", []))

    # Keep only text-rich objects that support retrieval semantics.
    candidate_types = {
        "attack-pattern": "technique",
        "intrusion-set": "threat_actor",
        "campaign": "campaign",
        "malware": "malware",
        "tool": "tool",
        "vulnerability": "vulnerability",
    }

    documents: list[DocumentRecord] = []
    queries: list[EvalQuery] = []

    kept = 0
    for obj in objects:
        obj_type = str(obj.get("type", ""))
        if obj_type not in candidate_types:
            continue

        description = str(obj.get("description", "")).strip()
        name = str(obj.get("name", "")).strip()
        if len(description) < 64 or not name:
            continue

        stix_id = str(obj.get("id", ""))
        doc_id = f"cyber_{stix_id.replace('--', '_')}"
        label = candidate_types[obj_type]

        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                domain="cybersecurity",
                source="mitre_cti",
                title=name,
                label=label,
                text=f"{name}. {description}",
                metadata={"stix_type": obj_type},
            )
        )

        if label == "technique":
            query = f"How is {name} related to credential theft or account compromise techniques?"
        elif label == "threat_actor":
            query = f"What tactics and malware are associated with threat actor {name}?"
        elif label == "campaign":
            query = f"Which techniques are linked with campaign {name}?"
        elif label == "vulnerability":
            query = f"What exploitation context is documented for vulnerability {name}?"
        else:
            query = f"What does threat intelligence report about {name}?"

        queries.append(
            EvalQuery(
                query_id=f"cyber_q_{kept:06d}",
                domain="cybersecurity",
                query=query,
                relevant_doc_ids=[doc_id],
                reference_answer=description[:280],
            )
        )

        kept += 1
        if kept >= max_records:
            break

    logger.info("Cyber loaded: {} docs | {} query seeds", len(documents), len(queries))
    return documents, queries


def locate_mitre_cti_root(project_root: Path) -> Path:
    """Locate MITRE CTI mirror from known local project paths."""
    candidates = [
        project_root.parent / "Cybersecurity-Threat-Intelligence-GraphRAG" / "data" / "raw" / "mitre-cti",
        project_root / "data" / "raw" / "mitre-cti",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "MITRE CTI data not found. Expected under ../Cybersecurity-Threat-Intelligence-GraphRAG/data/raw/mitre-cti"
    )
