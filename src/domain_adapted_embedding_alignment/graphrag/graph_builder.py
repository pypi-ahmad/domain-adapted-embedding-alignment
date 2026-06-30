"""Construct lightweight document-entity graphs for GraphRAG demos."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
from community import community_louvain
from loguru import logger

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "from",
    "this",
    "have",
    "are",
    "was",
    "were",
    "has",
    "had",
    "into",
    "about",
    "which",
    "their",
    "also",
    "than",
    "after",
    "under",
    "between",
    "where",
    "when",
    "while",
    "using",
    "used",
    "such",
}


def _extract_terms(text: str, max_terms: int = 12) -> list[str]:
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text)]
    tokens = [tok for tok in tokens if tok not in _STOPWORDS]
    counts = Counter(tokens)
    return [term for term, _ in counts.most_common(max_terms)]


def build_document_entity_graph(documents: list[dict[str, Any]]) -> tuple[nx.Graph, dict[str, int]]:
    """Build a bipartite-style graph with document and entity-term nodes."""
    graph = nx.Graph()

    term_to_docs: dict[str, list[str]] = defaultdict(list)

    for doc in documents:
        doc_id = doc["doc_id"]
        graph.add_node(doc_id, node_type="document", domain=doc.get("domain", "unknown"))

        terms = _extract_terms(doc.get("text", ""))
        graph.nodes[doc_id]["term_set"] = set(terms)
        for term in terms:
            term_node = f"term::{term}"
            graph.add_node(term_node, node_type="term")
            graph.add_edge(doc_id, term_node, weight=1.0)
            term_to_docs[term].append(doc_id)

    # Add doc-doc edges for shared terms to support neighborhood retrieval.
    for term, docs in term_to_docs.items():
        if len(docs) < 2:
            continue
        for i in range(len(docs) - 1):
            for j in range(i + 1, len(docs)):
                a, b = docs[i], docs[j]
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1.0
                else:
                    graph.add_edge(a, b, weight=1.0, relation=f"shared_term:{term}")

    doc_nodes = [node for node, data in graph.nodes(data=True) if data.get("node_type") == "document"]
    doc_subgraph = graph.subgraph(doc_nodes).copy()

    if doc_subgraph.number_of_nodes() > 0 and doc_subgraph.number_of_edges() > 0:
        communities = community_louvain.best_partition(doc_subgraph, random_state=17)
    else:
        communities = {node: 0 for node in doc_subgraph.nodes()}

    logger.info(
        "Graph built: nodes={} edges={} communities={}",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(set(communities.values())),
    )
    return graph, communities
