"""GraphRAG local/global retrieval over document-entity graph."""

from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx


def _query_terms(query: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query)}


def _document_terms(graph: nx.Graph, doc_id: str) -> set[str]:
    terms = graph.nodes[doc_id].get("term_set")
    if isinstance(terms, set):
        return terms
    if isinstance(terms, list):
        term_set = set(str(item) for item in terms)
        graph.nodes[doc_id]["term_set"] = term_set
        return term_set

    term_set = _query_terms(str(graph.nodes[doc_id].get("text", "")))
    graph.nodes[doc_id]["term_set"] = term_set
    return term_set


def local_graph_retrieval(
    graph: nx.Graph,
    dense_hits: list[tuple[str, float]],
    hops: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """Expand dense seeds with graph-neighbor evidence for local retrieval."""
    scores: dict[str, float] = defaultdict(float)

    for doc_id, dense_score in dense_hits:
        scores[doc_id] += dense_score

        frontier = {doc_id}
        visited = {doc_id}
        decay = 1.0
        for _ in range(hops):
            decay *= 0.65
            next_frontier = set()
            for node in frontier:
                for neighbor in graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)

                    node_type = graph.nodes[neighbor].get("node_type")
                    if node_type == "document":
                        scores[neighbor] += dense_score * decay
                    elif node_type == "term":
                        for doc_neighbor in graph.neighbors(neighbor):
                            if graph.nodes[doc_neighbor].get("node_type") == "document":
                                scores[doc_neighbor] += dense_score * (decay * 0.8)
            frontier = next_frontier

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def global_community_retrieval(
    graph: nx.Graph,
    communities: dict[str, int],
    query: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """Retrieve by scoring communities first, then ranking documents globally."""
    query_vocab = _query_terms(query)

    community_docs: dict[int, list[str]] = defaultdict(list)
    for doc_id, community_id in communities.items():
        community_docs[int(community_id)].append(doc_id)

    community_scores: dict[int, float] = defaultdict(float)
    for community_id, doc_ids in community_docs.items():
        score = 0.0
        for doc_id in doc_ids:
            text_terms = _document_terms(graph, doc_id)
            if not text_terms:
                continue
            overlap = len(query_vocab.intersection(text_terms))
            score += overlap / float(len(query_vocab) + 1e-6)
        community_scores[community_id] = score / float(max(1, len(doc_ids)))

    top_communities = sorted(community_scores.items(), key=lambda item: item[1], reverse=True)[:3]

    doc_scores: dict[str, float] = {}
    for community_id, c_score in top_communities:
        for doc_id in community_docs[community_id]:
            term_overlap = len(query_vocab.intersection(_document_terms(graph, doc_id)))
            doc_scores[doc_id] = c_score + (term_overlap * 0.05)

    ranked = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_k]
