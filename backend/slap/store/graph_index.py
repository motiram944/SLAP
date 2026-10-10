"""Entity-Relation Adjacency Graph Index for Topological Pruning.

Maintains continuous hypergraph connectivity linking domain entities, predicates,
and envelope document IDs. Enables topological candidate pruning before ANN retrieval.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Any
from slap.core.schema import HyperEdge


class TopologicalGraphIndex:
    """In-memory entity-relation adjacency graph linking entities to document IDs."""

    def __init__(self):
        # entity -> set of envelope IDs
        self.entity_to_docs: Dict[str, Set[str]] = defaultdict(set)
        # envelope_id -> list of HyperEdges
        self.doc_to_edges: Dict[str, List[HyperEdge]] = defaultdict(list)
        # entity adjacency: entity -> set of (relation, target_entity)
        self.adj: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    def add_envelope(self, envelope_id: str, edges: List[HyperEdge]) -> None:
        """Register an envelope and its hypergraph edges into the adjacency index."""
        self.doc_to_edges[envelope_id].extend(edges)
        for edge in edges:
            src_norm = edge.source.strip().lower()
            tgt_norm = edge.target.strip().lower()
            rel_norm = edge.relation.strip().lower()

            self.entity_to_docs[src_norm].add(envelope_id)
            self.entity_to_docs[tgt_norm].add(envelope_id)

            self.adj[src_norm].add((rel_norm, tgt_norm))
            self.adj[tgt_norm].add((f"rev_{rel_norm}", src_norm))

    def prune_candidates(self, query_entities: List[str], max_hops: int = 1) -> Set[str]:
        """Filters candidate document IDs topologically based on entity adjacency.
        If query_entities is empty, returns all registered envelope IDs (no pruning).
        Otherwise, returns document IDs linked to query entities or their n-hop neighbors.
        """
        if not query_entities:
            return set(self.doc_to_edges.keys())

        matched_docs: Set[str] = set()
        active_entities: Set[str] = {e.strip().lower() for e in query_entities if e.strip()}

        current_frontier = set(active_entities)
        visited_entities = set(active_entities)

        # Collect documents directly matching active entities
        for ent in active_entities:
            if ent in self.entity_to_docs:
                matched_docs.update(self.entity_to_docs[ent])

        # Topological expansion up to max_hops
        for _ in range(max_hops):
            next_frontier = set()
            for ent in current_frontier:
                for rel, neighbor in self.adj.get(ent, set()):
                    if neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        next_frontier.add(neighbor)
                        if neighbor in self.entity_to_docs:
                            matched_docs.update(self.entity_to_docs[neighbor])
            current_frontier = next_frontier

        return matched_docs

    def get_document_edges(self, envelope_id: str) -> List[HyperEdge]:
        """Retrieve all registered hyperedges for a given envelope."""
        return self.doc_to_edges.get(envelope_id, [])

    def get_subgraph(self, envelope_ids: List[str]) -> Dict[str, Any]:
        """Export serialized subgraph representation for visualization in Next.js UI."""
        nodes = set()
        links = []
        doc_nodes = []

        for doc_id in envelope_ids:
            doc_nodes.append({"id": f"doc_{doc_id[:8]}", "type": "document", "label": doc_id[:8]})
            edges = self.doc_to_edges.get(doc_id, [])
            for edge in edges:
                nodes.add(edge.source)
                nodes.add(edge.target)
                links.append({
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "confidence": edge.confidence,
                    "docId": doc_id,
                })

        return {
            "entityNodes": [{"id": n, "type": "entity", "label": n} for n in sorted(nodes)],
            "documentNodes": doc_nodes,
            "edges": links,
            "totalEntities": len(nodes),
            "totalEdges": len(links),
        }

    def stats(self) -> Dict[str, Any]:
        """Return graph index metrics."""
        return {
            "total_documents": len(self.doc_to_edges),
            "unique_entities": len(self.entity_to_docs),
            "total_adjacency_relations": sum(len(v) for v in self.adj.values()),
        }

    def clear(self) -> None:
        """Clear all in-memory graph entities and doc edges."""
        self.doc_to_edges.clear()
        self.entity_to_docs.clear()
        self.adj.clear()

