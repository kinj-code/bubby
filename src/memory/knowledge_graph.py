"""Lightweight Knowledge Graph for multi-hop entity reasoning.

Builds a directed graph connecting entities extracted from documents
(e.g., Salomon → Corporate Veil → Gilford Motor Co v Horne → Fraud).
Enables the LLM to traverse relationships when answering complex questions.

Uses NetworkX for graph operations with sqlite3 persistence.
RAM: ~15MB for NetworkX + sqlite3 graph structures.
"""

import logging
import sqlite3
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "knowledge" / "graph.db"


@dataclass
class Entity:
    """A knowledge graph entity (node)."""
    name: str
    entity_type: str = "concept"  # concept, case, person, statute, definition
    source_file: str = ""
    description: str = ""
    chunk_index: int = 0
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)


class KnowledgeGraph:
    """
    Local knowledge graph for multi-hop reasoning.

    Entities are extracted from document chunks and linked via
    co-occurrence and explicit relationships. The graph enables:
    - Entity linking: "Salomon" → related concepts
    - Multi-hop queries: "What case relates to the fraud exception?"
    - Context expansion: Retrieve connected entities for richer LLM context
    """

    RELATION_TYPES = [
        "relates_to",        # Generic relationship
        "established_by",    # Principle established by a case
        "exception_to",      # Exception to a rule
        "defined_in",        # Definition found in statute
        "cites",             # One case cites another
        "part_of",           # Entity is part of a broader concept
        "same_topic_as",     # Co-occurrence in same document
    ]

    CO_OCCURRENCE_WINDOW = 200  # Characters within which entities are "related"

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_entities_per_doc: int = 100,
    ) -> None:
        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entities = max_entities_per_doc
        
        # In-memory graph (NetworkX) + sqlite persistence
        self._graph = nx.DiGraph()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_from_db()
        
        self._entities_added = 0
        self._relations_added = 0
        
        logger.info(
            f"KnowledgeGraph initialized "
            f"(entities={self._graph.number_of_nodes()}, "
            f"edges={self._graph.number_of_edges()})"
        )

    def _init_db(self) -> None:
        """Create graph persistence tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                name TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL DEFAULT 'concept',
                source_file TEXT DEFAULT '',
                description TEXT DEFAULT '',
                chunk_index INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5,
                created_at REAL NOT NULL
            );
            
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT 'relates_to',
                weight REAL DEFAULT 1.0,
                source_file TEXT DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY(source) REFERENCES entities(name),
                FOREIGN KEY(target) REFERENCES entities(name)
            );
            
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_relation_source ON relations(source);
            CREATE INDEX IF NOT EXISTS idx_relation_target ON relations(target);
        """)
        self._conn.commit()

    def _load_from_db(self) -> None:
        """Load persisted entities and relations into NetworkX."""
        # Load entities
        for row in self._conn.execute("SELECT * FROM entities"):
            self._graph.add_node(
                row["name"],
                entity_type=row["entity_type"],
                source_file=row["source_file"],
                description=row["description"],
                chunk_index=row["chunk_index"],
                importance=row["importance"],
            )
            self._entities_added += 1

        # Load relations
        for row in self._conn.execute("SELECT * FROM relations"):
            self._graph.add_edge(
                row["source"],
                row["target"],
                relation_type=row["relation_type"],
                weight=row["weight"],
                source_file=row["source_file"],
            )
            self._relations_added += 1

    def add_entity(self, entity: Entity) -> bool:
        """
        Add an entity to the knowledge graph.

        Args:
            entity: Entity to add

        Returns:
            True if added, False if already exists
        """
        if entity.name in self._graph:
            # Update existing
            self._graph.nodes[entity.name].update({
                "description": entity.description or self._graph.nodes[entity.name].get("description", ""),
                "importance": max(entity.importance, self._graph.nodes[entity.name].get("importance", 0)),
            })
            return False

        self._graph.add_node(
            entity.name,
            entity_type=entity.entity_type,
            source_file=entity.source_file,
            description=entity.description,
            chunk_index=entity.chunk_index,
            importance=entity.importance,
        )

        # Persist
        self._conn.execute(
            """INSERT OR REPLACE INTO entities 
               (name, entity_type, source_file, description, chunk_index, importance, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (entity.name, entity.entity_type, entity.source_file,
             entity.description, entity.chunk_index, entity.importance, entity.created_at)
        )
        self._conn.commit()

        self._entities_added += 1
        return True

    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: str = "relates_to",
        weight: float = 1.0,
        source_file: str = "",
    ) -> bool:
        """
        Add a relationship between two entities.

        Args:
            source: Source entity name
            target: Target entity name
            relation_type: Type of relationship
            weight: Edge weight (higher = stronger)
            source_file: Source document

        Returns:
            True if relation was added
        """
        if not source or not target:
            return False

        # Ensure both entities exist
        if source not in self._graph:
            self.add_entity(Entity(name=source, source_file=source_file))
        if target not in self._graph:
            self.add_entity(Entity(name=target, source_file=source_file))

        # Normalize relation type
        if relation_type not in self.RELATION_TYPES:
            relation_type = "relates_to"

        # Add edge
        if self._graph.has_edge(source, target):
            # Increment weight
            current = self._graph[source][target].get("weight", 1.0)
            self._graph[source][target]["weight"] = current + 0.1
        else:
            self._graph.add_edge(
                source, target,
                relation_type=relation_type,
                weight=weight,
                source_file=source_file,
            )

        # Persist
        self._conn.execute(
            """INSERT INTO relations (source, target, relation_type, weight, source_file, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (source, target, relation_type, weight, source_file, time.time())
        )
        self._conn.commit()

        self._relations_added += 1
        return True

    def extract_co_occurrences(
        self,
        text: str,
        entities: List[str],
        source_file: str = "",
    ) -> int:
        """
        Extract co-occurrence relationships from text.
        
        Two entities are related if they appear within
        CO_OCCURRENCE_WINDOW characters of each other.

        Args:
            text: Document text to analyze
            entities: List of entity names extracted from the text
            source_file: Source file

        Returns:
            Number of new relations created
        """
        added = 0
        text_lower = text.lower()

        for i, entity_a in enumerate(entities):
            pos_a = text_lower.find(entity_a.lower())
            if pos_a < 0:
                continue

            for entity_b in entities[i + 1:]:
                pos_b = text_lower.find(entity_b.lower())
                if pos_b < 0:
                    continue

                if abs(pos_b - pos_a) <= self.CO_OCCURRENCE_WINDOW:
                    if self.add_relation(
                        entity_a, entity_b,
                        relation_type="same_topic_as",
                        source_file=source_file,
                    ):
                        added += 1

        return added

    def query_entity(self, name: str, max_depth: int = 1) -> Dict[str, Any]:
        """
        Query an entity and its immediate neighborhood.

        Args:
            name: Entity name to search
            max_depth: How many hops to traverse (default 1)

        Returns:
            Dict with entity info and related entities
        """
        name_lower = name.lower()

        # Fuzzy match — check if any entity contains the search term
        matches = [
            n for n in self._graph.nodes()
            if name_lower in n.lower()
        ]
        
        if not matches:
            return {"entity": name, "found": False, "related": []}

        best_match = matches[0]
        node_data = dict(self._graph.nodes[best_match])

        # Get neighbors within depth
        related = []
        for target in self._graph.successors(best_match):
            edge_data = self._graph[best_match][target]
            related.append({
                "entity": target,
                "relation": edge_data.get("relation_type", "relates_to"),
                "weight": edge_data.get("weight", 1.0),
            })

        # Also get reverse relations
        for source in self._graph.predecessors(best_match):
            edge_data = self._graph[source][best_match]
            related.append({
                "entity": source,
                "relation": edge_data.get("relation_type", "relates_to"),
                "weight": edge_data.get("weight", 1.0),
            })

        return {
            "entity": best_match,
            "found": True,
            "info": node_data,
            "related": related[:10],  # Cap at 10
        }

    def query_path(self, source: str, target: str, max_depth: int = 3) -> Optional[List[str]]:
        """
        Find the shortest path between two entities.

        Args:
            source: Starting entity
            target: Target entity
            max_depth: Maximum path length

        Returns:
            List of entity names forming the path, or None
        """
        try:
            # Convert to undirected for path finding
            undirected = self._graph.to_undirected()
            path = nx.shortest_path(undirected, source=source, target=target)
            if len(path) <= max_depth + 1:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        return None

    def get_related_context(self, query: str, max_entities: int = 5) -> str:
        """
        Build a context string of related entities for LLM prompts.

        Searches the graph for entities matching the query
        and returns formatted context.

        Args:
            query: Search text
            max_entities: Maximum entities to return

        Returns:
            Formatted context string
        """
        result = self.query_entity(query)
        if not result["found"]:
            return ""
        
        lines = [f"[KNOWLEDGE GRAPH: {result['entity']}]"]
        if result.get("info", {}).get("description"):
            lines.append(f"  {result['info']['description']}")
        
        if result.get("related"):
            lines.append("  Related concepts:")
            for r in result["related"][:max_entities]:
                relation_label = r["relation"].replace("_", " ")
                lines.append(f"    - {r['entity']} ({relation_label})")
        
        return "\n".join(lines)

    def get_entity_types(self) -> Dict[str, int]:
        """Get count of entities by type."""
        counts = defaultdict(int)
        for node in self._graph.nodes():
            etype = self._graph.nodes[node].get("entity_type", "concept")
            counts[etype] += 1
        return dict(counts)

    def get_central_entities(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get most central entities by degree centrality."""
        if self._graph.number_of_nodes() == 0:
            return []
        centrality = nx.degree_centrality(self._graph)
        sorted_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities[:top_n]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "entities": self._graph.number_of_nodes(),
            "relations": self._graph.number_of_edges(),
            "entity_types": self.get_entity_types(),
            "central_entities": self.get_central_entities(3),
            "db_path": str(self._db_path),
        }

    def clear(self) -> None:
        """Clear all graph data."""
        self._graph.clear()
        self._conn.execute("DELETE FROM relations")
        self._conn.execute("DELETE FROM entities")
        self._conn.commit()
        self._entities_added = 0
        self._relations_added = 0
        logger.info("Knowledge graph cleared")

    def close(self) -> None:
        """Close database and release resources."""
        self._conn.close()


# Testing helper
if __name__ == "__main__":
    import tempfile
    import shutil

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("KNOWLEDGE GRAPH TEST")
    logger.info("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_graph.db"
    kg = KnowledgeGraph(db_path=db_path)

    # Test 1: Add entities
    kg.add_entity(Entity(
        name="Salomon v Salomon",
        entity_type="case",
        description="Landmark company law case establishing separate legal personality [1897]",
        source_file="company_law.txt",
        importance=0.95,
    ))
    kg.add_entity(Entity(
        name="Corporate Veil",
        entity_type="concept",
        description="Legal concept separating company from shareholders",
        source_file="company_law.txt",
    ))
    kg.add_entity(Entity(
        name="Gilford Motor Co v Horne",
        entity_type="case",
        description="Fraud exception case where corporate veil was pierced [1933]",
        source_file="company_law.txt",
    ))
    assert kg._graph.number_of_nodes() == 3
    logger.info(f"✓ Entities added: {kg._graph.number_of_nodes()}")

    # Test 2: Add relations
    kg.add_relation("Salomon v Salomon", "Corporate Veil", "established_by")
    kg.add_relation("Corporate Veil", "Gilford Motor Co v Horne", "exception_to")
    assert kg._graph.number_of_edges() == 2
    logger.info(f"✓ Relations added: {kg._graph.number_of_edges()}")

    # Test 3: Query entity
    result = kg.query_entity("Salomon")
    assert result["found"]
    assert len(result["related"]) >= 1
    # Should find Corporate Veil as related
    related_names = [r["entity"] for r in result["related"]]
    assert "Corporate Veil" in related_names
    logger.info(f"✓ Entity query: {len(result['related'])} related entities")

    # Test 4: Path finding
    path = kg.query_path("Salomon v Salomon", "Gilford Motor Co v Horne")
    assert path is not None
    assert "Salomon" in path[0]
    assert "Horne" in path[-1]
    logger.info(f"✓ Path: {' → '.join(path)}")

    # Test 5: Related context for LLM
    context = kg.get_related_context("Salomon")
    assert "[KNOWLEDGE GRAPH:" in context
    assert "Corporate Veil" in context
    logger.info(f"✓ Context: {len(context)} chars")

    # Test 6: Central entities
    central = kg.get_central_entities(3)
    assert len(central) >= 2
    # Corporate Veil connects to both → most central
    logger.info(f"✓ Central entities: {[(n, round(s, 3)) for n, s in central]}")

    # Test 7: Co-occurrence extraction
    text = "The Salomon case established piercing the corporate veil."
    added = kg.extract_co_occurrences(
        text,
        ["Salomon v Salomon", "Corporate Veil", "Piercing"],
        "company_law.txt",
    )
    logger.info(f"✓ Co-occurrence: {added} new relations from text")

    # Test 8: Stats
    stats = kg.get_stats()
    assert stats["entities"] >= 5
    logger.info(f"✓ Stats: {stats}")

    kg.close()
    shutil.rmtree(tmpdir)
    logger.info("\nALL KNOWLEDGE GRAPH TESTS PASSED")