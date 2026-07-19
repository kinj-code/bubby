"""Knowledge Graph Builder — populates the graph from high-fidelity RAG chunks.

Reads chunks that scored above fidelity threshold in the automated evaluator,
extracts entities (case names, concepts, statutes, definitions) and builds
edges via co-occurrence and explicit relationship patterns.

RAM: Streaming extraction, ~5MB overhead.
"""

import logging
import re
from typing import Optional, Dict, Any, List, Set, Tuple
from collections import Counter

from src.memory.knowledge_graph import KnowledgeGraph, Entity

logger = logging.getLogger(__name__)

# Legal case law extractors
CASE_PATTERN = re.compile(
    r'\b((?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\s+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+&?\s*(?:Co|Ltd|Corporation|LLC|PLC))?(?:\s+\[\d{4}\])?)'
)
STATUTE_PATTERN = re.compile(r'((?:[A-Z][a-z]+\s+){1,4}Act\s+\d{4})')
CONCEPT_PATTERN = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,})\b')
DEFINITION_PATTERN = re.compile(r'(?:is|are|means|refers to)\s+(?:a|an|the)?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})')

# Relationship extraction patterns
RELATION_PATTERNS = [
    (re.compile(r'([A-Z][\w\s]+)\s+established\s+([A-Z][\w\s]+)', re.I), 'established_by'),
    (re.compile(r'([A-Z][\w\s]+)\s+exception\s+(?:to|of)\s+([A-Z][\w\s]+)', re.I), 'exception_to'),
    (re.compile(r'([A-Z][\w\s]+)\s+defined\s+(?:in|by|under)\s+([A-Z][\w\s]+)', re.I), 'defined_in'),
    (re.compile(r'([A-Z][\w\s]+)\s+cites\s+([A-Z][\w\s]+)', re.I), 'cites'),
    (re.compile(r'([A-Z][\w\s]+)\s+(?:is|are)\s+(?:a|an)\s+([A-Z][\w\s]+)', re.I), 'part_of'),
]

SENTENCE_PATTERN = re.compile(r'([^.!?]+(?:[.!?]|$))')


class GraphBuilder:
    """
    Builds the Knowledge Graph from high-fidelity chunks.

    Flow:
    1. Retrieve chunks with fidelity > threshold from Evaluator
    2. Extract entities (cases, concepts, statutes)
    3. Extract relationships from text patterns
    4. Add to KnowledgeGraph + co-occurrence links
    """

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        self._kg = knowledge_graph
        self._entities_extracted = 0
        self._relations_built = 0
        logger.info(f"GraphBuilder initialized (kg={knowledge_graph._graph.number_of_nodes()} nodes)")

    def build_from_chunks(
        self,
        high_fidelity_chunks: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Build graph nodes and edges from high-fidelity chunks.

        Args:
            high_fidelity_chunks: List of {source_file, chunk_text, utility, metadata}

        Returns:
            Dict with {entities_added, relations_added} counts
        """
        entities_before = self._kg._graph.number_of_nodes()
        relations_before = self._kg._graph.number_of_edges()

        for chunk_data in high_fidelity_chunks:
            text = chunk_data.get("chunk_text", "")
            source_file = chunk_data.get("source_file", "unknown")

            if not text or len(text) < 50:
                continue

            # Extract and add entities
            entities = self._extract_entities(text)
            entity_names = []
            for e in entities:
                self._kg.add_entity(e)
                entity_names.append(e.name)
                self._entities_extracted += 1

            # Extract explicit relationships from text
            relations = self._extract_relations(text, source_file)
            for rel_type, source, target in relations:
                self._kg.add_relation(source, target, rel_type, source_file=source_file)
                self._relations_built += 1

            # Add co-occurrence links between entities in same chunk
            if entity_names:
                self._kg.extract_co_occurrences(text, entity_names, source_file)

        entities_added = self._kg._graph.number_of_nodes() - entities_before
        relations_added = self._kg._graph.number_of_edges() - relations_before

        logger.info(
            f"Graph built: +{entities_added} entities, +{relations_added} relations "
            f"(total: {self._kg._graph.number_of_nodes()} nodes, {self._kg._graph.number_of_edges()} edges)"
        )

        return {
            "entities_added": entities_added,
            "relations_added": relations_added,
            "total_nodes": self._kg._graph.number_of_nodes(),
            "total_edges": self._kg._graph.number_of_edges(),
        }

    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract all entity types from text."""
        entities = []
        seen = set()

        # Case law
        for match in CASE_PATTERN.finditer(text):
            name = match.group(1).strip().rstrip(".,;:!?'\"")
            if len(name) > 5 and name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(
                    name=name, entity_type="case",
                    description="Landmark legal case",
                ))

        # Statutes
        for match in STATUTE_PATTERN.finditer(text):
            name = match.group(1).strip().rstrip(".,;:!?'\"")
            if len(name) > 5 and name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(
                    name=name, entity_type="statute",
                    description="Legislative act",
                ))

        # Capitalized concepts (4+ words likely to be named concepts)
        skip_words = {"The", "This", "That", "These", "Those", "When", "Where", "Which", "Each", "There", "Their"}
        for match in CONCEPT_PATTERN.finditer(text):
            name = match.group(1).strip().rstrip(".,;:!?'\"")
            if len(name) > 10 and name.lower() not in seen and name.split()[0] not in skip_words:
                seen.add(name.lower())
                entities.append(Entity(
                    name=name, entity_type="concept",
                    description="Key academic concept",
                ))

        # Definitions
        for match in DEFINITION_PATTERN.finditer(text):
            name = match.group(1).strip().rstrip(".,;:!?'\"")
            if len(name) > 5 and name.lower() not in seen:
                seen.add(name.lower())
                entities.append(Entity(
                    name=name, entity_type="definition",
                    description=f"Defined term: {name}",
                ))

        return entities

    def _extract_relations(self, text: str, source_file: str) -> List[Tuple[str, str, str]]:
        """Extract explicit relationships from text. Returns [(relation_type, source, target)]."""
        relations = []

        for pattern, rel_type in RELATION_PATTERNS:
            for match in pattern.finditer(text):
                source = match.group(1).strip().rstrip(".,;:!?'\"")
                target = match.group(2).strip().rstrip(".,;:!?'\"")
                if len(source) > 3 and len(target) > 3 and source.lower() != target.lower():
                    relations.append((rel_type, source, target))

        return relations

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entities_extracted": self._entities_extracted,
            "relations_built": self._relations_built,
            "kg_stats": self._kg.get_stats(),
        }


# Testing helper
if __name__ == "__main__":
    import tempfile
    import shutil
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    logger.info("=" * 60)
    logger.info("GRAPH BUILDER TEST")
    logger.info("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_graph.db"
    kg = KnowledgeGraph(db_path=db_path)
    builder = GraphBuilder(kg)

    # Simulate high-fidelity chunks
    chunks = [
        {
            "source_file": "company_law.txt",
            "chunk_text": (
                "The principle of Separate Legal Personality was established "
                "in Salomon v Salomon & Co Ltd [1897] AC 22. A company is a "
                "separate legal entity from its shareholders. The Companies Act "
                "2006 provides statutory exceptions. One exception to the rule "
                "is Fraud, where courts may pierce the corporate veil. "
                "Gilford Motor Co v Horne [1933] applied the fraud exception."
            ),
            "utility": 0.85,
        },
        {
            "source_file": "business_stats.txt",
            "chunk_text": (
                "Business Statistics uses three Measures of Central Tendency. "
                "The Mean is defined as the arithmetic average of a dataset. "
                "The Median is the middle value when data is ordered. The Mode "
                "is the most frequently occurring value. Standard Deviation "
                "measures the spread of data around the mean."
            ),
            "utility": 0.90,
        },
    ]

    # Build graph
    result = builder.build_from_chunks(chunks)
    assert result["entities_added"] >= 5, f"Expected >=5 entities, got {result['entities_added']}"
    assert result["relations_added"] >= 2, f"Expected >=2 relations, got {result['relations_added']}"
    logger.info(f"✓ Built: +{result['entities_added']} entities, +{result['relations_added']} relations")

    # Verify entities
    stats = kg.get_stats()
    assert stats["entities"] >= 5
    logger.info(f"✓ Graph stats: {stats['entities']} nodes, {stats['relations']} edges, types={stats['entity_types']}")

    # Verify specific entities
    assert kg.query_entity("Salomon")["found"]
    result = kg.query_entity("Separate Legal Personality")
    assert result["found"]
    logger.info(f"✓ Entity queries work")

    # Stats
    builder_stats = builder.get_stats()
    logger.info(f"✓ Builder stats: {builder_stats}")

    kg.close()
    shutil.rmtree(tmpdir)
    logger.info("\nALL GRAPH BUILDER TESTS PASSED")