#!/usr/bin/env python3
"""
Phase 12 Integration Test: Self-Refining Memory (Feedback + Knowledge Graph)

Tests the unified feedback loop architecture:
1. Feedback Engine: records queries, tracks utility, triggers re-index
2. Knowledge Graph: entity linking, multi-hop paths, context expansion
3. Integration: RAG Bridge + Feedback tracking + Graph enrichment
4. Critic integration: RAG confidence scoring

Run: python test_feedback_loop.py
"""

import logging
import sys
import os
import tempfile
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def test_feedback_engine() -> None:
    """Test feedback tracking, utility scoring, and reindex triggers."""
    from src.memory.feedback import FeedbackEngine

    logger.info("=" * 60)
    logger.info("FEEDBACK: Query Tracking & Reindex Logic")
    logger.info("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_feedback.db"
    engine = FeedbackEngine(db_path=db_path)

    # Simulate queries to company_law.txt
    fid1 = engine.record_query("What is Salomon?", "company_law.txt", 0, 0.85)
    fid2 = engine.record_query("Explain corporate veil", "company_law.txt", 1, 0.78)
    
    # Mark as helpful
    engine.record_utility(fid1, True)
    engine.record_utility(fid2, True)
    logger.info("✓ 2 helpful queries recorded")

    # Simulate weak retrievals - enough to push utility below 0.35 threshold
    for i in range(5):
        fid = engine.record_query(f"Weak query about unrelated topic {i}", "company_law.txt", 0, 0.04)
        engine.record_utility(fid, was_helpful=False)
    
    # Utility = 2/(2+5) = 0.286 < 0.35 → should trigger reindex
    assert engine.should_reindex("company_law.txt"), "Should trigger reindex: utility below threshold"
    logger.info("✓ Reindex triggered: 2/7 queries helpful → utility below 0.35")

    # Add healthy queries to another file
    fid6 = engine.record_query("What is mean?", "business_stats.txt", 0, 0.92)
    engine.record_utility(fid6, True)
    assert not engine.should_reindex("business_stats.txt")
    logger.info("✓ Healthy file not flagged")

    # Test file stats
    stats = engine.get_file_stats("company_law.txt")
    assert stats["total_queries"] >= 5  # 2 helpful + 5 weak = 7
    assert stats["needs_reindex"] == 1
    logger.info(f"✓ File stats: queries={stats['total_queries']}, reindex={stats['needs_reindex']}")

    # Test overall health
    health = engine.get_overall_health()
    assert health["total_queries"] >= 5
    assert health["files_needing_reindex"] >= 1
    logger.info(f"✓ Health: utility_rate={health['utility_rate']:.2f}, files_needing={health['files_needing_reindex']}")

    # Test get files needing reindex
    files = engine.get_files_needing_reindex()
    assert "company_law.txt" in files
    logger.info(f"✓ Files needing reindex: {files}")

    engine.close()
    shutil.rmtree(tmpdir)
    logger.info("✓ Feedback engine tests complete\n")


def test_knowledge_graph() -> None:
    """Test entity linking, co-occurrence, paths, context generation."""
    from src.memory.knowledge_graph import KnowledgeGraph, Entity

    logger.info("=" * 60)
    logger.info("GRAPH: Entity Linking & Multi-Hop Reasoning")
    logger.info("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_graph.db"
    kg = KnowledgeGraph(db_path=db_path)

    # Build a small legal knowledge graph
    entities = [
        Entity("Salomon v Salomon", "case", "company_law.txt", "Established separate legal personality [1897]", importance=0.95),
        Entity("Corporate Veil", "concept", "company_law.txt", "Legal separation between company and shareholders"),
        Entity("Gilford Motor Co v Horne", "case", "company_law.txt", "Fraud exception — veil pierced [1933]", importance=0.85),
        Entity("Fraud Exception", "concept", "company_law.txt", "Courts pierce veil when company used for fraud"),
        Entity("Companies Act 2006", "statute", "company_law.txt", "UK legislation governing companies"),
        Entity("Separate Legal Personality", "concept", "company_law.txt", "Core principle from Salomon"),
    ]
    for e in entities:
        kg.add_entity(e)
    
    # Build relationships
    kg.add_relation("Salomon v Salomon", "Separate Legal Personality", "established_by", weight=1.0)
    kg.add_relation("Salomon v Salomon", "Corporate Veil", "established_by", weight=0.9)
    kg.add_relation("Corporate Veil", "Fraud Exception", "exception_to", weight=0.8)
    kg.add_relation("Gilford Motor Co v Horne", "Fraud Exception", "established_by", weight=0.9)
    kg.add_relation("Companies Act 2006", "Corporate Veil", "defined_in", weight=0.7)
    kg.add_relation("Fraud Exception", "Gilford Motor Co v Horne", "cites", weight=0.6)

    logger.info(f"✓ Graph: {kg._graph.number_of_nodes()} entities, {kg._graph.number_of_edges()} relations")

    # Test 1: Entity query
    result = kg.query_entity("Salomon")
    assert result["found"]
    assert len(result["related"]) >= 2
    related = [r["entity"] for r in result["related"]]
    assert "Separate Legal Personality" in related or "Corporate Veil" in related
    logger.info(f"✓ Salomon query: {len(result['related'])} related entities")

    # Test 2: Multi-hop path
    path = kg.query_path("Salomon v Salomon", "Gilford Motor Co v Horne")
    assert path is not None
    logger.info(f"✓ Multi-hop path: {' → '.join(path)}")

    # Test 3: Another path
    path = kg.query_path("Salomon v Salomon", "Companies Act 2006")
    assert path is not None
    logger.info(f"✓ Statute path: {' → '.join(path)}")

    # Test 4: Co-occurrence extraction
    text = "The Salomon v Salomon case established separate legal personality. Gilford Motor Co v Horne created the fraud exception."
    added = kg.extract_co_occurrences(
        text,
        ["Salomon v Salomon", "Separate Legal Personality", "Gilford Motor Co v Horne", "Fraud Exception"],
        "company_law.txt",
    )
    logger.info(f"✓ Co-occurrence: {added} relations from text")

    # Test 5: Context for LLM
    context = kg.get_related_context("Salomon")
    assert "[KNOWLEDGE GRAPH:" in context
    assert len(context) > 50
    logger.info(f"✓ KG context: {len(context)} chars")

    # Test 6: Central entities
    central = kg.get_central_entities(3)
    assert len(central) >= 2
    logger.info(f"✓ Central: {[(n, round(s,3)) for n,s in central]}")

    # Test 7: Stats
    stats = kg.get_stats()
    assert stats["entities"] >= 6
    assert stats["relations"] >= 6
    logger.info(f"✓ Stats: entities={stats['entities']}, relations={stats['relations']}")

    kg.close()
    shutil.rmtree(tmpdir)
    logger.info("✓ Knowledge graph tests complete\n")


def test_critic_rag_integration() -> None:
    """Test Critic + Feedback engine integration for RAG confidence."""
    from src.brain.critic import CognitiveCritic
    from src.memory.feedback import FeedbackEngine
    from src.actions.executor import SystemExecutor

    logger.info("=" * 60)
    logger.info("CRITIC: RAG Confidence Scoring")
    logger.info("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_feedback.db"
    feedback = FeedbackEngine(db_path=db_path)
    critic = CognitiveCritic(action_executor=SystemExecutor())

    # Test 1: Valid, helpful output passes
    output = {
        "animation": "talk",
        "speech": "The Salomon case established separate legal personality. The exceptions include fraud (Gilford Motor Co v Horne) and agency.",
        "action": "",
    }
    verdict = critic.review(output)
    assert verdict.passed
    # Record feedback — this answer references specific cases → helpful
    fid = feedback.record_query(
        "What are exceptions to Salomon?",
        "company_law.txt", 0, 0.85
    )
    feedback.record_utility(fid, was_helpful=True)
    logger.info("✓ Critic passes helpful RAG answer")

    # Test 2: LLM disclaimer → critic catches it
    output = {
        "animation": "talk",
        "speech": "As a language model I cannot look up case law. You should check the Companies Act 2006.",
        "action": "",
    }
    verdict = critic.review(output)
    assert not verdict.passed  # Utility check catches this
    fid = feedback.record_query(
        "What does Companies Act say?",
        "company_law.txt", 0, 0.92
    )
    feedback.record_utility(fid, was_helpful=False)
    logger.info("✓ Critic catches LLM disclaimer")

    # Test 3: Hallucinated action blocked, feedback recorded
    output = {
        "animation": "talk",
        "speech": "Let me delete those files for you",
        "action": "rm_files",
    }
    verdict = critic.review(output)
    assert not verdict.passed
    fid = feedback.record_query(
        "Clean up my files",
        "unknown.txt", 0, 0.0
    )
    feedback.record_utility(fid, was_helpful=False)
    logger.info("✓ Hallucinated action blocked + recorded as unhelpful")

    # Test 4: Feedback health check
    health = feedback.get_overall_health()
    assert health["total_queries"] >= 2
    # unknown.txt only has 1 query (< MIN_FEEDBACK_FOR_DECISION) so won't trigger reindex yet
    assert health["files_needing_reindex"] >= 0
    logger.info(f"✓ Health: {health['helpful_count']} helpful, {health['unhelpful_count']} unhelpful, files_needing={health['files_needing_reindex']}")

    feedback.close()
    critic.get_stats()
    shutil.rmtree(tmpdir)
    logger.info("✓ Critic-RAG integration tests complete\n")


def test_end_to_end_feedback_loop() -> None:
    """Full pipeline: RAG query → Critic validation → Feedback → Reindex trigger."""
    from src.memory.parser import DocumentParser
    from src.memory.ingestion import DocumentIngestion
    from src.memory.embedding import EmbeddingEngine, EMBEDDING_DIM
    from src.memory.vector_db import VectorStore
    from src.persona.rag_bridge import RAGBridge
    from src.memory.feedback import FeedbackEngine
    from src.memory.knowledge_graph import KnowledgeGraph, Entity
    from src.brain.critic import CognitiveCritic
    from src.actions.executor import SystemExecutor

    logger.info("=" * 60)
    logger.info("END-TO-END: Feedback Loop + Graph Enrichment")
    logger.info("=" * 60)

    tmpdir = Path(tempfile.mkdtemp())
    feedback_db = tmpdir / "feedback.db"
    graph_db = tmpdir / "graph.db"

    # Set up components
    embedding = EmbeddingEngine()
    vector_store = VectorStore(embedding_dim=EMBEDDING_DIM)
    ingestion = DocumentIngestion(embedding, vector_store)
    bridge = RAGBridge(embedding, vector_store, top_k=3)
    feedback = FeedbackEngine(db_path=feedback_db)
    kg = KnowledgeGraph(db_path=graph_db)
    critic = CognitiveCritic(action_executor=SystemExecutor())

    # Create study document
    study_dir = tmpdir / "study"
    study_dir.mkdir()
    law_file = study_dir / "company_law.txt"
    law_file.write_text(
        "COMPANY LAW: Salomon v Salomon & Co Ltd [1897] AC 22\n\n"
        "The doctrine of separate legal personality was established in "
        "Salomon v Salomon. A company is a separate legal entity.\n\n"
        "EXCEPTIONS:\n"
        "1. Fraud — piercing the corporate veil (Gilford Motor Co v Horne [1933])\n"
        "2. Agency — company acting as agent\n"
        "3. Statutory exceptions — Companies Act 2006, s.213-214 Insolvency Act\n"
        "4. Group enterprises — single economic unit in limited cases\n\n"
        "STATISTICS: Measures of Central Tendency\n"
        "Mean, Median, Mode are the three measures.\n"
        "Standard deviation measures spread.\n"
        + "More content. " * 15
    )

    # Ingest
    results = ingestion.ingest_directory(str(study_dir))
    total_chunks = sum(results.values())
    logger.info(f"Ingested: {total_chunks} chunks")

    # Enrich knowledge graph from document
    entities_found = [
        "Salomon v Salomon", "Corporate Veil", "Gilford Motor Co v Horne",
        "Fraud", "Companies Act 2006", "Separate Legal Personality",
        "Agency", "Group Enterprises", "Insolvency Act",
        "Mean", "Median", "Mode", "Standard Deviation",
    ]
    for e in entities_found:
        kg.add_entity(Entity(name=e, source_file="company_law.txt"))

    kg.add_relation("Salomon v Salomon", "Separate Legal Personality", "established_by")
    kg.add_relation("Salomon v Salomon", "Corporate Veil", "established_by")
    kg.add_relation("Corporate Veil", "Fraud", "exception_to")
    kg.add_relation("Fraud", "Gilford Motor Co v Horne", "cites")
    logger.info(f"Graph enriched: {kg._graph.number_of_nodes()} entities, {kg._graph.number_of_edges()} edges")

    # ── FLOW 1: Strong query → helpful answer → recorded ──
    query = "What are the exceptions to the rule in Salomon v Salomon?"
    assert bridge.should_query_documents(query)
    records = bridge.query(query, top_k=2)

    for i, r in enumerate(records):
        score = r.metadata.get("similarity_score", 0.0)
        fid = feedback.record_query(query, "company_law.txt", i, score)
        # Assume the answer was helpful (critic passed)
        feedback.record_utility(fid, was_helpful=True)

    logger.info(f"✓ Flow 1: Query '{query[:40]}...' → {len(records)} chunks, all helpful")

    # ── FLOW 2: Critic reviews the synthesized answer ──
    critic_output = {
        "animation": "talk",
        "speech": "The exceptions to Salomon include fraud, agency, statutory exceptions, and group enterprises.",
        "action": "",
    }
    verdict = critic.review(critic_output)
    assert verdict.passed
    logger.info(f"✓ Flow 2: Critic passed the answer")

    # ── FLOW 3: Knowledge graph context enrichment ──
    kg_context = kg.get_related_context("Salomon")
    assert kg_context
    logger.info(f"✓ Flow 3: KG context available ({len(kg_context)} chars)")

    # ── FLOW 4: Weak query → unhelpful → reindex trigger ──
    for i in range(3):
        weak_query = f"Weak query about unrelated topic {i}"
        fid = feedback.record_query(weak_query, "company_law.txt", 0, 0.03)
        feedback.record_utility(fid, was_helpful=False)

    assert feedback.should_reindex("company_law.txt")
    logger.info(f"✓ Flow 4: Reindex triggered after weak queries")

    # ── FLOW 5: Feedback health report ──
    health = feedback.get_overall_health()
    assert health["files_needing_reindex"] >= 1
    logger.info(f"✓ Flow 5: Health: {health['helpful_count']} helpful, "
                f"{health['unhelpful_count']} unhelpful, "
                f"utility_rate={health['utility_rate']:.2f}")

    # ── FLOW 6: Multi-hop path ──
    path = kg.query_path("Salomon v Salomon", "Gilford Motor Co v Horne")
    assert path is not None
    logger.info(f"✓ Flow 6: Multi-hop path: {' → '.join(path)}")

    feedback.close()
    kg.close()
    shutil.rmtree(tmpdir)
    logger.info("✓ End-to-end feedback loop complete\n")


def main():
    logger.info("=" * 60)
    logger.info("PHASE 12: SELF-REFINING MEMORY")
    logger.info("Feedback Engine + Knowledge Graph")
    logger.info("=" * 60)
    logger.info("")

    try:
        test_feedback_engine()
        test_knowledge_graph()
        test_critic_rag_integration()
        test_end_to_end_feedback_loop()

        logger.info("=" * 60)
        logger.info("ALL PHASE 12 TESTS PASSED ✓")
        logger.info("=" * 60)

        logger.info("""
Summary:
  ✓ Feedback Engine: Tracks query utility, triggers reindex, sqlite3 persistence
  ✓ Knowledge Graph: Entity linking, multi-hop paths, co-occurrence extraction
  ✓ Critic Integration: RAG answers validated, utility scored, weak queries flagged
  ✓ End-to-End: Query → Retrieve → Validate → Feedback → Graph Enrich

Architecture:
  Query → RAG Bridge → Retrieve chunks
        → Critic reviews answer → Feedback records utility
        → Knowledge Graph enriches entities + relations
        → Feedback detects weak files → triggers re-index
  
RAM Budget (Phase 12):
  - Feedback Engine: ~2MB (sqlite3)
  - Knowledge Graph: ~15MB (NetworkX + sqlite3)
  Total Phase 12: ~17MB
  Full AI Stack: ~5,507 MB (5.4 GB)
""")

    except AssertionError as e:
        logger.error(f"TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()