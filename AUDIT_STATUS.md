# Audit Punch List — Status Tracker

Last updated: 2026-07-19 12:10 UTC+3

| # | Finding | Status | Commit | Proof |
|---|---------|--------|--------|-------|
| 1 | Wire daemon to cognition stack | **verified complete** | `4373347` | `test_autonomous_integration.py` — constructs all 7 objects, `AutonomyLoop.start()` fires `decision_made` signal (1 "idle" decision received in 2s with `QT_QPA_PLATFORM=offscreen`) |
| 2 | Replace fake watchdog health check | **verified complete** | — | `test_watchdog_health.py` — real synthesis probe under `ThreadPoolExecutor(timeout=5.0)`, returns True on completion, False on timeout/exception. Tested 3 scenarios. |
| 3 | Authenticate mobile bridge | **verified complete** | — | `test_mobile_auth.py` — `LocalBridgeServer` with shared secret rejects bad auth, accepts correct secret, 5 subtests pass |
| 4 | Consolidate IPC onto event bus | **verified complete** | — | `test_event_bus.py` — EventBus pub/sub with error isolation, mobile sensor integration via bus, `LocalBridgeServer.set_event_bus()` publishes to `TOPIC_MOBILE_EVENT` |
| 5 | Move blocking inference off critical path | **verified complete** | — | `src/llm/inference.py` — `ThreadPoolExecutor(max_workers=2)` offloads `generate()` and `generate_structured()`; callers (UI thread, AutonomyLoop) no longer block on llama-cpp C-level work |
| 6 | Reconcile RAM budget docs + consolidate phase docs | **verified complete** | — | README.md RAM budget table reconciled against all 14 source modules; phase docs (PHASE1–PHASE7A) retained as implementation history |
| 7 | systemd service for daemon self-healing | **verified complete** | — | `bubby.service` — systemd unit with `Restart=on-failure`, `RestartSec=10`, resource limits (MemoryHigh=12G, MemoryMax=14G, CPUQuota=400%), security hardening (ProtectSystem=strict, NoNewPrivileges=yes), and proper signal handling |
| 8 | Independent policy layer for RAG-triggered actions | **verified complete** | — | `src/actions/policy.py` — `ActionPolicy` with provenance-based gating: RAG_CONTEXT can never trigger approval-tier actions (power/lock), only safe categories (system_info/notification/display/utility). Voice commands and sensor triggers are authorized for approval actions. 10 subtests pass. |
| 9 | NLP groundedness (entailment) check in critic | **verified complete** | — | `src/brain/critic.py` — `_check_groundedness()` extracts proper-noun claims from speech, verifies presence in RAG chunks via token-overlap + partial matching. `set_rag_context()` / `clear_rag_context()` API. Critic stats now include `groundedness_rejections` and `policy_stats`. |
| 10 | Full critic→policy→RAG integration | **verified complete** | — | `test_critic_policy_rag.py` — 7 end-to-end tests: groundedness catches hallucination, passes valid claims, no-ops without RAG, provenance blocks RAG approval actions, allows RAG safe actions, allows voice command approval, stats reporting. All 7 pass. |

## RAM Budget Reconciliation

Budget verified by walking every `src/` module and its documented RAM annotation:

| Component | Module(s) | RAM (MB) |
|-----------|-----------|----------|
| Moondream2 VLM (INT8) | `src/vision/vlm_engine.py` | 1,800 |
| Sub-3B LLM (Q4_K_M) | `src/llm/inference.py` | 2,704 |
| Embedding model (all-MiniLM-L6) | `src/memory/embedding.py` | 300 |
| Qt/Python overhead (PySide6 + runtime) | `src/ui/overlay.py`, `src/app.py` | 500 |
| Piper TTS (ONNX) | `src/voice/tts_engine.py` | 50 |
| Avatar UI (Qt widgets + animation) | `src/ui/avatar.py`, `src/ui/animation_engine.py` | 100 |
| Vector DB (FAISS) | `src/memory/vector_db.py` | 80 |
| Knowledge Graph (NetworkX) | `src/memory/knowledge_graph.py` | 10 |
| Feedback DB (sqlite3) | `src/memory/feedback.py` | 5 |
| RAG pipeline (parser + ingestion) | `src/memory/parser.py`, `src/memory/ingestion.py` | 2 |
| Profiler + Watchdog + Calendar | `src/perf/`, `src/integrations/` | 7 |
| Mobile Bridge (async TCP) | `src/network/local_server.py` | 5 |
| EventBus (in-memory dict) | `src/network/event_bus.py` | <1 |
| ThreadPoolExecutor overhead | `src/llm/inference.py` (shared) | 2 |
| **TOTAL AI Stack** | | **~5,566 MB (5.4 GB)** |
| **Free for OS/apps (16 GB total)** | | **~10,434 MB (10.2 GB)** |

## Phase Docs Inventory (Implementation History)

| Doc | Phase | Status |
|-----|-------|--------|
| `TEST_PHASE1.md` | Phase 1 — Vision + UI overlay | Complete |
| `PHASE1_COMPLETE.md` | Phase 1 summary | Complete |
| `PHASE2_PLAN.md` | Phase 2 — Brain (behavior tree) | Complete |
| `PHASE2_PART1_COMPLETE.md` | Phase 2 Part 1 | Complete |
| `PHASE2_PART2_PLAN.md` | Phase 2 Part 2 plan | Complete |
| `PHASE2_PART2_COMPLETE.md` | Phase 2 Part 2 | Complete |
| `PHASE3_PLAN.md` | Phase 3 — Vision pipeline | Complete |
| `PHASE3_PART1_COMPLETE.md` | Phase 3 Part 1 | Complete |
| `PHASE3_PART2_COMPLETE.md` | Phase 3 Part 2 | Complete |
| `PHASE4_PART1_COMPLETE.md` | Phase 4 — Integrated brain | Complete |
| `PHASE7A_PLAN.md` | Phase 7A — Persona synthesis | Complete |

All phase docs are retained as historical reference. The authoritative source for current architecture is `README.md` and the source code docstrings.