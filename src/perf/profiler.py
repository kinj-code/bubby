"""Pipeline Profiler — non-blocking telemetry for every cognitive stage.

Records latency (ms) for each pipeline stage to identify bottlenecks:
VLM Inference → RAG Retrieval → Graph Lookup → LLM Generation → TTS Synthesis

RAM: ~1MB (rolling buffer of last 100 samples).
"""

import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque

@dataclass
class PipelineStage:
    name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    last_time_ms: float = 0.0
    samples: deque = field(default_factory=lambda: deque(maxlen=100))

class PipelineProfiler:
    """Non-blocking telemetry for the cognitive pipeline."""

    STAGE_ORDER = ["vision", "rag_retrieval", "graph_lookup", "llm_inference", "tts_synthesis", "action_execution"]

    def __init__(self) -> None:
        self._stages: Dict[str, PipelineStage] = {s: PipelineStage(name=s) for s in self.STAGE_ORDER}
        self._lock = threading.Lock()
        self._active_timers: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        with self._lock:
            self._active_timers[stage] = time.time()

    def stop(self, stage: str) -> float:
        with self._lock:
            start = self._active_timers.pop(stage, None)
            if not start:
                return 0.0
            elapsed = (time.time() - start) * 1000
            st = self._stages.setdefault(stage, PipelineStage(name=stage))
            st.total_calls += 1
            st.total_time_ms += elapsed
            st.avg_time_ms = st.total_time_ms / st.total_calls
            st.min_time_ms = min(st.min_time_ms, elapsed)
            st.max_time_ms = max(st.max_time_ms, elapsed)
            st.last_time_ms = elapsed
            st.samples.append(elapsed)
            return elapsed

    def get_heatmap(self) -> Dict[str, Any]:
        with self._lock:
            return {
                s: {"avg_ms": round(self._stages[s].avg_time_ms, 1), "calls": self._stages[s].total_calls,
                    "min_ms": round(self._stages[s].min_time_ms, 1) if self._stages[s].total_calls else 0,
                    "max_ms": round(self._stages[s].max_time_ms, 1)}
                for s in self.STAGE_ORDER if self._stages[s].total_calls > 0
            }

    def get_pipeline_summary(self) -> str:
        heatmap = self.get_heatmap()
        lines = ["Stage          | Avg(ms) | Calls | Min/Max"]
        for stage, data in heatmap.items():
            lines.append(f"{stage:<15}| {data['avg_ms']:>7.1f} | {data['calls']:>5} | {data['min_ms']:.0f}/{data['max_ms']:.0f}")
        return "\n".join(lines)

    def get_total_pipeline_ms(self) -> float:
        with self._lock:
            return sum(s.last_time_ms for s in self._stages.values() if s.total_calls > 0)

    def reset(self) -> None:
        with self._lock:
            self._stages = {s: PipelineStage(name=s) for s in self.STAGE_ORDER}

    def get_stats(self) -> Dict[str, Any]:
        return {"heatmap": self.get_heatmap(), "total_last_pipeline_ms": round(self.get_total_pipeline_ms(), 1)}


if __name__ == "__main__":
    import logging
    import random
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    logger = logging.getLogger(__name__)

    profiler = PipelineProfiler()
    for _ in range(5):
        for stage in profiler.STAGE_ORDER:
            profiler.start(stage)
            time.sleep(random.uniform(0.001, 0.05))
            profiler.stop(stage)

    logger.info("Pipeline Summary:\n" + profiler.get_pipeline_summary())
    stats = profiler.get_stats()
    assert stats["total_last_pipeline_ms"] > 0
    assert len(stats["heatmap"]) >= 3
    logger.info(f"Total pipeline: {stats['total_last_pipeline_ms']:.1f}ms")
    logger.info("ALL PROFILER TESTS PASSED")