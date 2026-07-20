"""Local LLM inference using llama-cpp-python — NO FALLBACkS, RAW TRUTH.

Fixes Bubby 2.2: All fallback strings removed. If the LLM cannot generate,
exceptions propagate with full tracebacks. The caller sees the raw truth.
"""

import logging
import time
import threading
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future

from src.persona.config import PersonaConfig
from src.persona.prompts import BUBBY_RESPONSE_JSON_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    model_path: str = ""
    n_ctx: int = 4096
    n_batch: int = 512
    n_threads: int = 4
    n_gpu_layers: int = -1
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 150
    use_mlock: bool = True
    use_mmap: bool = True
    verbose: bool = False
    persona: Optional[PersonaConfig] = None

    def __post_init__(self):
        if self.max_tokens > self.n_ctx:
            raise ValueError(f"max_tokens ({self.max_tokens}) must be <= n_ctx ({self.n_ctx})")


@dataclass
class InferenceResult:
    """Result of LLM inference."""
    text: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float
    stop_reason: str = "stop"


class LLMInference:
    """
    Local LLM inference engine using llama-cpp-python.

    ThreadPoolExecutor offloads blocking llama-cpp calls from caller threads,
    preventing the Qt event loop and autonomy background thread from
    stalling during generation.
    """

    _shared_executor: Optional[ThreadPoolExecutor] = None

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._llm = None
        self._lock = threading.RLock()
        self._initialized = False
        self._stats = {
            "total_inferences": 0,
            "total_tokens": 0,
            "total_time_ms": 0.0,
            "errors": 0,
        }
        logger.info(f"LLMInference created (model: {config.model_path})")

    @classmethod
    def _get_executor(cls) -> ThreadPoolExecutor:
        if cls._shared_executor is None:
            cls._shared_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="llm-inference",
            )
        return cls._shared_executor

    def is_ready(self) -> bool:
        """Check if the LLM model is fully loaded and ready for inference."""
        return self._initialized and self._llm is not None

    def initialize(self) -> bool:
        """
        Initialize the llama-cpp model. Returns True if model loaded.

        ══ FIX: Log explicit confirmation when C++ library finishes loading ══
        """
        with self._lock:
            if self._initialized:
                logger.debug("LLM already initialized")
                return True

            if not self._config.model_path:
                logger.error("No model path configured")
                return False

            model_path = Path(self._config.model_path)
            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False

            try:
                from llama_cpp import Llama

                logger.info(f"⏳ Loading model: {model_path.name}")
                start_time = time.time()

                self._llm = Llama(
                    model_path=str(model_path),
                    n_ctx=self._config.n_ctx,
                    n_batch=self._config.n_batch,
                    n_threads=self._config.n_threads,
                    n_gpu_layers=self._config.n_gpu_layers,
                    use_mlock=self._config.use_mlock,
                    use_mmap=self._config.use_mmap,
                    verbose=self._config.verbose,
                )

                load_time = time.time() - start_time
                logger.info(f"✅ Llama model loaded successfully in {load_time:.1f}s")
                logger.info(f"   Context: {self._config.n_ctx}, Threads: {self._config.n_threads}")

                self._initialized = True
                return True

            except ImportError:
                logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
                return False
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.error(traceback.format_exc())
                return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Generate a response from the LLM — NO FALLBACK STRINGS.

        If the model is not initialized, raises ValueError.
        If generation fails, raises RuntimeError with full traceback.
        The caller gets the RAW TRUTH.
        """
        # ══ FORCE initialize — NO silent fallback string ══
        if not self._initialized:
            ok = self.initialize()
            if not ok:
                raise ValueError("LLM model is None — initialization failed. No model loaded.")

        # ══ Force real inference on the shared executor ══
        logger.info("LLM inference starting...")
        future: Future = self._get_executor().submit(
            self._generate_blocking,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        result = future.result()
        logger.info(f"LLM inference complete in {result.generation_time_ms:.0f}ms "
                     f"({result.tokens_generated} tokens)")
        return result

    def _generate_blocking(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> InferenceResult:
        """Blocking llama-cpp generation — NO FALLBACK STRINGS.

        If the LLM is None, raises ValueError.
        If llama-cpp crashes, the exception propagates with full traceback.
        """
        with self._lock:
            if self._llm is None:
                raise ValueError("LLM model is None — cannot generate. Call initialize() first.")

            # Build messages for chat format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            gen_kwargs = {
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "repeat_penalty": self._config.repeat_penalty,
                "stop": stop or ["</s>", "<|endoftext|>", "<|eot_id|>"],
            }

            # ══ REAL TIMING: start right before llama-cpp call ══
            logger.info("⏳ Llama-cpp create_chat_completion starting...")
            start_time = time.time()
            tokens_generated = 0

            try:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    **gen_kwargs
                )

                generation_time = time.time() - start_time
                generated_text = response["choices"][0]["message"]["content"].strip()

                usage = response.get("usage", {})
                tokens_generated = usage.get("completion_tokens", 0)
                if tokens_generated == 0:
                    tokens_generated = int(len(generated_text.split()) * 1.3)

                tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

                self._stats["total_inferences"] += 1
                self._stats["total_tokens"] += tokens_generated
                self._stats["total_time_ms"] += generation_time * 1000

                logger.info(
                    f"✅ Llama-cpp inference: {tokens_generated} tokens in "
                    f"{generation_time*1000:.0f}ms ({tokens_per_sec:.1f} tok/s)"
                )

                return InferenceResult(
                    text=generated_text,
                    tokens_generated=int(tokens_generated),
                    generation_time_ms=generation_time * 1000,
                    tokens_per_second=tokens_per_sec,
                    stop_reason=response["choices"][0].get("finish_reason", "stop"),
                )

            except Exception as e:
                self._stats["errors"] += 1
                elapsed = time.time() - start_time
                logger.error(f"❌ LLM generation FAILED after {elapsed*1000:.0f}ms:")
                logger.error(traceback.format_exc())
                raise  # ══ Re-raise — NO fallback string ══

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ):
        """Generate response as a stream of tokens — NO FALLBACK STRINGS."""
        with self._lock:
            if self._llm is None:
                raise ValueError("LLM model is None — cannot stream.")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            gen_kwargs = {
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "repeat_penalty": self._config.repeat_penalty,
                "stop": stop or ["</s>", "<|endoftext|>", "<|eot_id|>"],
            }

            logger.info("⏳ Llama-cpp streaming starting...")
            start_time = time.time()

            try:
                stream = self._llm.create_chat_completion(
                    messages=messages,
                    stream=True,
                    **gen_kwargs,
                )

                for chunk in stream:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

                elapsed = time.time() - start_time
                logger.info(f"✅ Streaming complete in {elapsed*1000:.0f}ms")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ LLM streaming FAILED after {elapsed*1000:.0f}ms:")
                logger.error(traceback.format_exc())
                raise

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        json_schema: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> InferenceResult:
        """
        Generate a structured JSON response — NO FALLBACK STRINGS.

        Raises ValueError if model is None. Raises RuntimeError on generation failure.
        """
        if self._llm is None:
            raise ValueError("LLM model is None — cannot generate structured output.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        gen_kwargs = {
            "max_tokens": max_tokens or self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
        }

        if json_schema:
            gen_kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,
            }

        logger.info("⏳ Llama-cpp structured generation starting...")
        start_time = time.time()

        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                **gen_kwargs,
            )
            generation_time = time.time() - start_time
            generated_text = response["choices"][0]["message"]["content"].strip()
            usage = response.get("usage", {})
            tokens_generated = usage.get("completion_tokens", 0)
            if tokens_generated == 0:
                tokens_generated = int(len(generated_text.split()) * 1.3)
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

            logger.info(f"✅ Structured generation complete in {generation_time*1000:.0f}ms")

            return InferenceResult(
                text=generated_text,
                tokens_generated=int(tokens_generated),
                generation_time_ms=generation_time * 1000,
                tokens_per_second=tokens_per_sec,
                stop_reason=response["choices"][0].get("finish_reason", "stop"),
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Structured generation FAILED after {elapsed*1000:.0f}ms:")
            logger.error(traceback.format_exc())
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return dict(self._stats)

    def shutdown(self) -> None:
        """Cleanup LLM resources."""
        with self._lock:
            if self._llm:
                del self._llm
                self._llm = None
            self._initialized = False
        logger.info("LLM inference shut down")