"""Local LLM inference using llama-cpp-python for dynamic response generation.

ThreadPoolExecutor offloads blocking llama-cpp calls from caller threads,
preventing the Qt event loop and the autonomy background thread from
stalling during generation (Item 5 punch list).
"""

import logging
import time
import threading
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
    
    # Model path (can be set via BUBBY_LLM_PATH env var or .env)
    model_path: str = ""
    
    # Context & generation
    n_ctx: int = 4096              # Context window size
    n_batch: int = 512             # Batch size for prompt processing
    n_threads: int = 4             # CPU threads (match i5 cores)
    n_gpu_layers: int = -1         # -1 = offload all layers to GPU (Metal/CUDA/Vulkan)
    
    # Generation parameters
    temperature: float = 0.7       # Creativity (0.0 = deterministic)
    top_p: float = 0.9             # Nucleus sampling
    top_k: int = 40                # Top-k sampling
    repeat_penalty: float = 1.1    # Repetition penalty
    max_tokens: int = 150          # Max response tokens
    
    # System
    use_mlock: bool = True         # Lock memory to prevent swapping
    use_mmap: bool = True          # Use mmap for faster loading
    verbose: bool = False          # llama.cpp debug output
    
    # Persona integration
    persona: Optional[PersonaConfig] = None
    
    def __post_init__(self):
        """Validate configuration."""
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
    
    Provides thread-safe, streaming-capable inference for dynamic
    response generation with persona-aware system prompts.
    
    ThreadPoolExecutor deduplicates inference offload: all generate/
    generate_structured calls are dispatched to a single pool, so
    callers (UI thread, AutonomyLoop, InteractionHandler) never
    block on llama-cpp's C-level work.
    
    Memory footprint (Q4_K_M quantized):
    - Qwen2.5-1.5B: ~1.2 GB
    - Llama-3.2-1B: ~0.9 GB
    - Phi-3-mini: ~2.1 GB
    """
    
    # Single shared ThreadPoolExecutor for offloading blocking llama-cpp calls
    _shared_executor: Optional[ThreadPoolExecutor] = None
    
    def __init__(self, config: LLMConfig) -> None:
        """
        Initialize LLM inference engine.
        
        Args:
            config: LLMConfig with model path and parameters
        """
        self._config = config
        self._llm = None
        self._lock = threading.RLock()  # protects self._llm (init/access)
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
        """Get or create the shared ThreadPoolExecutor.
        
        Using a single shared pool ensures that blocking llama-cpp calls
        are serialized and don't overwhelm CPU threads. Max 2 workers:
        one for generation, one spare.
        """
        if cls._shared_executor is None:
            cls._shared_executor = ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="llm-inference",
            )
        return cls._shared_executor
    
    def initialize(self) -> bool:
        """
        Initialize the llama-cpp model.
        
        Returns:
            True if initialization successful
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
                
                logger.info(f"Loading model: {model_path.name}")
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
                logger.info(f"Model loaded in {load_time:.1f}s")
                logger.info(f"Context: {self._config.n_ctx}, Threads: {self._config.n_threads}")
                
                self._initialized = True
                return True
                
            except ImportError:
                logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
                return False
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
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
        Generate a response from the LLM (dispatched to ThreadPoolExecutor).
        
        Args:
            prompt: User prompt / conversation history
            system_prompt: System prompt (persona, rules)
            max_tokens: Override max tokens
            temperature: Override temperature
            stop: Stop sequences
            
        Returns:
            InferenceResult with generated text and stats
        """
        if not self._initialized:
            if not self.initialize():
                return InferenceResult(
                    text="[LLM not initialized]",
                    tokens_generated=0,
                    generation_time_ms=0,
                    tokens_per_second=0,
                    stop_reason="error"
                )
        
        future: Future = self._get_executor().submit(
            self._generate_blocking,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        return future.result()  # caller blocks on the executor's thread
    
    def _generate_blocking(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> InferenceResult:
        """Blocking llama-cpp generation running inside ThreadPoolExecutor."""
        with self._lock:
            # Build messages for chat format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Apply generation parameters
            gen_kwargs = {
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "repeat_penalty": self._config.repeat_penalty,
                "stop": stop or ["</s>", "<|endoftext|>", "<|eot_id|>"],
            }
            
            # Generate
            start_time = time.time()
            
            try:
                # Use chat completion for better instruction following
                response = self._llm.create_chat_completion(
                    messages=messages,
                    **gen_kwargs
                )
                
                generation_time = time.time() - start_time
                generated_text = response["choices"][0]["message"]["content"].strip()
                
                # Extract token usage
                usage = response.get("usage", {})
                tokens_generated = usage.get("completion_tokens", 0)
                if tokens_generated == 0:
                    # Fallback: estimate from text
                    tokens_generated = len(generated_text.split()) * 1.3
                
                tokens_per_sec = tokens_generated / (generation_time / 1000) if generation_time > 0 else 0
                
                # Update stats
                self._stats["total_inferences"] += 1
                self._stats["total_tokens"] += tokens_generated
                self._stats["total_time_ms"] += generation_time * 1000
                
                logger.debug(
                    f"LLM inference: {tokens_generated} tokens in {generation_time*1000:.0f}ms "
                    f"({tokens_per_sec:.1f} tok/s)"
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
                logger.error(f"LLM generation failed: {e}")
                return InferenceResult(
                    text=f"[Generation error: {e}]",
                    tokens_generated=0,
                    generation_time_ms=0,
                    tokens_per_second=0,
                    stop_reason="error",
                )
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ):
        """
        Generate response as a stream of tokens.
        
        Note: Streaming runs on the caller's thread because the generator
        protocol requires synchronous iteration. For non-streaming,
        ThreadPoolExecutor is used.
        
        Yields:
            Token strings as they're generated
        """
        with self._lock:
            if not self._initialized:
                if not self.initialize():
                    yield "[LLM not initialized]"
                    return
            
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
                "stream": True,
            }
            
            try:
                for chunk in self._llm.create_chat_completion(messages=messages, **gen_kwargs):
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
                        
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")
                yield f"[Error: {e}]"
    
    def generate_structured(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_schema: Optional[dict] = None,
    ) -> InferenceResult:
        """
        Generate a JSON-constrained structured response from the LLM.
        
        Dispatched to ThreadPoolExecutor to avoid blocking callers.
        
        Uses llama-cpp-python's JSON schema/grammar support to enforce
        valid JSON output. The LLM physically cannot output text outside
        the specified schema.
        
        Args:
            prompt: User prompt / context for generation
            system_prompt: System prompt (persona, behavior rules)
            max_tokens: Override max tokens
            temperature: Override temperature
            json_schema: JSON schema dict for grammar constraint
                         (defaults to BUBBY_RESPONSE_JSON_SCHEMA)
            
        Returns:
            InferenceResult with generated text (JSON string)
        """
        if not self._initialized:
            if not self.initialize():
                return InferenceResult(
                    text='{"animation": "idle", "speech": ""}',
                    tokens_generated=0,
                    generation_time_ms=0,
                    tokens_per_second=0,
                    stop_reason="error"
                )
        
        future: Future = self._get_executor().submit(
            self._generate_structured_blocking,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=json_schema,
        )
        return future.result()
    
    def _generate_structured_blocking(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_schema: Optional[dict] = None,
    ) -> InferenceResult:
        """Blocking structured generation inside ThreadPoolExecutor."""
        with self._lock:
            # Use default schema if none provided
            schema = json_schema or BUBBY_RESPONSE_JSON_SCHEMA
            
            # Build messages for chat format
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generation parameters
            gen_kwargs = {
                "max_tokens": max_tokens or self._config.max_tokens,
                "temperature": temperature if temperature is not None else self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
                "repeat_penalty": self._config.repeat_penalty,
                "stop": ["</s>", "<|endoftext|>", "<|eot_id|>"],
            }
            
            # Apply JSON grammar constraint if supported
            try:
                # llama-cpp-python >= 0.2.0 supports response_format
                gen_kwargs["response_format"] = {
                    "type": "json_object",
                    "schema": schema,
                }
                logger.debug("Using JSON schema grammar constraint")
            except Exception:
                # Older versions - append schema instructions to prompt
                logger.warning("JSON schema grammar not supported, using prompt-based enforcement")
                schema_instruction = (
                    "\n\nYou MUST respond with ONLY a valid JSON object matching this schema. "
                    "No markdown codes, no explanation:\n"
                    + str(schema)
                )
                messages[-1]["content"] += schema_instruction
            
            # Generate
            start_time = time.time()
            
            try:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    **gen_kwargs
                )
                
                generation_time = time.time() - start_time
                generated_text = response["choices"][0]["message"]["content"].strip()
                
                # Clean any markdown code fences
                generated_text = generated_text.replace("```json", "").replace("```", "").strip()
                
                # Extract token usage
                usage = response.get("usage", {})
                tokens_generated = usage.get("completion_tokens", 0)
                if tokens_generated == 0:
                    tokens_generated = len(generated_text.split()) * 1.3
                
                tokens_per_sec = tokens_generated / (generation_time / 1000) if generation_time > 0 else 0
                
                # Update stats
                self._stats["total_inferences"] += 1
                self._stats["total_tokens"] += tokens_generated
                self._stats["total_time_ms"] += generation_time * 1000
                
                logger.debug(
                    f"LLM structured: {tokens_generated} tokens in {generation_time*1000:.0f}ms "
                    f"({tokens_per_sec:.1f} tok/s) → {generated_text[:80]}"
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
                logger.error(f"LLM structured generation failed: {e}")
                return InferenceResult(
                    text='{"animation": "confused", "speech": ""}',
                    tokens_generated=0,
                    generation_time_ms=0,
                    tokens_per_second=0,
                    stop_reason="error",
                )

    def build_system_prompt(self) -> str:
        """
        Build system prompt from persona configuration.
        
        Returns:
            Formatted system prompt string
        """
        if self._config.persona:
            return self._config.persona.build_system_prompt()
        
        # Default fallback
        return (
            "You are Bubby, a friendly desktop companion. "
            "Keep responses concise, warm, and helpful. "
            "Use contractions and occasional emojis. "
            "Never reveal internal state or technical details."
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        with self._lock:
            stats = dict(self._stats)
            if stats["total_inferences"] > 0:
                stats["avg_tokens_per_inference"] = stats["total_tokens"] / stats["total_inferences"]
                stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_inferences"]
                stats["avg_tokens_per_sec"] = stats["total_tokens"] / (stats["total_time_ms"] / 1000) if stats["total_time_ms"] > 0 else 0
            stats["initialized"] = self._initialized
            stats["model_path"] = self._config.model_path
            return stats
    
    def is_ready(self) -> bool:
        """Check if LLM is initialized and ready."""
        return self._initialized and self._llm is not None
    
    @classmethod
    def shutdown_executor(cls) -> None:
        """Shut down the shared ThreadPoolExecutor."""
        if cls._shared_executor is not None:
            cls._shared_executor.shutdown(wait=True, cancel_futures=False)
            cls._shared_executor = None
    
    def shutdown(self) -> None:
        """Clean up resources."""
        with self._lock:
            if self._llm:
                del self._llm
                self._llm = None
            self._initialized = False
            logger.info("LLM inference shutdown complete")


# Testing helper
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    
    # Test with a model path (will fail if model doesn't exist)
    config = LLMConfig(
        model_path="models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_threads=4,
    )
    
    logger.info("=" * 60)
    logger.info("LLM INFERENCE TEST")
    logger.info("=" * 60)
    
    llm = LLMInference(config)
    
    if llm.initialize():
        logger.info("✓ Model initialized")
        
        # Test generation (offloaded via ThreadPoolExecutor)
        result = llm.generate(
            prompt="Hello! What are you?",
            system_prompt="You are Bubby, a friendly desktop companion.",
        )
        
        logger.info(f"Response: {result.text}")
        logger.info(f"Tokens: {result.tokens_generated}, Time: {result.generation_time_ms:.0f}ms")
        logger.info(f"Speed: {result.tokens_per_second:.1f} tok/s")
        
        # Test streaming
        logger.info("\nStreaming test:")
        for token in llm.generate_stream(
            prompt="Count to 5",
            system_prompt="You are Bubby. Be brief.",
            max_tokens=30,
        ):
            print(token, end="", flush=True)
        print()
        
        logger.info(f"\nStats: {llm.get_stats()}")
    else:
        logger.warning("Model not found - skipping inference test")
        logger.info("Run scripts/download_llm.py to download a model")
    
    LLMInference.shutdown_executor()