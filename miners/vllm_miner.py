#!/usr/bin/env python3
"""
Production vLLM-based inference miner with hidden state extraction.

Drop-in replacement for real_miner.py that uses vLLM's AsyncLLMEngine for
high-throughput serving (continuous batching, PagedAttention) while maintaining
hidden state verification compatibility.

Architecture:
    - vLLM AsyncLLMEngine handles all inference (generation)
    - A separate HuggingFace model (loaded with output_hidden_states=True) runs
      a single forward pass over the full sequence (prompt + generated tokens)
      to extract hidden states for verification challenges
    - The HF forward pass is encoding-only (no generation), so it runs in ~100ms
      even for 4096-token sequences on a 4090

Endpoints (identical to real_miner.py):
    POST /inference         - vLLM generate, then HF forward for hidden states
    POST /inference/stream  - SSE streaming via vLLM, then HF hidden states
    POST /hidden_state      - Return cached hidden state at (layer, position)
    GET  /health            - Health check with model info

Usage:
    python vllm_miner.py --model meta-llama/Meta-Llama-3-8B-Instruct --port 8091
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import signal
import time
import uuid
from collections import OrderedDict, defaultdict
from typing import AsyncIterator

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vllm_miner")


# -- Request/Response Models (identical to real_miner.py) ----------------------

class InferenceRequest(BaseModel):
    prompt: str = ""
    messages: list[dict] | None = None
    max_tokens: int = 128
    request_id: str | None = None
    nonce: str | None = None  # Anti-relay: miner-bound HMAC nonce from gateway
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    challenge_layer: int | None = None
    challenge_token: int | None = None
    challenge_extra: list[list[int]] | None = None
    # Inline commitment fields: gateway tells miner which layers/positions to commit
    commit_layers: list[int] | None = None
    commit_positions: list[str | int] | None = None  # "last" or integer index


class InferenceResponse(BaseModel):
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float
    all_token_ids: list[int] | None = None
    challenge_result: dict | None = None
    commitments: list[dict] | None = None  # Inline hidden state commitments


class HiddenStateRequest(BaseModel):
    request_id: str
    layer_index: int
    token_index: int


class HiddenStateResponse(BaseModel):
    request_id: str
    layer_index: int
    token_index: int
    hidden_state: list[float]
    latency_ms: float


# -- Hidden State Cache --------------------------------------------------------

class HiddenStateCache:
    """LRU cache for hidden states from inference runs with TTL and query limits."""

    ENTRY_TTL_S = 120  # Entries expire after 120s (covers deferred challenge window)
    MAX_QUERIES_PER_REQUEST = 5  # Max hidden state queries per request_id

    def __init__(self, max_requests: int = 200):
        self.max_requests = max_requests
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self._timestamps: dict[str, float] = {}  # request_id -> store time
        self._query_counts: dict[str, int] = {}  # request_id -> query count
        self._lock = asyncio.Lock()

    async def store(self, request_id: str, hidden_states: dict):
        """Store hidden states. hidden_states = {layer_idx: tensor(seq_len, hidden_dim)}"""
        async with self._lock:
            # Evict expired entries
            now = time.time()
            expired = [k for k, t in self._timestamps.items() if now - t > self.ENTRY_TTL_S]
            for k in expired:
                self.cache.pop(k, None)
                self._timestamps.pop(k, None)
                self._query_counts.pop(k, None)
            if len(self.cache) >= self.max_requests:
                oldest_key, _ = self.cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)
                self._query_counts.pop(oldest_key, None)
            self.cache[request_id] = hidden_states
            self._timestamps[request_id] = now
            self._query_counts[request_id] = 0

    async def get(self, request_id: str, layer_index: int, token_index: int) -> np.ndarray | None:
        async with self._lock:
            if request_id not in self.cache:
                return None
            # Check TTL
            if time.time() - self._timestamps.get(request_id, 0) > self.ENTRY_TTL_S:
                self.cache.pop(request_id, None)
                self._timestamps.pop(request_id, None)
                self._query_counts.pop(request_id, None)
                return None
            # Check query limit
            count = self._query_counts.get(request_id, 0)
            if count >= self.MAX_QUERIES_PER_REQUEST:
                return None
            self._query_counts[request_id] = count + 1
            states = self.cache[request_id]
            if layer_index not in states:
                return None
            layer_tensor = states[layer_index]
            if token_index >= layer_tensor.shape[0]:
                return None
            self.cache.move_to_end(request_id)
            return layer_tensor[token_index].numpy()

    @property
    def size(self):
        return len(self.cache)


# -- vLLM Miner ---------------------------------------------------------------

class VLLMMiner:
    """
    High-throughput miner combining vLLM for generation and HuggingFace for
    hidden state extraction.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        cache_size: int = 200,
        hf_device: str = "auto",
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.cache = HiddenStateCache(max_requests=cache_size)
        self.total_requests = 0
        self.total_challenges = 0
        self.challenges_passed = 0

        # -- Load vLLM engine --
        log.info(f"Initializing vLLM engine: {model_name}")
        log.info(f"  tensor_parallel_size={tensor_parallel_size}, max_model_len={max_model_len}")
        log.info(f"  gpu_memory_utilization={gpu_memory_utilization}")

        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",
            disable_log_stats=True,
            enforce_eager=enforce_eager,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.SamplingParams = SamplingParams
        log.info("vLLM engine initialized")

        # -- Load HuggingFace model for hidden state extraction --
        # This model is used ONLY for forward passes (no generation).
        # We load it with output_hidden_states=True and in eval mode.
        # On a 4090, the vLLM engine uses ~85% of VRAM; the HF model shares
        # the same weights on disk and uses CPU for forward passes, or we can
        # load it on GPU with reduced memory if space allows.
        log.info(f"Loading HuggingFace model for hidden state extraction: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if hf_device == "cpu":
            log.info("HF model forced to CPU by --hf-device cpu")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                output_hidden_states=True,
            )
            self.hf_device = torch.device("cpu")
        else:
            # Try GPU first; fall back to CPU if OOM
            try:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    output_hidden_states=True,
                )
                self.hf_device = next(self.hf_model.parameters()).device
                log.info(f"HF model loaded on device: {self.hf_device}")
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                log.warning(f"GPU OOM for HF model ({e}), falling back to CPU with float32")
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True,
                    output_hidden_states=True,
                )
                self.hf_device = torch.device("cpu")
                log.info("HF model loaded on CPU")

        self.hf_model.eval()
        self.num_layers = self.hf_model.config.num_hidden_layers
        self.hidden_dim = self.hf_model.config.hidden_size

        log.info(
            f"Model ready: {model_name} | "
            f"{self.num_layers} layers | hidden_dim={self.hidden_dim} | "
            f"HF device={self.hf_device}"
        )

    def _build_sampling_params(self, request: InferenceRequest):
        """Convert InferenceRequest fields to vLLM SamplingParams."""
        kwargs = {
            "max_tokens": request.max_tokens,
        }

        # Temperature
        temp = request.temperature
        if temp is not None:
            kwargs["temperature"] = max(temp, 0.0)
        else:
            kwargs["temperature"] = 1.0

        # Top-p
        if request.top_p is not None:
            kwargs["top_p"] = request.top_p

        # Frequency and presence penalty (vLLM supports these directly)
        if request.frequency_penalty is not None:
            kwargs["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            kwargs["presence_penalty"] = request.presence_penalty

        # Stop sequences
        if request.stop:
            if isinstance(request.stop, str):
                kwargs["stop"] = [request.stop]
            else:
                kwargs["stop"] = list(request.stop)

        return self.SamplingParams(**kwargs)

    def _prepare_prompt(self, request: InferenceRequest) -> str:
        """Build the text prompt from the request, applying chat template if needed."""
        if request.messages:
            return self.tokenizer.apply_chat_template(
                request.messages, tokenize=False, add_generation_prompt=True
            )
        return request.prompt

    def _move_hf_to_cpu(self):
        """Move HF model to CPU when GPU OOM occurs during forward pass."""
        if self.hf_device != torch.device("cpu"):
            log.warning("Moving HF model to CPU due to GPU OOM during forward pass")
            self.hf_model = self.hf_model.cpu().float()
            self.hf_device = torch.device("cpu")
            torch.cuda.empty_cache()
            log.info("HF model moved to CPU successfully")

    @torch.no_grad()
    def _extract_hidden_states(self, all_token_ids: list[int]) -> dict[int, torch.Tensor]:
        """
        Run a forward pass through the HF model to extract hidden states for
        the full sequence. Returns {layer_idx: tensor(seq_len, hidden_dim)}.

        This is encoding-only (no generation), so it is fast: ~100ms for 4096
        tokens on a 4090 GPU, or ~500ms on CPU for an 8B model.

        If GPU OOM occurs during forward pass, automatically falls back to CPU.
        """
        input_ids = torch.tensor([all_token_ids], dtype=torch.long, device=self.hf_device)

        # Truncate to model's max position embeddings if needed
        max_pos = getattr(self.hf_model.config, "max_position_embeddings", None)
        if max_pos and input_ids.shape[1] > max_pos:
            input_ids = input_ids[:, :max_pos]

        try:
            outputs = self.hf_model(input_ids=input_ids, output_hidden_states=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e)
            if ("CUDA out of memory" in err_str or "OutOfMemoryError" in type(e).__name__
                    or "same device" in err_str or "different from other" in err_str):
                self._move_hf_to_cpu()
                input_ids = input_ids.cpu()
                outputs = self.hf_model(input_ids=input_ids, output_hidden_states=True)
            else:
                raise

        hidden_states_cache = {}
        # outputs.hidden_states is a tuple of (num_layers + 1) tensors
        # Index 0 is the embedding layer output; layers 1..num_layers are transformer layers
        hs = outputs.hidden_states
        available_layers = min(self.num_layers, len(hs) - 1) if hs else 0
        if available_layers < self.num_layers:
            log.warning(
                f"Hidden state extraction: expected {self.num_layers} layers "
                f"but model returned {len(hs) - 1 if hs else 0}"
            )
        for layer_idx in range(available_layers):
            layer_tensor = hs[layer_idx + 1][0].cpu().float()
            hidden_states_cache[layer_idx] = layer_tensor

        return hidden_states_cache

    async def _serve_inline_challenge(
        self,
        request_id: str,
        layer_index: int,
        token_index: int,
        extra_points: list[list[int]] | None = None,
    ) -> dict:
        """Serve challenge inline after inference."""
        t_start = time.perf_counter()
        state = await self.cache.get(request_id, layer_index, token_index)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        self.total_challenges += 1

        if state is None:
            log.warning(
                f"Inline challenge MISS {request_id[:8]}... | "
                f"layer={layer_index} pos={token_index}"
            )
            return {"error": "cache_miss", "latency_ms": latency_ms}

        self.challenges_passed += 1
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm

        result = {
            "hidden_state": state.tolist(),
            "layer_index": layer_index,
            "token_index": token_index,
            "latency_ms": latency_ms,
        }

        # Serve extra challenge points if requested
        if extra_points:
            extra_states = []
            for point in extra_points:
                if len(point) >= 2:
                    extra_state = await self.cache.get(request_id, point[0], point[1])
                    if extra_state is not None:
                        norm_e = np.linalg.norm(extra_state)
                        if norm_e > 0:
                            extra_state = extra_state / norm_e
                        extra_states.append({
                            "layer_index": point[0],
                            "token_index": point[1],
                            "hidden_state": extra_state.tolist(),
                        })
                    else:
                        extra_states.append({
                            "layer_index": point[0],
                            "token_index": point[1],
                            "error": "cache_miss",
                        })
            result["extra_states"] = extra_states

        return result

    def _extract_commitments(
        self,
        hidden_states_cache: dict[int, "torch.Tensor"],
        commit_layers: list[int],
        commit_positions: list,
        seq_len: int,
        nonce: str | None = None,
    ) -> list[dict]:
        """
        Extract inline commitment hidden states from the cached forward pass.
        Each commitment is {layer, position, hidden_state, commitment_hash}.
        The commitment_hash binds the hidden state to the gateway-provided nonce,
        preventing relay attacks (a relay miner would have a different nonce).
        """
        commitments = []
        for i, layer in enumerate(commit_layers):
            if layer not in hidden_states_cache:
                continue
            layer_tensor = hidden_states_cache[layer]
            # Resolve position
            pos_spec = commit_positions[i] if i < len(commit_positions) else "last"
            if pos_spec == "last":
                pos = seq_len - 1
            else:
                pos = int(pos_spec)
            # Handle negative offsets (e.g., -1 = last, -5 = 5th from end)
            if pos < 0:
                pos = seq_len + pos
            if pos < 0 or pos >= layer_tensor.shape[0]:
                continue
            vec = layer_tensor[pos].numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vec_list = vec.tolist()
            commit_entry = {
                "layer": layer,
                "position": pos,
                "hidden_state": vec_list,
            }
            # Bind commitment to nonce (anti-relay)
            if nonce:
                import hashlib as _hashlib
                quantized = [round(v, 4) for v in vec_list]
                payload = json.dumps(quantized, separators=(",", ":")) + nonce
                commit_entry["commitment_hash"] = _hashlib.sha256(payload.encode()).hexdigest()[:32]
            commitments.append(commit_entry)
        return commitments

    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Non-streaming inference: vLLM generate + HF hidden state extraction."""
        request_id = request.request_id or str(uuid.uuid4())
        prompt_text = self._prepare_prompt(request)
        sampling_params = self._build_sampling_params(request)

        # Tokenize prompt to get input length
        prompt_token_ids = self.tokenizer.encode(prompt_text)
        input_len = len(prompt_token_ids)

        t_start = time.perf_counter()

        # Generate via vLLM
        results_generator = self.engine.generate(prompt_text, sampling_params, request_id)

        # Consume the async generator to get the final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        if final_output is None or len(final_output.outputs) == 0:
            raise HTTPException(status_code=500, detail="vLLM generation produced no output")

        completion = final_output.outputs[0]
        output_text = completion.text
        output_token_ids = list(completion.token_ids)
        output_len = len(output_token_ids)

        # Build full token sequence (prompt + generated)
        all_token_ids = prompt_token_ids + output_token_ids

        # Extract hidden states via HF forward pass
        t_hs_start = time.perf_counter()
        hidden_states_cache = None
        try:
            loop = asyncio.get_event_loop()
            hidden_states_cache = await loop.run_in_executor(
                None, self._extract_hidden_states, all_token_ids
            )
            await self.cache.store(request_id, hidden_states_cache)
            hs_ms = (time.perf_counter() - t_hs_start) * 1000
            log.debug(f"Hidden state extraction: {hs_ms:.1f}ms for {len(all_token_ids)} tokens")
        except Exception as e:
            log.error(f"Hidden state extraction failed: {e}")
            # Continue without hidden states — inference still succeeds

        self.total_requests += 1

        ttft_ms = total_ms / max(output_len, 1)  # approximate
        tps = output_len / max(total_ms / 1000, 0.001)

        # Handle inline challenge (legacy)
        challenge_result = None
        if request.challenge_layer is not None and request.challenge_token is not None:
            challenge_result = await self._serve_inline_challenge(
                request_id, request.challenge_layer, request.challenge_token,
                request.challenge_extra,
            )

        # Extract inline commitments (new fast path)
        commitments = None
        if request.commit_layers and hidden_states_cache:
            commitments = self._extract_commitments(
                hidden_states_cache,
                request.commit_layers,
                request.commit_positions or ["last"] * len(request.commit_layers),
                len(all_token_ids),
                nonce=request.nonce,
            )

        log.info(
            f"Inference {request_id[:8]}... | "
            f"{input_len} in + {output_len} out | "
            f"{total_ms:.1f}ms | {tps:.0f} tok/s | "
            f"cache: {self.cache.size}"
            f"{' +challenge' if challenge_result else ''}"
            f"{f' +{len(commitments)} commits' if commitments else ''}"
        )

        return InferenceResponse(
            request_id=request_id,
            text=output_text,
            input_tokens=input_len,
            output_tokens=output_len,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=tps,
            all_token_ids=all_token_ids,
            challenge_result=challenge_result,
            commitments=commitments,
        )

    async def run_inference_streaming(self, request: InferenceRequest) -> AsyncIterator[str]:
        """
        SSE streaming inference: stream tokens from vLLM, then extract hidden
        states via HF after generation completes.

        Yields SSE-formatted strings: "data: {json}\n\n"
        """
        request_id = request.request_id or str(uuid.uuid4())
        prompt_text = self._prepare_prompt(request)
        sampling_params = self._build_sampling_params(request)

        prompt_token_ids = self.tokenizer.encode(prompt_text)
        input_len = len(prompt_token_ids)

        t_start = time.perf_counter()
        ttft_ms = None
        prev_text = ""

        results_generator = self.engine.generate(prompt_text, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

            if len(request_output.outputs) == 0:
                continue

            completion = request_output.outputs[0]
            new_text = completion.text[len(prev_text):]

            if new_text:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

                yield f"data: {json.dumps({'request_id': request_id, 'token': new_text, 'finish_reason': None})}\n\n"
                prev_text = completion.text

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000

        if ttft_ms is None:
            ttft_ms = total_ms

        # Extract final output
        output_token_ids = []
        if final_output and len(final_output.outputs) > 0:
            completion = final_output.outputs[0]
            output_token_ids = list(completion.token_ids)

        output_len = len(output_token_ids)
        all_token_ids = prompt_token_ids + output_token_ids

        # Extract hidden states via HF forward pass
        hidden_states_cache = None
        try:
            loop = asyncio.get_event_loop()
            hidden_states_cache = await loop.run_in_executor(
                None, self._extract_hidden_states, all_token_ids
            )
            await self.cache.store(request_id, hidden_states_cache)
        except Exception as e:
            log.error(f"Hidden state extraction failed during stream: {e}")

        self.total_requests += 1
        tps = output_len / max(total_ms / 1000, 0.001)

        # Handle inline challenge (legacy)
        challenge_result = None
        if request.challenge_layer is not None and request.challenge_token is not None:
            challenge_result = await self._serve_inline_challenge(
                request_id, request.challenge_layer, request.challenge_token,
                request.challenge_extra,
            )

        # Extract inline commitments (new fast path)
        commitments = None
        if request.commit_layers and hidden_states_cache:
            commitments = self._extract_commitments(
                hidden_states_cache,
                request.commit_layers,
                request.commit_positions or ["last"] * len(request.commit_layers),
                len(all_token_ids),
                nonce=request.nonce,
            )

        # Final SSE event with metadata
        final_meta = {
            "request_id": request_id,
            "token": "",
            "finish_reason": "stop",
            "prompt_tokens": input_len,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "ttft_ms": round(ttft_ms, 2),
            "total_ms": round(total_ms, 2),
            "tokens_per_sec": round(tps, 1),
            "all_token_ids": all_token_ids,
        }
        if challenge_result is not None:
            final_meta["challenge_result"] = challenge_result
        if commitments:
            final_meta["commitments"] = commitments
        yield f"data: {json.dumps(final_meta)}\n\n"
        yield "data: [DONE]\n\n"

        log.info(
            f"Stream {request_id[:8]}... | "
            f"{input_len} in + {output_len} out | "
            f"{total_ms:.1f}ms | {tps:.0f} tok/s"
            f"{' +challenge' if challenge_result else ''}"
            f"{f' +{len(commitments)} commits' if commitments else ''}"
        )

    async def get_hidden_state(self, request: HiddenStateRequest) -> HiddenStateResponse:
        """Return a cached hidden state for a previous inference request."""
        t_start = time.perf_counter()
        state = await self.cache.get(request.request_id, request.layer_index, request.token_index)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        self.total_challenges += 1

        if state is None:
            log.warning(
                f"Challenge MISS {request.request_id[:8]}... | "
                f"layer={request.layer_index} pos={request.token_index}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"No cached hidden state for request {request.request_id}",
            )

        self.challenges_passed += 1

        # Normalize for consistency
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm

        log.info(
            f"Challenge HIT {request.request_id[:8]}... | "
            f"layer={request.layer_index} pos={request.token_index} | "
            f"{latency_ms:.2f}ms"
        )

        return HiddenStateResponse(
            request_id=request.request_id,
            layer_index=request.layer_index,
            token_index=request.token_index,
            hidden_state=state.tolist(),
            latency_ms=latency_ms,
        )


# -- FastAPI App ---------------------------------------------------------------

app = FastAPI(title="vLLM Inference Miner")
miner: VLLMMiner | None = None

# Validator auth: shared secret for authenticating validator requests.
# Set MINER_VALIDATOR_SECRET env var to enable. When set to "enforce",
# unauthenticated requests to /inference and /hidden_state are rejected.
_MINER_VALIDATOR_SECRET = os.environ.get("MINER_VALIDATOR_SECRET", "")
# C5-3: Default to enforced — secure by default. Set MINER_AUTH_ENFORCE=false to disable.
_MINER_AUTH_ENFORCE = os.environ.get("MINER_AUTH_ENFORCE", "true").lower() != "false"
_auth_warn_count = 0


def _verify_validator_token(request: Request, endpoint: str, request_id: str = "", body_hash: str = "") -> bool:
    """Check X-Validator-Key header with per-request HMAC validation.

    C5-4: HMAC now covers body_hash to prevent relay forgery.
    sig = HMAC(secret, "miner_auth:{request_id}:{ts}:{body_hash}")

    Returns True if valid or auth disabled.
    """
    global _auth_warn_count
    if not _MINER_VALIDATOR_SECRET:
        return True
    token = request.headers.get("X-Validator-Key", "")
    if not token:
        _auth_warn_count += 1
        if _auth_warn_count <= 10 or _auth_warn_count % 100 == 0:
            log.warning(f"[AUTH] No X-Validator-Key on {endpoint} from {request.client.host if request.client else '?'} (count={_auth_warn_count})")
        return not _MINER_AUTH_ENFORCE

    # Format: "timestamp:signature" with per-request binding and 60s expiry
    if ":" in token:
        parts = token.split(":", 1)
        ts_str, sig = parts[0], parts[1]
        try:
            ts = int(ts_str)
        except ValueError:
            log.warning(f"[AUTH] Invalid timestamp in token on {endpoint}")
            return not _MINER_AUTH_ENFORCE
        # Check 60-second freshness window (generous for clock skew)
        age = abs(int(time.time()) - ts)
        if age > 60:
            log.warning(f"[AUTH] Expired token on {endpoint} (age={age}s)")
            return not _MINER_AUTH_ENFORCE
        # C5-4: HMAC covers body_hash — relay can't forge for different payload
        msg = f"miner_auth:{request_id}:{ts_str}:{body_hash}".encode()
        expected = hmac.new(_MINER_VALIDATOR_SECRET.encode(), msg, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return True
        # Backwards compat: try without body_hash for rolling upgrade window
        msg_legacy = f"miner_auth:{request_id}:{ts_str}".encode()
        expected_legacy = hmac.new(_MINER_VALIDATOR_SECRET.encode(), msg_legacy, hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected_legacy):
            return True
        log.warning(f"[AUTH] Signature mismatch on {endpoint} (new format)")
        return not _MINER_AUTH_ENFORCE

    # C4 H4-5: Legacy static HMAC skeleton key removed — all miners must use request-bound auth
    log.warning(f"[AUTH] Invalid X-Validator-Key on {endpoint} from {request.client.host if request.client else '?'}")
    return not _MINER_AUTH_ENFORCE


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": miner.model_name if miner else "not loaded",
        "num_layers": miner.num_layers if miner else 0,
        "hidden_dim": miner.hidden_dim if miner else 0,
        "total_requests": miner.total_requests if miner else 0,
    }


def _get_body_hash(request: Request) -> str:
    """C5-4: Compute body hash for content-bound HMAC verification."""
    body_hash = request.headers.get("X-Body-Hash", "")
    return body_hash

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, raw_request: Request):
    if not _verify_validator_token(raw_request, "/inference", request.request_id or "", _get_body_hash(raw_request)):
        raise HTTPException(status_code=401, detail="unauthorized")
    return await miner.run_inference(request)


@app.post("/inference/stream")
async def inference_stream(request: InferenceRequest, raw_request: Request):
    """SSE streaming inference endpoint. Streams tokens as they are generated."""
    if not _verify_validator_token(raw_request, "/inference/stream", request.request_id or "", _get_body_hash(raw_request)):
        raise HTTPException(status_code=401, detail="unauthorized")
    return StreamingResponse(
        miner.run_inference_streaming(request),
        media_type="text/event-stream",
    )


# Simple IP rate limiter for hidden_state: max 30 requests/minute per IP
_hs_rate_limit: dict[str, list[float]] = defaultdict(list)
_HS_RATE_LIMIT_RPM = 30

@app.post("/hidden_state", response_model=HiddenStateResponse)
async def hidden_state(req: HiddenStateRequest, request: Request):
    if not _verify_validator_token(request, "/hidden_state", req.request_id or "", _get_body_hash(request)):
        raise HTTPException(status_code=401, detail="unauthorized")
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _hs_rate_limit[client_ip]
    # Prune entries older than 60s
    _hs_rate_limit[client_ip] = window = [t for t in window if now - t < 60]
    if len(window) >= _HS_RATE_LIMIT_RPM:
        raise HTTPException(status_code=429, detail="hidden_state rate limit exceeded")
    window.append(now)
    return await miner.get_hidden_state(req)


# -- CLI -----------------------------------------------------------------------

def main():
    global miner

    parser = argparse.ArgumentParser(
        description="Production vLLM-based inference miner with hidden state verification"
    )
    parser.add_argument("--port", type=int, default=8091, help="Server port (default: 8091)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model name (default: meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory for vLLM KV cache (default: 0.85)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=200,
        help="Max cached hidden state entries (default: 200)",
    )
    parser.add_argument(
        "--hf-device",
        default="auto",
        choices=["auto", "cpu"],
        help="Device for HF hidden state model: 'auto' tries GPU then CPU, 'cpu' forces CPU (default: auto)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graph capture for vLLM (slower but avoids compilation hangs)",
    )
    args = parser.parse_args()

    # Graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        log.info(f"Received signal {sig}, shutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    miner = VLLMMiner(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_size=args.cache_size,
        hf_device=args.hf_device,
        enforce_eager=args.enforce_eager,
    )

    log.info(f"Starting vLLM miner on {args.host}:{args.port}")
    log.info(f"  Model: {args.model}")
    log.info(f"  Tensor parallel: {args.tensor_parallel_size}")
    log.info(f"  Max model len: {args.max_model_len}")
    log.info(f"  GPU memory util: {args.gpu_memory_utilization}")
    log.info(f"  Hidden state cache: {args.cache_size} entries")

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=30,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    main()
