#!/usr/bin/env python3
"""
Hardened Gateway Validator — Exploit-resistant inference routing and verification.

This is the main entry point for the inference subnet. It serves two roles:
1. OpenAI-compatible inference endpoint for organic traffic
2. Validator that scores miners and sets on-chain weights

Key hardening features:
- Challenges indistinguishable from organic traffic (same format, timing, prompts)
- Cryptographically unpredictable challenge parameters
- Multi-dimensional scoring (speed × verification × consistency)
- Asymmetric penalties (cheating costs more than honest operation)
- KV cache-aware session routing (multi-turn conversations stick to same miner)
- Intelligent load balancing (weighted by reliability + speed)
- Rate limiting and auth for organic endpoint
- Comprehensive audit logging

Architecture:
  External clients → POST /v1/chat/completions (OpenAI-compatible)
                   → Hardened Gateway
                   → Intelligent Router (selects best miner)
                   → Miner (serves inference + hidden states)
                   → Challenge Engine (verifies execution)
                   → Scoring Engine (records points)
                   → R2 Publisher (audit log)
"""

import argparse
import asyncio
import hashlib
import hmac
import html as html_mod
import json
import logging
import os
import secrets
import sys
import time
import uuid
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional, Union

import aiohttp
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from model import MockModel, ModelConfig
from hardened_scoring import (
    HardenedScoringEngine, RequestScore, ChallengeResult,
    cosine_similarity, compute_speed_score, compute_output_quality,
    compute_verification_score,
    COSINE_THRESHOLD, CHALLENGE_TIMEOUT_MS, CHALLENGE_TIMEOUT_HARD_MS,
)
from challenge_engine import ChallengeEngine
from r2_publisher import R2Publisher, AuditRecord
from kv_cache_prober import (
    KVCacheProber, CacheProbeResult, compute_cache_score,
    generate_probe_pair,
    PROBE_DELAY_MIN_S, PROBE_DELAY_MAX_S,
)
from collusion_detector import (
    CollusionDetector, MinerTimingSample, MinerErrorEvent,
)

log = logging.getLogger("hardened_gateway")
log.setLevel(logging.INFO)
# Force exactly one handler to sys.stdout — clear any duplicates
log.handlers.clear()
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_handler)
log.propagate = False
# Prevent root logger from adding a second output path
logging.root.handlers.clear()
logging.root.addHandler(logging.NullHandler())

# Version tracking for watchtower/deployment debugging
GATEWAY_VERSION = "0.3.0"
GATEWAY_START_TIME = time.time()

# ── Miner Response Signature Verification ────────────────────────────────────
# Verifies that the miner response was actually signed by the on-chain hotkey
# for that UID. Prevents impersonation attacks where an attacker points their
# axon at another miner's IP:port to steal rewards without running a GPU.

# Grace period: log warnings but don't reject unsigned responses.
# Set REQUIRE_MINER_SIGNATURES=true once all miners are upgraded.
_REQUIRE_MINER_SIGNATURES = os.environ.get("REQUIRE_MINER_SIGNATURES", "false").lower() == "true"
_sig_verify_pass = 0
_sig_verify_fail = 0
_sig_verify_missing = 0


def verify_miner_signature(
    resp_headers: dict,
    request_id: str,
    response_body: bytes,
    expected_hotkey: str,
) -> tuple[bool, str]:
    """Verify miner response signature.

    The miner signs sha256(request_id + response_body) with its hotkey.
    We verify using the on-chain hotkey (ss58 address) for that UID.

    Returns (valid, reason). valid=True means signature checks out or
    signing is not yet required.
    """
    global _sig_verify_pass, _sig_verify_fail, _sig_verify_missing

    hotkey_header = resp_headers.get("X-Miner-Hotkey", "")
    sig_header = resp_headers.get("X-Miner-Signature", "")

    if not sig_header or not hotkey_header:
        _sig_verify_missing += 1
        if _REQUIRE_MINER_SIGNATURES:
            return False, "missing_signature"
        return True, "unsigned_allowed"

    # Check hotkey matches what's registered on-chain for this UID
    if expected_hotkey and hotkey_header != expected_hotkey:
        _sig_verify_fail += 1
        return False, f"hotkey_mismatch: got {hotkey_header[:16]}... expected {expected_hotkey[:16]}..."

    # Verify signature
    try:
        from bittensor import Keypair
        kp = Keypair(ss58_address=hotkey_header)
        msg = hashlib.sha256(request_id.encode() + response_body).digest()
        sig_bytes = bytes.fromhex(sig_header.replace("0x", ""))
        if kp.verify(msg, sig_bytes):
            _sig_verify_pass += 1
            return True, "verified"
        else:
            _sig_verify_fail += 1
            return False, "invalid_signature"
    except Exception as e:
        _sig_verify_fail += 1
        return False, f"verify_error: {e}"


# ── Chain Weight Setter ──────────────────────────────────────────────────────

class MetagraphDiscovery:
    """
    Discovers miners from the Bittensor metagraph via subprocess.
    Same subprocess pattern as ChainWeightSetter to avoid blocking asyncio.
    """

    def __init__(self, netuid: int, network: str, validator_hotkey: str = None):
        self.netuid = netuid
        self.network = network
        self.validator_hotkey = validator_hotkey
        self.last_sync = 0.0
        self.sync_interval = 120  # seconds between syncs

    async def discover_miners(self) -> list[dict]:
        """
        Query metagraph and return list of miner dicts:
        [{"uid": int, "endpoint": str, "hotkey": str}, ...]
        """
        script = f"""
import json, sys
try:
    import bittensor as bt
    import numpy as np
    sub = bt.Subtensor(network="{self.network}")
    meta = bt.Metagraph(netuid={self.netuid}, network=sub.network, sync=True)
    # Miners = neurons with zero dividends (non-validators)
    dividends = np.array([float(d) for d in meta.dividends])
    miners = []
    for uid in range(meta.n):
        if dividends[uid] > 0:
            continue  # skip validators
        hotkey = str(meta.hotkeys[uid])
        if hotkey == "{self.validator_hotkey or ''}":
            continue  # skip ourselves
        axon = meta.axons[uid]
        ip = str(getattr(axon, 'ip', ''))
        port = int(getattr(axon, 'port', 0))
        if ip and port and ip != '0.0.0.0':
            miners.append({{"uid": int(uid), "endpoint": f"http://{{ip}}:{{port}}", "hotkey": hotkey}})
    print(json.dumps(miners))
except Exception as e:
    print(f"ERR:{{e}}", file=sys.stderr)
    sys.exit(1)
"""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode == 0:
                miners = json.loads(stdout.decode().strip())
                self.last_sync = time.time()
                log.info(f"[METAGRAPH] Discovered {len(miners)} miners on netuid {self.netuid}")
                return miners
            else:
                log.error(f"[METAGRAPH] Discovery failed: {stderr.decode().strip()}")
                return []
        except asyncio.TimeoutError:
            log.error("[METAGRAPH] Discovery timed out (60s)")
            try:
                proc.kill()
            except Exception:
                pass
            return []
        except Exception as e:
            log.error(f"[METAGRAPH] Discovery error: {e}")
            return []


class ChainWeightSetter:
    """
    Sets weights on the Bittensor chain via subprocess to avoid blocking asyncio.
    The bittensor SDK uses synchronous websockets that conflict with asyncio's event loop.
    """

    def __init__(self, wallet_name: str, hotkey: str, netuid: int, network: str, wallet_path: str = None):
        self.wallet_name = wallet_name
        self.hotkey = hotkey
        self.netuid = netuid
        self.network = network
        self.wallet_path = wallet_path or os.path.expanduser("~/.bittensor/wallets")
        self.last_set_time = 0.0
        self.total_sets = 0
        self.total_failures = 0

    async def set_weights(self, weights: dict[int, float], retries: int = 3) -> bool:
        """Set weights on chain via subprocess with retry. Returns True on success."""
        if not weights:
            log.warning("[CHAIN] No weights to set")
            return False

        uids = list(weights.keys())
        weight_values = [weights[uid] for uid in uids]

        # Build a Python script to run in subprocess
        script = f"""
import sys
import numpy as np
try:
    import bittensor as bt
    wallet = bt.Wallet(name="{self.wallet_name}", hotkey="{self.hotkey}", path="{self.wallet_path}")
    sub = bt.Subtensor(network="{self.network}")
    response = sub.set_weights(
        wallet=wallet,
        netuid={self.netuid},
        uids=np.array({uids}, dtype=np.int64),
        weights=np.array({weight_values}, dtype=np.float32),
        wait_for_finalization=True,
        period=None,
    )
    # response is (bool, message) — check success flag
    if isinstance(response, (tuple, list)):
        success, msg = response[0], response[1] if len(response) > 1 else ""
        if success:
            print(f"OK:{{response}}")
        else:
            print(f"ERR:set_weights returned False: {{msg}}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"OK:{{response}}")
except Exception as e:
    print(f"ERR:{{e}}", file=sys.stderr)
    sys.exit(1)
"""

        for attempt in range(1 + retries):
            if attempt > 0:
                log.info(f"[CHAIN] Retry {attempt}/{retries} for weight setting...")
                await asyncio.sleep(5)

            log.info(f"[CHAIN] Setting weights for {len(uids)} miners on netuid {self.netuid}...")
            log.info(f"[CHAIN] Weights: {dict(zip(uids, [f'{w:.4f}' for w in weight_values]))}")

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", script,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)

                if proc.returncode == 0:
                    result = stdout.decode().strip()
                    log.info(f"[CHAIN] Weights set successfully: {result}")
                    self.last_set_time = time.time()
                    self.total_sets += 1
                    return True
                else:
                    err = stderr.decode().strip()
                    log.error(f"[CHAIN] Weight setting failed: {err}")
                    self.total_failures += 1
            except asyncio.TimeoutError:
                log.error(f"[CHAIN] Weight setting timed out (120s), attempt {attempt + 1}/{1 + retries}")
                self.total_failures += 1
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception as e:
                log.error(f"[CHAIN] Weight setting error: {e}")
                self.total_failures += 1

        return False


# ── Real Model for Validator Verification ────────────────────────────────────

class RealValidatorModel:
    """
    Wraps a real HuggingFace transformer model for hidden state verification.
    Provides the same interface as MockModel so the gateway can use either.
    """

    def __init__(self, model_name: str):
        import threading
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._lock = threading.Lock()
        trust_remote = os.getenv("TRUST_REMOTE_CODE", "0") == "1"
        log.info(f"Loading validator model: {model_name} (trust_remote_code={trust_remote})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=trust_remote,
        )
        self.model.eval()
        self._torch = torch

        class _Config:
            name = model_name
            num_layers = self.model.config.num_hidden_layers
            hidden_dim = self.model.config.hidden_size
            vocab_size = self.model.config.vocab_size
            max_seq_len = getattr(self.model.config, "max_position_embeddings", 2048)

        self.config = _Config()
        log.info(f"Validator model loaded: {model_name} | {self.config.num_layers} layers | hidden_dim={self.config.hidden_dim}")

    @property
    def _device(self):
        return next(self.model.parameters()).device

    def generate(self, prompt: str, max_tokens: int = 64) -> dict:
        """Run inference and return tokens + hidden states."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        input_len = inputs["input_ids"].shape[1]
        input_tokens = inputs["input_ids"][0].tolist()

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
            )

        output_ids = outputs.sequences[0][input_len:].tolist()
        all_tokens = input_tokens + output_ids
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Build hidden state dict: {layer: {position: ndarray}}
        hidden_states = {}
        if outputs.hidden_states:
            prefill_states = outputs.hidden_states[0]
            for layer_idx in range(self.config.num_layers):
                layer_tensor = prefill_states[layer_idx + 1][0].cpu().float()
                hidden_states[layer_idx] = {}
                for pos in range(layer_tensor.shape[0]):
                    vec = layer_tensor[pos].numpy()
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    hidden_states[layer_idx][pos] = vec

            for step_idx in range(1, len(outputs.hidden_states)):
                step_states = outputs.hidden_states[step_idx]
                pos = input_len + step_idx - 1
                for layer_idx in range(self.config.num_layers):
                    vec = step_states[layer_idx + 1][0][0].cpu().float().numpy()
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                    hidden_states[layer_idx][pos] = vec

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_ids,
            "all_tokens": all_tokens,
            "hidden_states": hidden_states,
            "text": text,
        }

    def compute_hidden_state_at(self, tokens: list[int], layer: int, position: int) -> np.ndarray:
        """Compute hidden state at a single (layer, position) for verification.
        Thread-safe: serialized via lock to prevent concurrent forward pass corruption."""
        with self._lock:
            input_ids = self._torch.tensor([tokens[:position + 1]], device=self._device)
            with self._torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
            vec = outputs.hidden_states[layer + 1][0][-1].cpu().float().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def compute_hidden_states_batch(self, tokens: list[int], points: list[tuple[int, int]]) -> dict:
        """Compute hidden states at multiple (layer, position) points in a single forward pass.
        Thread-safe: serialized via lock to prevent concurrent forward pass corruption."""
        if not points:
            return {}
        with self._lock:
            max_pos = max(pos for _, pos in points)
            input_ids = self._torch.tensor([tokens[:max_pos + 1]], device=self._device)
            with self._torch.no_grad():
                outputs = self.model(input_ids, output_hidden_states=True)
            result = {}
            for layer, pos in points:
                vec = outputs.hidden_states[layer + 1][0][pos].cpu().float().numpy()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                result[(layer, pos)] = vec
        return result


def load_validator_model(model_name: str = None):
    """Load either a real or mock model for the validator."""
    if model_name and model_name != "mock":
        return RealValidatorModel(model_name)
    return MockModel(ModelConfig())


# ── Configuration ────────────────────────────────────────────────────────────

class GatewayConfig:
    # Challenge rate: fraction of requests that get hidden state verification
    # In production, this should be 1.0 (challenge every request)
    CHALLENGE_RATE = 1.0

    # Synthetic probe configuration
    SYNTHETIC_RATE = 0.3        # 30% of validator-initiated requests are synthetic
    SYNTHETIC_INTERVAL_S = 8    # Randomized: ±50% jitter applied
    SYNTHETIC_MIN_INTERVAL_S = 3
    SYNTHETIC_MAX_INTERVAL_S = 15

    # Model verification
    COSINE_THRESHOLD = COSINE_THRESHOLD
    CHALLENGE_TIMEOUT_MS = CHALLENGE_TIMEOUT_MS

    # Inference
    INFERENCE_TIMEOUT_S = 120
    MAX_MINER_RESPONSE_BYTES = 10 * 1024 * 1024  # 10MB max response from miners
    MAX_CONTEXT_TOKENS = 32768  # Max context window (prompt + output tokens)

    # Epoch
    EPOCH_LENGTH_S = 60  # PoC: 60s, production: 4320s

    # Gateway
    GATEWAY_PORT = 8080

    # Auth
    API_KEYS: set[str] = set()  # Empty = no auth required for inference
    MONITORING_KEYS: set[str] = set()  # Empty = monitoring endpoints open; set to restrict
    INTERNAL_RELAY_SECRET: str = ""  # Shared secret for auditor relay (empty = relay disabled)
    MINER_VALIDATOR_SECRET: str = ""  # Shared secret for miner auth (empty = no auth header sent)
    RATE_LIMIT_RPM = 600        # Requests per minute per API key (high for C=200 target)

    # KV cache probing
    CACHE_PROBE_INTERVAL_S = 30     # Base interval between cache probes
    CACHE_PROBE_JITTER = 0.5        # ±50% jitter

    # Synthetic prompts — MUST be indistinguishable from organic traffic
    # Topics span ALL domains real users ask about, not just CS/ML.
    # Distribution targets: ~25% tech, ~20% creative/writing, ~15% general knowledge,
    # ~15% practical/how-to, ~15% personal/casual, ~10% science/academic
    _TOPICS = [
        # Tech / CS / ML (~25%)
        "quantum computing", "merge sort", "proof of stake",
        "transformer architecture", "backpropagation",
        "natural language processing", "attention mechanisms", "RLHF",
        "hash tables", "CAP theorem", "bloom filters",
        "TCP vs UDP", "git internals", "database indexing",
        "consensus algorithms", "zero-knowledge proofs",
        "garbage collection", "load balancing", "rate limiting",
        "container orchestration", "distributed tracing",
        "diffusion models", "mixture of experts",
        "flash attention", "quantization", "vector databases", "RAG systems",
        "REST API design", "microservices", "Docker", "Kubernetes",
        "React hooks", "TypeScript generics", "Python async",
        "SQL vs NoSQL", "OAuth 2.0", "WebSockets",
        # Creative / Writing (~20%)
        "writing a compelling opening paragraph", "character development in fiction",
        "worldbuilding for a fantasy novel", "poetry forms and meter",
        "screenwriting dialogue tips", "how to write a mystery plot",
        "creative writing prompts", "story pacing and structure",
        "writing persuasive essays", "journaling techniques",
        "songwriting basics", "short story endings",
        "writing a cover letter", "blogging tips",
        "how to give a good presentation", "public speaking",
        # General knowledge (~15%)
        "how the stock market works", "the history of the internet",
        "how vaccines work", "climate change causes and effects",
        "the solar system", "how airplanes fly",
        "the French Revolution", "how credit scores work",
        "the water cycle", "how elections work in the US",
        "ancient Rome", "the periodic table",
        "how GPS works", "the history of jazz",
        # Practical / How-to (~15%)
        "how to make sourdough bread", "basic car maintenance",
        "home gardening tips", "budgeting and saving money",
        "meal prepping for beginners", "fixing a leaky faucet",
        "time management techniques", "how to negotiate a raise",
        "learning a new language", "beginner yoga poses",
        "planning a road trip", "how to train a puppy",
        "organizing a small apartment", "basic first aid",
        # Personal / Casual (~15%)
        "dealing with procrastination", "work-life balance",
        "making friends as an adult", "handling job interviews",
        "dealing with stress", "career change advice",
        "how to be more productive", "building good habits",
        "improving sleep quality", "managing email overload",
        "staying motivated", "overcoming imposter syndrome",
        "remote work tips", "small talk advice",
        # Science / Academic (~10%)
        "photosynthesis", "black holes", "plate tectonics",
        "CRISPR gene editing", "the theory of relativity",
        "evolution by natural selection", "neurotransmitters",
        "the Fibonacci sequence in nature", "entropy",
        "game theory basics", "behavioral economics",
    ]
    _STYLES = [
        "Explain {topic} in simple terms.",
        "Write a detailed explanation of {topic}.",
        "What are the key trade-offs in {topic}?",
        "How does {topic} work under the hood?",
        "Compare two approaches to {topic}.",
        "Give a practical example of {topic}.",
        "What are common misconceptions about {topic}?",
        "Describe {topic} as if teaching a beginner.",
        "What problems does {topic} solve, and what are its limitations?",
        "Why is {topic} important?",
        "Summarize the history and evolution of {topic}.",
        "What should a beginner know about {topic}?",
        "Write a short introduction to {topic}.",
        "What are the most important things about {topic}?",
        "What are the pros and cons of {topic}?",
        "How do experts think about {topic}?",
    ]
    _CONSTRAINTS = [
        "",  # No constraint (most common)
        "",
        "",
        " Keep it under 100 words.",
        " Use a concrete code example.",
        " Focus on performance characteristics.",
        " Include the mathematical intuition.",
        " Explain it with an analogy.",
        " Address common pitfalls.",
        " Compare with an alternative approach.",
    ]
    _PERSONAS = [
        "",  # No persona (most common)
        "",
        "",
        "",
        "You are a helpful tutor. ",
        "You are a knowledgeable friend. ",
        "You are a professional consultant. ",
        "You are explaining this to someone new to the topic. ",
    ]


# ── OpenAI-Compatible Request/Response Models ───────────────────────────────

VALID_ROLES = {"system", "user", "assistant", "tool", "function"}

class ChatMessage(BaseModel):
    role: str = Field(..., max_length=32)
    content: Union[str, list] = Field(..., max_length=100_000)  # str or [{type:"text",text:"..."}]

    @property
    def text(self) -> str:
        """Extract plain text from content (handles OpenAI multimodal format)."""
        if isinstance(self.content, str):
            return self.content
        # List of content parts — extract text parts, ignore images etc.
        parts = []
        for part in self.content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)

class ChatCompletionRequest(BaseModel):
    model: str = Field("default", max_length=128, pattern=r'^[a-zA-Z0-9][a-zA-Z0-9/_.\-:]*$')
    messages: list[ChatMessage] = Field(..., max_length=256)  # Max 256 messages
    max_tokens: int = Field(2048, ge=1, le=32768)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False
    # Session ID for KV cache routing (multi-turn conversations)
    session_id: Optional[str] = Field(None, max_length=256)
    # Common OpenAI params — accepted for compatibility, forwarded to miner where applicable
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    stop: Optional[list[str] | str] = Field(None, max_length=16)  # Max 16 stop sequences
    user: Optional[str] = None
    n: Optional[int] = Field(None, ge=1, le=1)  # Only n=1 supported

    model_config = {"extra": "ignore"}  # Silently ignore unknown fields (logit_bias, etc.)

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage

class StreamChoice(BaseModel):
    index: int = 0
    delta: dict
    finish_reason: Optional[str] = None


# ── Miner Pool & Routing ────────────────────────────────────────────────────

@dataclass
class MinerInfo:
    uid: int
    endpoint: str
    hotkey: str = ""
    alive: bool = True
    last_seen: float = 0.0
    requests_served: int = 0
    requests_failed: int = 0
    avg_ttft_ms: float = 0.0
    avg_tps: float = 0.0
    reliability_score: float = 1.0  # 0-1, EMA of success rate
    active_requests: int = 0  # In-flight requests for load balancing


class SessionRouter:
    """
    KV cache-aware session routing.

    Multi-turn conversations should go to the same miner to benefit from
    KV cache reuse. This router maintains session → miner affinity.

    Anti-gaming: Sessions expire and affinity is soft (can be overridden
    if the miner is down or overloaded).
    """

    def __init__(self, session_ttl_s: float = 300.0, max_sessions: int = 10000):
        self.session_ttl_s = session_ttl_s
        self.max_sessions = max_sessions
        # session_id → (miner_uid, last_access_time)
        self._sessions: OrderedDict[str, tuple[int, float]] = OrderedDict()

    def get_affinity(self, session_id: str) -> Optional[int]:
        """Get the miner UID this session has affinity to, or None."""
        try:
            miner_uid, last_access = self._sessions[session_id]
        except KeyError:
            return None
        if time.time() - last_access > self.session_ttl_s:
            self._sessions.pop(session_id, None)
            return None
        try:
            self._sessions.move_to_end(session_id)
        except KeyError:
            pass  # Evicted between lookup and move — benign
        return miner_uid

    def set_affinity(self, session_id: str, miner_uid: int):
        """Set session affinity to a miner."""
        now = time.time()
        # Evict expired sessions periodically (every 100 new sessions)
        if len(self._sessions) >= self.max_sessions:
            # First try evicting expired sessions (snapshot to avoid dict-changed-during-iteration)
            expired = [sid for sid, (_, t) in list(self._sessions.items()) if now - t > self.session_ttl_s]
            for sid in expired[:100]:  # Batch cleanup
                self._sessions.pop(sid, None)
            # If still full, evict oldest
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)
        self._sessions[session_id] = (miner_uid, now)

    def remove_miner(self, miner_uid: int):
        """Remove all session affinities for a dead miner."""
        to_remove = [sid for sid, (uid, _) in list(self._sessions.items()) if uid == miner_uid]
        for sid in to_remove:
            self._sessions.pop(sid, None)


class IntelligentRouter:
    """
    Intelligent request routing across miners.

    Routing strategy:
    1. Skip miners blocked by scoring (negative points, low pass_rate)
    2. If session has affinity → prefer that miner (KV cache reuse)
    3. Otherwise → weighted random by (reliability × speed × availability)
       where speed = 0.4 * TTFT_score + 0.6 * TPS_score
    4. Fallback → least loaded miner

    Anti-gaming: Routing decisions are not predictable by miners.
    The routing weights use cryptographic randomness and are
    not disclosed to miners.
    """

    # Minimum TPS threshold — miners below this with enough samples are deprioritized.
    # 5 TPS is well below anything healthy (even slow 4090s do 20+ TPS).
    # Miners at 1-4 TPS are either broken, OOM, or serving from CPU fallback.
    MIN_TPS_FLOOR = 5.0
    MIN_TPS_SAMPLES = 3  # Need at least N served requests before judging

    def __init__(self, miners: dict[int, MinerInfo]):
        self.miners = miners
        self.session_router = SessionRouter()
        self._reliability_ema_alpha = 0.1  # Slow EMA for reliability
        self._speed_ema_alpha = 0.2  # Faster EMA for TPS/TTFT — adapts in ~5 requests
        self._blocked_uids: set[int] = set()  # UIDs blocked by scoring engine
        self._auditor_blocked: set[int] = set()  # UIDs blocked by external auditor

    def update_blocked_uids(self, scoring_engine):
        """Update blocked UIDs from local scoring engine — merges with auditor blocks."""
        local_blocked = set()
        for entry in scoring_engine.get_scoreboard():
            uid = entry["uid"]
            net_points = entry.get("net_points", 0)
            pass_rate = entry.get("pass_rate", 1.0)
            total_challenged = entry.get("passed_challenges", 0) + entry.get("failed_challenges", 0)
            # Block miners with negative net_points (confirmed cheaters)
            if net_points < 0:
                local_blocked.add(uid)
            # Block miners with low pass_rate and enough samples
            elif total_challenged >= 4 and pass_rate < 0.7:
                local_blocked.add(uid)
        merged = local_blocked | self._auditor_blocked
        if merged != self._blocked_uids:
            log.info(f"[ROUTER] Blocked UIDs updated: {merged} (local={local_blocked}, auditor={self._auditor_blocked})")
        self._blocked_uids = merged

    def update_auditor_blocked(self, blocked_uids: set[int]):
        """Update blocked UIDs from external auditor — persists across local resets."""
        self._auditor_blocked = blocked_uids
        merged = self._blocked_uids | blocked_uids
        if merged != self._blocked_uids:
            log.info(f"[ROUTER] Blocked UIDs updated from auditor: {merged}")
        self._blocked_uids = merged

    def _is_stalled_miner(self, m: MinerInfo) -> bool:
        """Check if miner is below the TPS floor (broken/stalled)."""
        return (m.requests_served >= self.MIN_TPS_SAMPLES and
                m.avg_tps > 0 and
                m.avg_tps < self.MIN_TPS_FLOOR)

    def _compute_speed_factor(self, m: MinerInfo) -> float:
        """Combined speed factor: 40% TTFT + 60% TPS, squared for aggressive differentiation.

        Speed is king: a 200 TPS miner should get ~4x the routing weight of a 50 TPS miner,
        not ~2x. Squaring the combined score amplifies the gap so fast miners get
        proportionally more traffic — which directly serves the "more compute = more
        incentive" goal.
        """
        # Hard floor: stalled miners (< 5 TPS with enough samples) get near-zero weight
        if self._is_stalled_miner(m):
            return 0.0001  # Effectively starved — only used if ALL other miners are worse

        # TTFT score: lower is better, normalize to [0, 1]
        TTFT_EXCELLENT_MS = 50.0    # Best observed (local fast miners)
        TTFT_POOR_MS = 1000.0       # Unacceptable for user experience
        if m.avg_ttft_ms > 0:
            ttft_score = max(0.05, min(1.0, 1.0 - (m.avg_ttft_ms - TTFT_EXCELLENT_MS) /
                                                     (TTFT_POOR_MS - TTFT_EXCELLENT_MS)))
        else:
            ttft_score = 0.5  # Unknown TTFT — neutral

        # TPS score: higher is better, cap at 250 for wider differentiation
        tps_score = max(0.1, min(1.0, m.avg_tps / 250.0)) if m.avg_tps > 0 else 0.5

        # Weight TPS higher (60%) — throughput is the dominant user experience factor
        combined = 0.4 * ttft_score + 0.6 * tps_score
        # Square for aggressive differentiation: 0.8 → 0.64, 0.4 → 0.16
        return combined ** 2

    def get_router_stats(self) -> list[dict]:
        """Return current routing weights for all alive miners (debug/monitoring)."""
        stats = []
        for uid, m in sorted(self.miners.items()):
            if not m.alive:
                continue
            sf = self._compute_speed_factor(m)
            lf = 0.5 ** m.active_requests
            w = m.reliability_score * lf * sf
            stats.append({
                "uid": uid,
                "avg_tps": round(m.avg_tps, 1),
                "avg_ttft_ms": round(m.avg_ttft_ms, 1),
                "reliability": round(m.reliability_score, 3),
                "active_requests": m.active_requests,
                "requests_served": m.requests_served,
                "speed_factor": round(sf, 4),
                "load_factor": round(lf, 4),
                "routing_weight": round(max(w, 0.001), 4),
                "stalled": self._is_stalled_miner(m),
                "blocked": uid in self._blocked_uids,
            })
        return stats

    def select_miner(self, session_id: Optional[str] = None) -> Optional[MinerInfo]:
        """Select the best miner for a request."""
        # Snapshot to avoid RuntimeError if discovery_loop modifies dict concurrently
        # Exclude blocked UIDs (negative scorers, low pass_rate)
        alive_miners = {uid: m for uid, m in list(self.miners.items())
                        if m.alive and uid not in self._blocked_uids}
        if not alive_miners:
            # Fallback: try all alive miners if blocking leaves none
            alive_miners = {uid: m for uid, m in list(self.miners.items()) if m.alive}
        if not alive_miners:
            return None

        # Session affinity — only honour if miner is responsive, fast, and not overloaded
        if session_id:
            affinity_uid = self.session_router.get_affinity(session_id)
            if affinity_uid is not None and affinity_uid in alive_miners:
                miner = alive_miners[affinity_uid]
                if (miner.active_requests < 5 and
                    miner.reliability_score > 0.7 and
                    (miner.avg_ttft_ms <= 0 or miner.avg_ttft_ms < 1000)):
                    return miner

        # Weighted selection based on reliability, load, and speed (TTFT + TPS)
        # Load factor uses exponential decay so heavily loaded miners are deprioritized
        # faster — this directly improves TTFT for end users.
        weights = []
        miner_list = list(alive_miners.values())
        for m in miner_list:
            load_factor = 0.5 ** m.active_requests  # 1.0, 0.5, 0.25, 0.125, ...
            speed_factor = self._compute_speed_factor(m)
            w = m.reliability_score * load_factor * speed_factor
            # Only add positive weights — no minimum floor for cheaters
            weights.append(max(w, 0.001))

        total_w = sum(weights)
        if total_w == 0:
            return miner_list[0]

        # Weighted random selection (cryptographic randomness)
        r = secrets.randbelow(10000) / 10000.0 * total_w
        cumulative = 0.0
        for m, w in zip(miner_list, weights):
            cumulative += w
            if r <= cumulative:
                return m

        return miner_list[-1]

    def report_success(self, miner: MinerInfo, ttft_ms: float, tps: float):
        """Update miner stats after successful request."""
        miner.last_seen = time.time()
        miner.requests_served += 1
        miner.active_requests = max(0, miner.active_requests - 1)

        # EMA update for reliability (slow — alpha=0.1)
        alpha = self._reliability_ema_alpha
        miner.reliability_score = miner.reliability_score * (1 - alpha) + 1.0 * alpha

        # EMA update for speed metrics (faster — alpha=0.2 adapts in ~5 requests)
        # Running avg (old code) froze after N>>1 — a miner with 100 requests needed
        # 100 more to shift its avg meaningfully. EMA gives recent data 20% weight always.
        sa = self._speed_ema_alpha
        if miner.avg_ttft_ms <= 0:
            miner.avg_ttft_ms = ttft_ms  # First measurement — initialize directly
        else:
            miner.avg_ttft_ms = miner.avg_ttft_ms * (1 - sa) + ttft_ms * sa
        if miner.avg_tps <= 0:
            miner.avg_tps = tps  # First measurement
        else:
            miner.avg_tps = miner.avg_tps * (1 - sa) + tps * sa

    def report_failure(self, miner: MinerInfo, timeout: bool = False):
        """Update miner stats after failed request.

        Args:
            timeout: If True, use softer penalty (timeouts on large contexts are expected).
        """
        miner.requests_failed += 1
        miner.active_requests = max(0, miner.active_requests - 1)

        # Aggressive decay: use higher alpha for consecutive failures
        # If miner has never served successfully, mark dead after 3 failures
        if miner.requests_served == 0 and miner.requests_failed >= 3:
            miner.reliability_score = 0.0
        else:
            # Timeouts get softer penalty (0.1 alpha) — they're expected for large contexts
            # Hard failures get 0.3 alpha so dead miners are detected in ~7 failures
            alpha = 0.1 if timeout else 0.3
            miner.reliability_score = miner.reliability_score * (1 - alpha)

        # Mark dead if too unreliable
        if miner.reliability_score < 0.1:
            miner.alive = False
            miner._death_time = time.time()
            self.session_router.remove_miner(miner.uid)
            log.warning(f"Miner {miner.uid}: marked DEAD (reliability={miner.reliability_score:.3f})")

    def select_miner_excluding(self, exclude_uids: set, session_id: Optional[str] = None) -> Optional[MinerInfo]:
        """Select a miner excluding specific UIDs (used for failover)."""
        all_excluded = exclude_uids | self._blocked_uids
        alive_miners = {uid: m for uid, m in list(self.miners.items()) if m.alive and uid not in all_excluded}
        if not alive_miners:
            # Fallback: only apply explicit exclusions
            alive_miners = {uid: m for uid, m in list(self.miners.items()) if m.alive and uid not in exclude_uids}
        if not alive_miners:
            return None
        # Same weighted logic as select_miner but without session affinity
        weights = []
        miner_list = list(alive_miners.values())
        for m in miner_list:
            load_factor = 0.5 ** m.active_requests  # Match select_miner's exponential decay
            speed_factor = self._compute_speed_factor(m)
            w = m.reliability_score * load_factor * speed_factor
            weights.append(max(w, 0.001))
        total_w = sum(weights)
        if total_w == 0:
            return miner_list[0]
        r = secrets.randbelow(10000) / 10000.0 * total_w
        cumulative = 0.0
        for m, w in zip(miner_list, weights):
            cumulative += w
            if r <= cumulative:
                return m
        return miner_list[-1]

    async def health_check_dead_miners(self, session: aiohttp.ClientSession = None):
        """Periodically retry dead miners to see if they've recovered.

        Uses the provided shared session to avoid creating temporary sessions
        that leak connectors. Checks all dead miners CONCURRENTLY to avoid
        blocking the event loop (sequential checks with 5s timeout × N dead
        miners would block for N×5s).
        """
        dead_miners = [m for m in list(self.miners.values()) if not m.alive]
        eligible = []
        for miner in dead_miners:
            death_time = getattr(miner, '_death_time', 0)
            if time.time() - death_time < 30:
                continue
            eligible.append(miner)
        if not eligible:
            return

        async def _check_one(sem, miner):
            async with sem:
                try:
                    _session = session
                    _tmp = None
                    if not _session:
                        _conn = aiohttp.TCPConnector(limit=10, resolver=aiohttp.AsyncResolver())
                        _tmp = aiohttp.ClientSession(connector=_conn)
                        _session = _tmp
                    try:
                        async with _session.get(
                            f"{miner.endpoint}/health",
                            timeout=aiohttp.ClientTimeout(total=3),
                        ) as resp:
                            if resp.status == 200:
                                miner.alive = True
                                miner.reliability_score = 0.3
                                log.info(f"Miner {miner.uid}: RECOVERED (was dead, now alive)")
                    finally:
                        if _tmp:
                            await _tmp.close()
                except Exception as e:
                    log.debug(f"Miner {miner.uid}: health check failed ({type(e).__name__})")

        sem = asyncio.Semaphore(10)  # Max 10 concurrent health checks
        await asyncio.gather(*[_check_one(sem, m) for m in eligible])

    def add_miner(self, uid: int, endpoint: str, hotkey: str = "") -> MinerInfo:
        """Add a new miner dynamically (e.g., from metagraph discovery)."""
        if uid in self.miners:
            # Update endpoint if changed
            existing = self.miners[uid]
            if existing.endpoint != endpoint:
                log.info(f"Miner {uid}: endpoint changed {existing.endpoint} → {endpoint}")
                existing.endpoint = endpoint
            if hotkey and existing.hotkey != hotkey:
                existing.hotkey = hotkey
            return existing
        miner = MinerInfo(uid=uid, endpoint=endpoint, hotkey=hotkey)
        self.miners[uid] = miner
        log.info(f"Miner {uid}: added (endpoint={endpoint})")
        return miner

    def remove_stale_miners(self, active_uids: set):
        """Remove miners no longer in the metagraph."""
        to_remove = [uid for uid in list(self.miners) if uid not in active_uids]
        for uid in to_remove:
            self.session_router.remove_miner(uid)
            del self.miners[uid]
            log.info(f"Miner {uid}: removed (no longer in metagraph)")


# ── Timing-safe key check ────────────────────────────────────────────────────

def _timing_safe_key_in(candidate: str, valid_keys: set[str]) -> bool:
    """Check if candidate is in valid_keys using constant-time comparison.
    Iterates ALL keys to prevent timing leaks about which key matched."""
    found = False
    for k in valid_keys:
        if hmac.compare_digest(candidate.encode(), k.encode()):
            found = True
    return found


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """Sliding-window rate limiter with remaining/reset info for headers."""

    def __init__(self, max_rpm: int = 120):
        self.max_rpm = max_rpm
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Evict stale entries every 5 minutes

    def _maybe_cleanup(self, now: float):
        """Evict stale keys that have no recent activity to prevent memory leak."""
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        stale = [k for k, ts in self._windows.items() if not ts or now - ts[-1] > 120]
        for k in stale:
            del self._windows[k]

    def check(self, key: str) -> bool:
        """Returns True if request is allowed."""
        now = time.time()
        self._maybe_cleanup(now)
        # Build filtered list and check+append atomically (single reference swap)
        filtered = [t for t in self._windows[key] if now - t < 60]
        if len(filtered) >= self.max_rpm:
            self._windows[key] = filtered
            return False
        filtered.append(now)
        self._windows[key] = filtered
        return True

    def get_info(self, key: str) -> dict:
        """Return rate limit info for response headers."""
        now = time.time()
        window = [t for t in self._windows.get(key, []) if now - t < 60]
        remaining = max(0, self.max_rpm - len(window))
        # Reset = seconds until oldest entry expires from the window
        reset = int(60 - (now - window[0])) if window else 60
        return {"limit": self.max_rpm, "remaining": remaining, "reset": max(0, reset)}


# ── Hardened Gateway Validator ───────────────────────────────────────────────

class HardenedGatewayValidator:
    """
    The main validator. Routes organic traffic, generates synthetic probes,
    verifies hidden states, scores miners, and publishes audits.
    """

    def __init__(
        self,
        miner_endpoints: list[str],
        config: GatewayConfig = None,
        r2_local_dir: str = None,
        model=None,
        chain_weight_setter: ChainWeightSetter = None,
        metagraph_discovery: MetagraphDiscovery = None,
    ):
        self.config = config or GatewayConfig()
        self.model = model or MockModel(ModelConfig())
        self.scoring = HardenedScoringEngine(epoch_length_s=self.config.EPOCH_LENGTH_S)
        self.challenge_engine = ChallengeEngine(
            cosine_threshold=self.config.COSINE_THRESHOLD,
            timing_threshold_ms=self.config.CHALLENGE_TIMEOUT_MS,
            timing_hard_cutoff_ms=CHALLENGE_TIMEOUT_HARD_MS,
        )

        # Chain integration (optional)
        self.chain = chain_weight_setter

        # Metagraph discovery (optional — replaces static --miners)
        self.discovery = metagraph_discovery

        # R2 publisher — prefer real R2 upload via env vars, fall back to local
        r2_endpoint = os.environ.get("R2_URL", "").rstrip("/")
        r2_access = os.environ.get("R2_ACCESS_KEY_ID", "")
        r2_secret = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        r2_bucket = os.environ.get("R2_BUCKET", "affine")
        if r2_endpoint and r2_access and r2_secret:
            self.r2 = R2Publisher(
                bucket_name=r2_bucket,
                endpoint_url=r2_endpoint,
                access_key=r2_access,
                secret_key=r2_secret,
            )
        else:
            self.r2 = R2Publisher(local_dir=r2_local_dir or "/tmp/r2-audit")

        # Build miner pool
        miners = {}
        for i, endpoint in enumerate(miner_endpoints):
            miners[i] = MinerInfo(uid=i, endpoint=endpoint, hotkey=f"miner-{i}")
        self.router = IntelligentRouter(miners)

        # Rate limiter
        self.rate_limiter = RateLimiter(max_rpm=self.config.RATE_LIMIT_RPM)

        # KV cache prober and collusion detector
        self.cache_prober = KVCacheProber()
        self.collusion_detector = CollusionDetector()

        # Stats
        self.total_organic = 0
        self.total_synthetic = 0
        self.active_organic_requests = 0  # In-flight organic requests (for synthetic throttling)
        self.total_timeouts = 0
        self.total_miner_errors = 0
        self.total_failovers = 0
        self._quality_scores: list[float] = []  # Rolling quality scores for metrics
        self._challenge_latencies: list[float] = []  # Challenge latency samples
        self.epoch_summaries: list[dict] = []  # Bounded in check_epoch (max 100)

        # Track recent organic prompts to mix with synthetic (rotating buffer)
        self._recent_organic_prompts: list[str] = []
        self._max_recent_prompts = 200

        # Shared aiohttp session (created lazily, reused across requests)
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _get_http_session(self) -> aiohttp.ClientSession:
        """Get or create the shared aiohttp session."""
        if self._http_session is None or self._http_session.closed:
            # Limit concurrent connections to prevent slow-miner connection exhaustion.
            # Per-host limit prevents a single slow miner from consuming the entire pool.
            connector = aiohttp.TCPConnector(
                limit=200,          # Total connection pool
                limit_per_host=20,  # Max connections per miner endpoint
                resolver=aiohttp.AsyncResolver(),  # Avoid ThreadedResolver GIL contention
            )
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.INFERENCE_TIMEOUT_S,
                    connect=5,  # Fast-fail on unreachable miners
                ),
            )
        return self._http_session

    async def close(self):
        """Close the shared HTTP session."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()

    # Additional freeform prompt templates that don't follow persona+style+constraint
    _FREEFORM_TEMPLATES = [
        "What's the difference between {topicA} and {topicB}?",
        "I'm trying to learn about {topic} and got confused about {aspect}. Can you explain?",
        "Can you help me understand {topic}?",
        "How would you explain {topic} to someone who knows nothing about it?",
        "Can you give me a quick example of {topic}?",
        "I have an interview soon and need to understand {topic}. Help?",
        "Why do people prefer {topicA} over {topicB}?",
        "Explain {topic} like I'm five.",
        "Is {topic} still worth learning or has something replaced it?",
        "Give me a quick summary of {topic}.",
        "{topicA} vs {topicB} — which is better and why?",
        "I keep hearing about {topic}. What's the big deal?",
        "What should I watch out for with {topic}?",
        "{topic} — what are the pros and cons?",
        "Help me understand {topic}. I'm a complete beginner.",
        "What do most people get wrong about {topic}?",
        "Can you recommend resources for learning {topic}?",
        "How does {topic} relate to {topicB}?",
        "I need to explain {topic} to my boss. How would you summarize it?",
        "What's your take on {topic}?",
    ]
    _ASPECTS = [
        "edge cases", "the basics", "common mistakes", "real-world applications",
        "getting started", "best practices", "the details", "the theory",
        "practical tips", "the fundamentals",
    ]

    def _generate_synthetic_prompt(self) -> str:
        """
        Generate a synthetic prompt that is indistinguishable from organic traffic.

        Strategy:
        1. 40% — reuse recent organic prompts (miners literally can't distinguish)
        2. 30% — persona + style(topic) + constraint (original template space)
        3. 30% — freeform conversational templates (varied structures)
        Miners cannot fingerprint because prompts use multiple distinct structures
        and 40% are recycled organic prompts.
        """
        prompts_snapshot = self._recent_organic_prompts
        if prompts_snapshot and secrets.randbelow(100) < 40:
            try:
                return secrets.choice(prompts_snapshot)
            except IndexError:
                pass  # Concurrent modification emptied the list — fall through

        # 50/50 split between two generation strategies
        if secrets.randbelow(2) == 0:
            # Strategy A: persona + style + constraint
            topic = secrets.choice(self.config._TOPICS)
            style = secrets.choice(self.config._STYLES)
            constraint = secrets.choice(self.config._CONSTRAINTS)
            persona = secrets.choice(self.config._PERSONAS)
            return persona + style.format(topic=topic) + constraint
        else:
            # Strategy B: freeform conversational
            template = secrets.choice(self._FREEFORM_TEMPLATES)
            topic = secrets.choice(self.config._TOPICS)
            topicA = secrets.choice(self.config._TOPICS)
            topicB = secrets.choice(self.config._TOPICS)
            aspect = secrets.choice(self._ASPECTS)
            return template.format(topic=topic, topicA=topicA, topicB=topicB, aspect=aspect)

    def _generate_dummy_challenge_fields(self, max_tokens: int, prompt: str = None, messages: list = None) -> dict:
        """
        Generate plausible-looking dummy challenge fields for non-challenged requests.
        These are indistinguishable from real challenge params — miners ALWAYS receive
        challenge_layer/challenge_token so they cannot fingerprint by field presence.

        Uses the SAME distribution as real challenges:
        - Same estimated_seq_len formula (prompt_tokens + max_tokens ± 10% noise)
        - Same layer exclusion (skip last 2 layers)
        - Same multi-point probability and count (20% chance, exactly 3 extra points)
        """
        num_layers = self.model.config.num_layers
        # Exclude last 2 layers — same as real challenges in challenge_engine
        safe_layers = max(num_layers - 2, 1)

        # Compute estimated_seq_len using the same formula as real challenges
        if prompt or messages:
            if hasattr(self.model, 'tokenizer'):
                if messages and hasattr(self.model.tokenizer, 'apply_chat_template'):
                    est_prompt = self.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    est_prompt = prompt or ""
                prompt_token_count = len(self.model.tokenizer.encode(est_prompt))
            else:
                prompt_token_count = max_tokens  # fallback
        else:
            prompt_token_count = max_tokens  # fallback

        estimated_seq_len = prompt_token_count + max_tokens
        # Add ±10% noise — same as real challenges
        noise_range = max(1, estimated_seq_len // 10)
        estimated_seq_len += secrets.randbelow(2 * noise_range + 1) - noise_range
        estimated_seq_len = max(2, estimated_seq_len)

        dummy_layer = secrets.randbelow(safe_layers)
        dummy_token = secrets.randbelow(max(estimated_seq_len, 1))
        result = {"challenge_layer": dummy_layer, "challenge_token": dummy_token}
        # 20% chance of multi-point — exactly 3 extra points (matches challenge_engine)
        if secrets.randbelow(5) == 0:
            result["challenge_extra"] = [
                [secrets.randbelow(safe_layers), secrets.randbelow(max(estimated_seq_len, 1))]
                for _ in range(3)
            ]
        return result

    async def _send_inference(
        self,
        session: aiohttp.ClientSession,
        miner: MinerInfo,
        prompt: str,
        request_id: str,
        max_tokens: int = 64,
        messages: list = None,
        challenge_params: dict = None,
        sampling_params: dict = None,
    ) -> Optional[dict]:
        """Send inference request to a miner, always with challenge fields (real or dummy)."""
        url = f"{miner.endpoint}/inference"
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "request_id": request_id,
        }
        if messages:
            payload["messages"] = messages
        # Forward OpenAI sampling params to miner
        if sampling_params:
            for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"):
                if sampling_params.get(key) is not None:
                    payload[key] = sampling_params[key]
        # ALWAYS send challenge fields — prevents miners from fingerprinting by
        # presence/absence of challenge_layer. For non-challenged requests, send
        # dummy values. Miner always computes challenge result; gateway ignores
        # the result when it's a dummy.
        if challenge_params:
            payload["challenge_layer"] = challenge_params["layer_index"]
            payload["challenge_token"] = challenge_params["token_index"]
            if challenge_params.get("extra_points"):
                payload["challenge_extra"] = challenge_params["extra_points"]
        else:
            dummy = self._generate_dummy_challenge_fields(max_tokens, prompt=prompt, messages=messages)
            payload["challenge_layer"] = dummy["challenge_layer"]
            payload["challenge_token"] = dummy["challenge_token"]
            if "challenge_extra" in dummy:
                payload["challenge_extra"] = dummy["challenge_extra"]
        headers = {}
        if self.config.MINER_VALIDATOR_SECRET:
            ts = str(int(time.time()))
            # C5-4: Content-bound HMAC — include body hash to prevent relay forgery
            body_bytes = json.dumps(payload).encode()
            body_hash = hashlib.sha256(body_bytes).hexdigest()[:16]
            msg = f"miner_auth:{request_id}:{ts}:{body_hash}".encode()
            sig = hmac.new(self.config.MINER_VALIDATOR_SECRET.encode(), msg, hashlib.sha256).hexdigest()
            headers["X-Validator-Key"] = f"{ts}:{sig}"
            headers["X-Body-Hash"] = body_hash
        miner.active_requests += 1
        try:
            t_start = time.perf_counter()
            async with session.post(
                url, json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.INFERENCE_TIMEOUT_S,
                    sock_connect=5,  # Fail fast on unreachable miners (TCP connect)
                ),
            ) as resp:
                t_end = time.perf_counter()
                if resp.status == 200:
                    # Guard against oversized miner responses (OOM protection)
                    content_length = resp.content_length
                    if content_length and content_length > self.config.MAX_MINER_RESPONSE_BYTES:
                        log.warning(f"Miner {miner.uid}: response too large ({content_length} bytes)")
                        self.router.report_failure(miner)
                        return None
                    body = await resp.read()
                    if len(body) > self.config.MAX_MINER_RESPONSE_BYTES:
                        log.warning(f"Miner {miner.uid}: response exceeded {self.config.MAX_MINER_RESPONSE_BYTES} bytes")
                        self.router.report_failure(miner)
                        return None
                    # Verify miner response signature (anti-impersonation)
                    resp_headers = {k: v for k, v in resp.headers.items()}
                    sig_valid, sig_reason = verify_miner_signature(
                        resp_headers, request_id, body, miner.hotkey,
                    )
                    if not sig_valid:
                        log.warning(f"Miner {miner.uid}: signature verification FAILED ({sig_reason})")
                        self.router.report_failure(miner)
                        return None
                    data = json.loads(body)
                    data["_wall_time_ms"] = (t_end - t_start) * 1000
                    data["_sig_status"] = sig_reason
                    self.router.report_success(
                        miner,
                        ttft_ms=data.get("ttft_ms", 0),
                        tps=data.get("tokens_per_sec", 0),
                    )
                    return data
                log.warning(f"Miner {miner.uid} returned {resp.status}")
                self.router.report_failure(miner)
                return None
        except asyncio.TimeoutError:
            log.warning(f"Miner {miner.uid} timed out ({self.config.INFERENCE_TIMEOUT_S}s)")
            self.total_timeouts += 1
            self.router.report_failure(miner, timeout=True)
            return None
        except Exception as e:
            log.debug(f"Miner {miner.uid} error: {e}")
            self.router.report_failure(miner)
            return None

    def _validate_token_ids(
        self,
        miner_token_ids: list[int],
        prompt: str,
        response_text: str,
    ) -> tuple[bool, str]:
        """
        Cross-validate miner-reported token IDs against known text.

        Mitigates attacks where miners lie about token IDs to:
        - Shorten sequence length (limiting challenge positions)
        - Pad with fake tokens (diluting challenge probability)
        - Report entirely fabricated tokens

        Returns (valid, reason). If tokenizer is unavailable, applies heuristic checks only.
        """
        if not miner_token_ids or not isinstance(miner_token_ids, list):
            return False, "empty or invalid token_ids"

        # Length sanity: token count should be reasonable for the text
        # Most tokenizers produce ~1-4 tokens per word. A wild mismatch is suspicious.
        total_chars = len(prompt) + len(response_text)
        if total_chars > 0:
            chars_per_token = total_chars / max(len(miner_token_ids), 1)
            # Typical range: 1-8 chars per token. Allow wide bounds to avoid false positives.
            if chars_per_token < 0.5:  # Way too many tokens claimed
                return False, f"token count {len(miner_token_ids)} implausible for {total_chars} chars"
            if chars_per_token > 20:  # Way too few tokens claimed
                return False, f"token count {len(miner_token_ids)} too low for {total_chars} chars"

        # If we have a real tokenizer, do a decode-match check
        if hasattr(self.model, 'tokenizer') and self.model.tokenizer is not None:
            tokenizer = self.model.tokenizer
            try:
                # Independently tokenize the prompt
                prompt_tokens = tokenizer.encode(prompt)
                prompt_len = len(prompt_tokens)

                # Check that the miner's sequence starts with the correct prompt tokens
                if len(miner_token_ids) < prompt_len:
                    return False, "sequence shorter than prompt tokens"

                miner_prompt_portion = miner_token_ids[:prompt_len]
                if miner_prompt_portion != prompt_tokens:
                    # Allow small mismatches (BOS token, whitespace normalization)
                    # but the bulk should match
                    mismatches = sum(1 for a, b in zip(miner_prompt_portion, prompt_tokens) if a != b)
                    mismatch_rate = mismatches / max(prompt_len, 1)
                    if mismatch_rate > 0.1:  # More than 10% mismatch
                        return False, f"prompt token mismatch rate {mismatch_rate:.1%}"

                # Decode the output portion and check it matches response text
                output_portion = miner_token_ids[prompt_len:]
                if output_portion:
                    decoded = tokenizer.decode(output_portion, skip_special_tokens=True)
                    # Normalize whitespace for comparison
                    decoded_normalized = " ".join(decoded.split())
                    response_normalized = " ".join(response_text.split())
                    # Allow some mismatch from tokenization edge cases, but bulk must match
                    if response_normalized and decoded_normalized:
                        # Use a simple overlap check: the decoded text should substantially match
                        shorter = min(len(decoded_normalized), len(response_normalized))
                        if shorter > 10:
                            # Check first 80% of characters match
                            check_len = int(shorter * 0.8)
                            if decoded_normalized[:check_len] != response_normalized[:check_len]:
                                return False, "decoded tokens don't match response text"
            except Exception as e:
                log.debug(f"Token validation decode error: {e}")
                # Don't fail on tokenizer edge cases — fall through to heuristics

        # All integer check — use random sample to avoid miners hiding
        # non-integer tokens after index 100.
        check_indices = list(range(len(miner_token_ids)))
        if len(check_indices) > 200:
            # Deterministic random sample: check first 100 + 100 random positions
            import random as _rng
            _r = _rng.Random(len(miner_token_ids))  # Seed on length so it's consistent
            check_indices = list(range(min(100, len(miner_token_ids)))) + _r.sample(
                range(100, len(miner_token_ids)), min(100, len(miner_token_ids) - 100)
            )
        if not all(isinstance(miner_token_ids[i], int) for i in check_indices):
            return False, "non-integer token IDs"

        # Token ID range check: most LLM vocabularies are <200k tokens.
        # IDs outside [0, 250000] are almost certainly fabricated.
        # Check ALL tokens — sampling only the first N lets miners pad with
        # out-of-range tokens beyond the sample boundary.
        MAX_PLAUSIBLE_TOKEN_ID = 250_000
        out_of_range = sum(1 for t in miner_token_ids if not (0 <= t <= MAX_PLAUSIBLE_TOKEN_ID))
        if out_of_range > len(miner_token_ids) * 0.05:  # >5% out of range = fabricated
            return False, f"token IDs out of plausible range: {out_of_range}/{len(miner_token_ids)} tokens > {MAX_PLAUSIBLE_TOKEN_ID}"

        return True, "ok"

    async def _verify_inline_challenge(
        self,
        challenge,
        inline_result: dict,
        all_tokens: list[int],
        gateway_wall_time_ms: float = 0.0,
    ) -> dict:
        """
        Verify a challenge result returned inline with inference response.
        This avoids the need for a separate /hidden_state endpoint call,
        preventing miners from distinguishing challenge vs organic traffic.

        Uses gateway-measured wall time for latency scoring, NOT the miner's
        self-reported latency_ms (which miners can fabricate).
        """
        # For inline challenges, the challenge is computed as part of the inference
        # request so we can't separate challenge time from inference time. The gateway
        # wall time covers the entire request (inference + challenge + network).
        # Using this for latency scoring would unfairly penalize miners since it
        # includes inference time. Instead, set latency_ms to 0 to avoid latency
        # penalty for inline challenges. The latency penalty is only meaningful for
        # the legacy separate /hidden_state endpoint where we can measure challenge-only time.
        # Miner's self-reported latency_ms is untrusted and NOT used.
        latency_ms = 0.0
        expected_dim = self.model.config.hidden_dim

        try:
            miner_state = np.array(inline_result["hidden_state"], dtype=np.float32)
        except (ValueError, TypeError) as e:
            return {
                "passed": False,
                "reason": f"Invalid hidden_state format: {e}",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        if not np.all(np.isfinite(miner_state)):
            return {
                "passed": False,
                "reason": "Hidden state contains NaN or Inf values",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        if miner_state.ndim != 1 or miner_state.shape[0] != expected_dim:
            return {
                "passed": False,
                "reason": f"Hidden state shape mismatch: got {miner_state.shape}, expected ({expected_dim},)",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        # Collect all challenge points for batch reference computation
        all_points = [(challenge.layer_index, challenge.token_index)]

        extra_miner_states = []
        extra_point_indices = []
        extra_failed_count = 0

        if challenge.extra_points and inline_result.get("extra_states"):
            for extra_data in inline_result["extra_states"]:
                if "error" in extra_data or "hidden_state" not in extra_data:
                    extra_failed_count += 1
                    continue
                try:
                    state = np.array(extra_data["hidden_state"], dtype=np.float32)
                    if state.ndim == 1 and state.shape[0] == expected_dim and np.all(np.isfinite(state)):
                        extra_miner_states.append(state)
                        all_points.append((extra_data["layer_index"], extra_data["token_index"]))
                        extra_point_indices.append(len(all_points) - 1)
                    else:
                        extra_failed_count += 1
                except (ValueError, TypeError):
                    extra_failed_count += 1

            if extra_failed_count > 0 and extra_failed_count == len(challenge.extra_points):
                self.challenge_engine._pending.pop(challenge.challenge_id, None)
                self.challenge_engine.total_failed += 1
                return {
                    "passed": False,
                    "reason": f"Multi-point challenge: miner failed all {len(challenge.extra_points)} extra points",
                    "latency_ms": latency_ms,
                    "cosine_sim": 0.0,
                    "layer": challenge.layer_index,
                    "token_pos": challenge.token_index,
                    "challenge_type": challenge.challenge_type,
                    "extra_results": [],
                }
        elif challenge.extra_points:
            # Miner didn't return extra_states — all extra points failed
            extra_failed_count = len(challenge.extra_points)
            if extra_failed_count == len(challenge.extra_points):
                self.challenge_engine._pending.pop(challenge.challenge_id, None)
                self.challenge_engine.total_failed += 1
                return {
                    "passed": False,
                    "reason": f"Multi-point challenge: miner returned no extra states",
                    "latency_ms": latency_ms,
                    "cosine_sim": 0.0,
                    "layer": challenge.layer_index,
                    "token_pos": challenge.token_index,
                    "challenge_type": challenge.challenge_type,
                    "extra_results": [],
                }

        # Compute reference hidden states
        if hasattr(self.model, 'compute_hidden_states_batch'):
            ref_batch = self.model.compute_hidden_states_batch(all_tokens, all_points)
            reference = ref_batch[all_points[0]]
            extra_reference_states = [ref_batch[all_points[i]] for i in extra_point_indices]
        else:
            reference = self.model.compute_hidden_state_at(
                all_tokens, challenge.layer_index, challenge.token_index
            )
            extra_reference_states = [
                self.model.compute_hidden_state_at(all_tokens, all_points[i][0], all_points[i][1])
                for i in extra_point_indices
            ]

        verification = self.challenge_engine.verify_response(
            challenge_id=challenge.challenge_id,
            miner_hidden_state=miner_state,
            reference_hidden_state=reference,
            latency_ms=latency_ms,
            extra_miner_states=extra_miner_states if extra_miner_states else None,
            extra_reference_states=extra_reference_states if extra_reference_states else None,
        )

        return {
            "passed": verification.passed,
            "reason": verification.reason,
            "latency_ms": verification.latency_ms,
            "cosine_sim": verification.cosine_sim,
            "layer": challenge.layer_index,
            "token_pos": challenge.token_index,
            "challenge_type": challenge.challenge_type,
            "extra_results": verification.extra_results,
        }

    async def _send_challenge(
        self,
        session: aiohttp.ClientSession,
        miner: MinerInfo,
        request_id: str,
        all_tokens: list[int],
    ) -> dict:
        """
        Send hidden state challenge with hardened verification.

        Anti-cheat:
        - Layer and position chosen cryptographically
        - Multi-point challenges randomly triggered
        - Tight timing requirement proves GPU-cached states
        - Challenge ID prevents replay
        """
        num_layers = self.model.config.num_layers
        seq_len = len(all_tokens)

        challenge = self.challenge_engine.create_challenge(
            request_id=request_id,
            num_layers=num_layers,
            seq_len=seq_len,
        )

        # Send primary challenge
        url = f"{miner.endpoint}/hidden_state"
        primary_payload = {
            "request_id": request_id,
            "layer_index": challenge.layer_index,
            "token_index": challenge.token_index,
        }

        try:
            # Hidden state response should be small (< 1MB for any reasonable hidden dim)
            max_challenge_bytes = 1 * 1024 * 1024
            t_start = time.perf_counter()
            async with session.post(
                url, json=primary_payload,
                timeout=aiohttp.ClientTimeout(total=CHALLENGE_TIMEOUT_HARD_MS / 1000),
            ) as resp:
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000

                if resp.status != 200:
                    return {
                        "passed": False,
                        "reason": f"HTTP {resp.status}",
                        "latency_ms": latency_ms,
                        "cosine_sim": 0.0,
                        "layer": challenge.layer_index,
                        "token_pos": challenge.token_index,
                    }

                body = await resp.read()
                if len(body) > max_challenge_bytes:
                    return {
                        "passed": False,
                        "reason": "Challenge response too large",
                        "latency_ms": latency_ms,
                        "cosine_sim": 0.0,
                        "layer": challenge.layer_index,
                        "token_pos": challenge.token_index,
                    }
                data = json.loads(body)
        except asyncio.TimeoutError:
            return {
                "passed": False,
                "reason": "Timeout",
                "latency_ms": CHALLENGE_TIMEOUT_HARD_MS,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }
        except Exception as e:
            return {
                "passed": False,
                "reason": str(e),
                "latency_ms": 0,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        # Validate challenge response format
        if not isinstance(data, dict) or "hidden_state" not in data:
            return {
                "passed": False,
                "reason": "Missing hidden_state in response",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        try:
            miner_state = np.array(data["hidden_state"], dtype=np.float32)
        except (ValueError, TypeError) as e:
            return {
                "passed": False,
                "reason": f"Invalid hidden_state format: {e}",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        # Validate no NaN/Inf values (prevents cosine similarity poisoning)
        if not np.all(np.isfinite(miner_state)):
            return {
                "passed": False,
                "reason": "Hidden state contains NaN or Inf values",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        # Validate hidden state dimensions
        expected_dim = self.model.config.hidden_dim
        if miner_state.ndim != 1 or miner_state.shape[0] != expected_dim:
            return {
                "passed": False,
                "reason": f"Hidden state shape mismatch: got {miner_state.shape}, expected ({expected_dim},)",
                "latency_ms": latency_ms,
                "cosine_sim": 0.0,
                "layer": challenge.layer_index,
                "token_pos": challenge.token_index,
            }

        # Collect all challenge points for batch reference computation
        all_points = [(challenge.layer_index, challenge.token_index)]

        # Multi-point challenges — fetch miner states first
        extra_miner_states = []
        extra_point_indices = []
        extra_failed_count = 0
        if challenge.extra_points:
            for layer_idx, token_idx in challenge.extra_points:
                try:
                    extra_payload = {
                        "request_id": request_id,
                        "layer_index": layer_idx,
                        "token_index": token_idx,
                    }
                    async with session.post(
                        url, json=extra_payload,
                        timeout=aiohttp.ClientTimeout(total=CHALLENGE_TIMEOUT_HARD_MS / 1000),
                    ) as extra_resp:
                        if extra_resp.status == 200:
                            extra_data = await extra_resp.json()
                            if isinstance(extra_data, dict) and "hidden_state" in extra_data:
                                state = np.array(extra_data["hidden_state"], dtype=np.float32)
                                if state.ndim == 1 and state.shape[0] == expected_dim and np.all(np.isfinite(state)):
                                    extra_miner_states.append(state)
                                    all_points.append((layer_idx, token_idx))
                                    extra_point_indices.append(len(all_points) - 1)
                                else:
                                    extra_failed_count += 1
                            else:
                                extra_failed_count += 1
                        else:
                            extra_failed_count += 1
                except Exception:
                    extra_failed_count += 1

            # If miner failed to provide ANY extra states for a multi-point challenge,
            # treat as failure — honest miners should serve all requested points
            if extra_failed_count > 0 and extra_failed_count == len(challenge.extra_points):
                # Pop the pending challenge so it doesn't leak
                self.challenge_engine._pending.pop(challenge.challenge_id, None)
                self.challenge_engine.total_failed += 1
                return {
                    "passed": False,
                    "reason": f"Multi-point challenge: miner failed all {len(challenge.extra_points)} extra points",
                    "latency_ms": latency_ms,
                    "cosine_sim": 0.0,
                    "layer": challenge.layer_index,
                    "token_pos": challenge.token_index,
                    "challenge_type": challenge.challenge_type,
                    "extra_results": [],
                }

        # Compute ALL reference hidden states in one forward pass (much faster)
        if hasattr(self.model, 'compute_hidden_states_batch'):
            ref_batch = self.model.compute_hidden_states_batch(all_tokens, all_points)
            reference = ref_batch[all_points[0]]
            extra_reference_states = [ref_batch[all_points[i]] for i in extra_point_indices]
        else:
            # MockModel fallback — individual calls
            reference = self.model.compute_hidden_state_at(
                all_tokens, challenge.layer_index, challenge.token_index
            )
            extra_reference_states = [
                self.model.compute_hidden_state_at(all_tokens, all_points[i][0], all_points[i][1])
                for i in extra_point_indices
            ]

        # Verify
        verification = self.challenge_engine.verify_response(
            challenge_id=challenge.challenge_id,
            miner_hidden_state=miner_state,
            reference_hidden_state=reference,
            latency_ms=latency_ms,
            extra_miner_states=extra_miner_states if extra_miner_states else None,
            extra_reference_states=extra_reference_states if extra_reference_states else None,
        )

        return {
            "passed": verification.passed,
            "reason": verification.reason,
            "latency_ms": verification.latency_ms,
            "cosine_sim": verification.cosine_sim,
            "layer": challenge.layer_index,
            "token_pos": challenge.token_index,
            "challenge_type": challenge.challenge_type,
            "extra_results": verification.extra_results,
        }

    async def process_request(
        self,
        prompt: str,
        max_tokens: int = 64,
        is_synthetic: bool = False,
        session_id: Optional[str] = None,
        messages: list = None,
        sampling_params: dict = None,
    ) -> Optional[dict]:
        """
        Full request processing pipeline with hardened verification.
        Includes failover: if first miner fails, retry with another.
        """
        # Track in-flight organic requests so synthetic probes can yield capacity
        if not is_synthetic:
            self.active_organic_requests += 1
        try:
            return await self._process_request_inner(
                prompt, max_tokens, is_synthetic, session_id, messages, sampling_params
            )
        finally:
            if not is_synthetic:
                self.active_organic_requests = max(0, self.active_organic_requests - 1)

    async def _process_request_inner(
        self,
        prompt: str,
        max_tokens: int = 64,
        is_synthetic: bool = False,
        session_id: Optional[str] = None,
        messages: list = None,
        sampling_params: dict = None,
    ) -> Optional[dict]:
        """Inner implementation of process_request."""
        # Select miner
        miner = self.router.select_miner(session_id=session_id)
        if not miner:
            return None

        request_id = str(uuid.uuid4())

        # Track session affinity
        if session_id:
            self.router.session_router.set_affinity(session_id, miner.uid)

        # Track organic prompts for synthetic mixing (rotating buffer)
        if not is_synthetic:
            if len(self._recent_organic_prompts) >= self._max_recent_prompts:
                # Replace a random entry to keep the buffer fresh
                idx = secrets.randbelow(len(self._recent_organic_prompts))
                self._recent_organic_prompts[idx] = prompt
            else:
                self._recent_organic_prompts.append(prompt)

        session = await self._get_http_session()

        # Determine challenge BEFORE inference so we can bundle it.
        # Use cryptographic randomness for challenge decision.
        should_challenge = secrets.randbelow(1000) < int(self.config.CHALLENGE_RATE * 1000)
        challenge_params = None

        if should_challenge:
            # Pre-compute challenge parameters. For inline challenges, we need to
            # predict the token sequence length from the prompt. Use our own model
            # to generate and get the token count.
            if hasattr(self.model, 'tokenizer'):
                # When messages are provided, miner applies chat template — use same for estimation
                if messages and hasattr(self.model.tokenizer, 'apply_chat_template'):
                    est_prompt = self.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    est_prompt = prompt
                prompt_tokens = self.model.tokenizer.encode(est_prompt)
                estimated_seq_len = len(prompt_tokens) + max_tokens
            else:
                gen_result = self.model.generate(prompt, max_tokens)
                estimated_seq_len = len(gen_result["all_tokens"])

            # Add ±10% noise to prevent miners from predicting challenge positions
            # by computing the same estimated_seq_len deterministically.
            noise_range = max(1, estimated_seq_len // 10)
            estimated_seq_len += secrets.randbelow(2 * noise_range + 1) - noise_range
            estimated_seq_len = max(2, estimated_seq_len)  # Need at least 2 positions

            num_layers = self.model.config.num_layers
            challenge = self.challenge_engine.create_challenge(
                request_id=request_id,
                num_layers=num_layers,
                seq_len=estimated_seq_len,
            )
            challenge_params = {
                "layer_index": challenge.layer_index,
                "token_index": challenge.token_index,
                "challenge": challenge,
            }
            if challenge.extra_points:
                challenge_params["extra_points"] = [
                    [lyr, tok] for lyr, tok in challenge.extra_points
                ]

        # Step 1: Inference + inline challenge (with failover)
        result = await self._send_inference(
            session, miner, prompt, request_id, max_tokens,
            messages=messages, challenge_params=challenge_params,
            sampling_params=sampling_params,
        )
        if result is None:
            self.total_miner_errors += 1
            # Failover: try another miner with FRESH challenge params.
            # Re-using the same challenge params would let colluding miners relay
            # the exact (layer, token) to the fallback miner ahead of time.
            fallback = self.router.select_miner_excluding({miner.uid}, session_id=session_id)
            if fallback:
                log.info(f"[FAILOVER] Miner {miner.uid} failed, trying miner {fallback.uid}")
                self.total_failovers += 1
                # Clean up the original challenge (was never verified)
                if challenge_params and "challenge" in challenge_params:
                    self.challenge_engine._pending.pop(
                        challenge_params["challenge"].challenge_id, None
                    )
                # Generate fresh challenge for the fallback miner
                failover_challenge_params = None
                if should_challenge:
                    if hasattr(self.model, 'tokenizer'):
                        if messages and hasattr(self.model.tokenizer, 'apply_chat_template'):
                            fo_prompt = self.model.tokenizer.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        else:
                            fo_prompt = prompt
                        prompt_tokens = self.model.tokenizer.encode(fo_prompt)
                        estimated_seq_len = len(prompt_tokens) + max_tokens
                    else:
                        gen_result = self.model.generate(prompt, max_tokens)
                        estimated_seq_len = len(gen_result["all_tokens"])
                    failover_request_id = str(uuid.uuid4())
                    num_layers = self.model.config.num_layers
                    failover_challenge = self.challenge_engine.create_challenge(
                        request_id=failover_request_id,
                        num_layers=num_layers,
                        seq_len=estimated_seq_len,
                    )
                    failover_challenge_params = {
                        "layer_index": failover_challenge.layer_index,
                        "token_index": failover_challenge.token_index,
                        "challenge": failover_challenge,
                    }
                    if failover_challenge.extra_points:
                        failover_challenge_params["extra_points"] = [
                            [lyr, tok] for lyr, tok in failover_challenge.extra_points
                        ]
                    challenge_params = failover_challenge_params
                    request_id = failover_request_id
                miner = fallback
                result = await self._send_inference(
                    session, miner, prompt, request_id, max_tokens,
                    messages=messages, challenge_params=challenge_params,
                    sampling_params=sampling_params,
                )
        if result is None:
            self.total_miner_errors += 1
            return None

        # Validate miner-reported metrics against wall-clock measurement.
        # Miners can fabricate ttft_ms/tokens_per_sec — use wall time as a sanity check.
        wall_time_ms = result.get("_wall_time_ms", 0)
        reported_ttft = result.get("ttft_ms", 0)
        reported_tps = result.get("tokens_per_sec", 0)
        raw_output_tokens = result.get("output_tokens", 0)
        # output_tokens may be a list (token IDs) or an int (count)
        miner_output_count = len(raw_output_tokens) if isinstance(raw_output_tokens, list) else int(raw_output_tokens or 0)

        # Derive output token count from validated all_token_ids if available.
        # Miners can inflate output_tokens to game TPS. Using all_token_ids minus
        # prompt length is more trustworthy since all_token_ids is cross-validated.
        all_token_ids = result.get("all_token_ids")
        if all_token_ids and isinstance(all_token_ids, list):
            if hasattr(self.model, 'tokenizer') and self.model.tokenizer:
                prompt_len = len(self.model.tokenizer.encode(prompt))
            else:
                # Rough estimate: all_tokens - input_tokens
                raw_input = result.get("input_tokens", 0)
                prompt_len = len(raw_input) if isinstance(raw_input, list) else (int(raw_input) if raw_input else 0) or len(prompt.split())
            output_token_count = max(0, len(all_token_ids) - prompt_len)
            # Use the more conservative of validated count and miner's count
            output_token_count = min(output_token_count, miner_output_count) if miner_output_count > 0 else output_token_count
        else:
            # Fallback: estimate from response text length
            response_text_for_count = result.get("text", "")
            text_based_estimate = max(1, len(response_text_for_count.split()))
            output_token_count = min(miner_output_count, text_based_estimate * 3) if miner_output_count > 0 else text_based_estimate

        # TTFT: can't exceed wall time, and must be at least a plausible fraction.
        # Minimum: wall_time / (1 + num_output_tokens), i.e. TTFT >= per-token time.
        # This prevents miners from claiming near-zero TTFT on slow requests.
        if wall_time_ms > 0:
            min_ttft = wall_time_ms / max(1 + output_token_count, 2)
            ttft_ms = max(min_ttft, min(reported_ttft, wall_time_ms))
        else:
            ttft_ms = reported_ttft

        # TPS: compute from wall time if available. Use the more conservative of
        # miner-reported and wall-clock-derived value.
        if wall_time_ms > 0 and output_token_count > 0:
            wall_tps = output_token_count / (wall_time_ms / 1000.0)
            # Allow miner's value only if within 1.5x of wall-clock estimate
            # (some overhead is expected from network round-trip)
            tps = min(reported_tps, wall_tps * 1.5)
        else:
            tps = reported_tps

        response_text = result.get("text", "")

        # Step 2: Verify inline challenge result (if we sent one)
        challenge_result = None
        challenge_passed = None  # None = no challenge, True/False = result
        cos_sim = 0.0
        challenge_latency = 0.0

        if should_challenge and challenge_params:
            challenge = challenge_params["challenge"]
            inline_result = result.get("challenge_result")

            # Get tokens for reference computation
            # When messages are provided, the miner applies apply_chat_template,
            # so we must use the same template-wrapped text for validation.
            if messages and hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'apply_chat_template'):
                effective_prompt = self.model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                effective_prompt = prompt

            miner_token_ids = result.get("all_token_ids")
            if miner_token_ids:
                valid, reason = self._validate_token_ids(miner_token_ids, effective_prompt, response_text)
                if valid:
                    all_tokens = miner_token_ids
                else:
                    log.warning(f"Miner {miner.uid} token_ids rejected: {reason}")
                    if hasattr(self.model, 'tokenizer'):
                        full_text = effective_prompt + response_text
                        all_tokens = self.model.tokenizer.encode(full_text)
                    else:
                        gen_result = self.model.generate(prompt, max_tokens)
                        all_tokens = gen_result["all_tokens"]
            elif hasattr(self.model, 'tokenizer'):
                full_text = effective_prompt + response_text
                all_tokens = self.model.tokenizer.encode(full_text)
            else:
                gen_result = self.model.generate(prompt, max_tokens)
                all_tokens = gen_result["all_tokens"]

            # Validate challenge positions against actual sequence length.
            # Challenge was created using estimated_seq_len (pre-inference), but
            # the miner may have generated fewer tokens. If any challenge position
            # exceeds actual sequence length, the miner could exploit this by
            # returning cached hidden states from unrelated positions.
            actual_seq_len = len(all_tokens)
            challenge_oob = challenge.token_index >= actual_seq_len
            if not challenge_oob and challenge.extra_points:
                challenge_oob = any(tok >= actual_seq_len for _, tok in challenge.extra_points)
            if challenge_oob:
                log.warning(
                    f"Miner {miner.uid}: challenge position out of bounds "
                    f"(token_idx={challenge.token_index}, actual_seq_len={actual_seq_len})"
                )
                # Auto-fail: miner truncated output to evade challenge
                challenge_result = {
                    "passed": False,
                    "reason": f"challenge position {challenge.token_index} exceeds actual sequence length {actual_seq_len}",
                    "latency_ms": 0.0,
                    "cosine_sim": 0.0,
                    "layer": challenge.layer_index,
                    "token_pos": challenge.token_index,
                }
            elif inline_result and "error" not in inline_result and "hidden_state" in inline_result:
                # Verify inline challenge result — use gateway-measured wall time,
                # not miner's self-reported latency_ms
                challenge_result = await self._verify_inline_challenge(
                    challenge, inline_result, all_tokens,
                    gateway_wall_time_ms=wall_time_ms,
                )
            elif inline_result is None:
                # Miner doesn't support inline challenges — auto-fail.
                # The legacy /hidden_state fallback reveals challenge classification
                # (a separate HTTP call tells the miner "this was a challenge").
                # Miners MUST support inline challenges to pass verification.
                log.warning(f"Miner {miner.uid}: no inline challenge support — auto-fail")
                challenge_result = {
                    "passed": False,
                    "reason": "miner does not support inline challenges",
                    "latency_ms": 0.0,
                    "cosine_sim": 0.0,
                    "layer": challenge.layer_index,
                    "token_pos": challenge.token_index,
                }
            else:
                # Miner returned an error (cache_miss, etc.)
                challenge_result = {
                    "passed": False,
                    "reason": inline_result.get("error", "unknown error"),
                    "latency_ms": inline_result.get("latency_ms", 0),
                    "cosine_sim": 0.0,
                    "layer": challenge.layer_index,
                    "token_pos": challenge.token_index,
                }

            challenge_passed = challenge_result["passed"]
            cos_sim = challenge_result["cosine_sim"]
            challenge_latency = challenge_result["latency_ms"]
            log.info(
                f"[CHALLENGE] Miner {miner.uid}: {'PASS' if challenge_passed else 'FAIL'} | "
                f"cosine={cos_sim:.4f} latency={challenge_latency:.1f}ms "
                f"layer={challenge_result.get('layer', '?')} pos={challenge_result.get('token_pos', '?')}"
            )

        # Step 3: Score (Sybil-resistant: use per-miner medians for population ranking)
        medians_ttft, medians_tps = self.scoring.get_miner_medians()
        speed = compute_speed_score(ttft_ms, tps, miner_medians_ttft=medians_ttft, miner_medians_tps=medians_tps)
        verification = compute_verification_score(challenge_passed, cos_sim, challenge_latency)
        quality = compute_output_quality(response_text)

        score = RequestScore(
            request_id=request_id,
            miner_uid=miner.uid,
            timestamp=time.time(),
            is_synthetic=is_synthetic,
            speed_score=speed,
            verification_score=verification,
            quality_score=quality,
            ttft_ms=ttft_ms,
            tokens_per_sec=tps,
            cosine_sim=cos_sim,
            challenge_latency_ms=challenge_latency,
            challenge_passed=challenge_passed,
        )
        self.scoring.record_request(score)

        # Update router blocked UIDs after each scored request
        self.router.update_blocked_uids(self.scoring)

        # Track metrics
        self._quality_scores.append(quality)
        if len(self._quality_scores) > 1000:
            self._quality_scores = self._quality_scores[-500:]
        if challenge_latency > 0:
            self._challenge_latencies.append(challenge_latency)
            if len(self._challenge_latencies) > 1000:
                self._challenge_latencies = self._challenge_latencies[-500:]

        # Step 4: Publish to R2
        audit = AuditRecord(
            request_id=request_id,
            miner_uid=miner.uid,
            miner_hotkey=miner.hotkey,
            is_synthetic=is_synthetic,
            prompt=prompt,
            response_text=response_text,
            ttft_ms=ttft_ms,
            tokens_per_sec=tps,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            challenge_layer=challenge_result["layer"] if challenge_result else None,
            challenge_token_pos=challenge_result["token_pos"] if challenge_result else None,
            cosine_sim=cos_sim if challenge_result else None,
            challenge_latency_ms=challenge_latency if challenge_result else None,
            challenge_passed=challenge_passed if challenge_result else None,
            speed_score=speed,
            verification_score=verification,
            points_awarded=score.points,
        )
        self.r2.publish(audit)

        # Feed collusion detector with timing and error data
        self.collusion_detector.record_timing(
            MinerTimingSample(miner.uid, ttft_ms, tps, time.time())
        )
        # Only record error events when a challenge was actually attempted
        if challenge_passed is not None:
            self.collusion_detector.record_error(
                MinerErrorEvent(miner.uid, challenge_passed, time.time())
            )

        if is_synthetic:
            self.total_synthetic += 1
        else:
            self.total_organic += 1

        status = "PASS" if challenge_passed is True else ("FAIL" if challenge_passed is False else "SKIP")
        req_type = "SYNTH" if is_synthetic else "ORGANIC"
        log.info(
            f"[{req_type}] Miner {miner.uid}: {status} | "
            f"speed={speed:.3f} verify={verification:.0f} pts={score.points:.3f} | "
            f"TTFT={ttft_ms:.1f}ms TPS={tps:.0f}"
        )

        return {
            "request_id": request_id,
            "miner_uid": miner.uid,
            "text": response_text,
            "ttft_ms": ttft_ms,
            "tokens_per_sec": tps,
            "points": score.points,
            "verification": verification,
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        }

    _SYSTEM_PROMPTS = [
        "You are a helpful assistant.",
        "You are a friendly and knowledgeable assistant.",
        "Answer clearly and accurately.",
        "Be detailed but concise in your responses.",
        "You are a helpful tutor who explains things simply.",
        "Be direct and to the point.",
        "You help people learn new things.",
    ]

    async def run_synthetic_probe(self):
        """Generate and process a synthetic request with randomized parameters."""
        prompt = self._generate_synthetic_prompt()
        # Use a range that mimics organic traffic patterns (powers of 2 + common round numbers)
        max_tokens = secrets.choice([50, 64, 100, 128, 150, 200, 256, 300, 500, 512, 1024])

        # 30% chance: send as multi-turn messages (mimics real chat traffic)
        messages = None
        if secrets.randbelow(100) < 30:
            messages = []
            # 50% chance: include a system message
            if secrets.randbelow(2):
                messages.append({"role": "system", "content": secrets.choice(self._SYSTEM_PROMPTS)})
            # 25% chance: include a fake prior turn
            if secrets.randbelow(4) == 0:
                prior = self._generate_synthetic_prompt()
                messages.append({"role": "user", "content": prior})
                messages.append({"role": "assistant", "content": secrets.choice([
                    "Sure, let me explain.", "Here's what I know about that.",
                    "Great question.", "Let me break this down.",
                    "That's an interesting topic.", "I'll do my best to help.",
                    "Of course.", "Here's a quick overview.",
                ])})
            messages.append({"role": "user", "content": prompt})

        await self.process_request(prompt, max_tokens=max_tokens, is_synthetic=True, messages=messages)

    async def run_cross_probe(self):
        """
        Active collusion detection: send the same prompt to two different miners
        and compare their responses at the token level.

        This catches miners sharing a backend or relaying answers, which passive
        timing/error correlation alone cannot reliably detect.
        """
        alive_miners = [m for m in self.router.miners.values() if m.alive]
        if len(alive_miners) < 2:
            return

        # Pick two random miners
        idx_a = secrets.randbelow(len(alive_miners))
        idx_b = secrets.randbelow(len(alive_miners) - 1)
        if idx_b >= idx_a:
            idx_b += 1
        miner_a = alive_miners[idx_a]
        miner_b = alive_miners[idx_b]

        # Generate a prompt (looks organic)
        prompt = self._generate_synthetic_prompt()
        max_tokens = secrets.choice([50, 64, 100, 128, 150, 200, 256])
        request_id_a = str(uuid.uuid4())
        request_id_b = str(uuid.uuid4())

        session = await self._get_http_session()

        # Send to both miners concurrently
        result_a, result_b = await asyncio.gather(
            self._send_inference(session, miner_a, prompt, request_id_a, max_tokens),
            self._send_inference(session, miner_b, prompt, request_id_b, max_tokens),
        )

        if result_a is None or result_b is None:
            return

        # Compare output token IDs (if available) or text
        tokens_a = result_a.get("all_token_ids") or result_a.get("output_tokens", [])
        tokens_b = result_b.get("all_token_ids") or result_b.get("output_tokens", [])

        from collusion_detector import compute_response_similarity, CrossProbeResult

        # Token-level similarity
        if isinstance(tokens_a, list) and isinstance(tokens_b, list) and tokens_a and tokens_b:
            # Use output portion only for comparison to avoid trivially shared prompt tokens
            resp_sim = compute_response_similarity(tokens_a, tokens_b)
        else:
            resp_sim = 0.0

        # Hidden state comparison for the two responses
        hidden_cosine = 0.0  # Would need both miners' hidden states — not available in basic flow

        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        probe = CrossProbeResult(
            prompt_hash=prompt_hash,
            miner_a_uid=miner_a.uid,
            miner_b_uid=miner_b.uid,
            response_similarity=resp_sim,
            hidden_state_cosine=hidden_cosine,
            ttft_a_ms=result_a.get("ttft_ms", 0),
            ttft_b_ms=result_b.get("ttft_ms", 0),
        )
        self.collusion_detector.record_cross_probe(probe)

        # Cross-probes are NOT recorded as regular scored requests.
        # Recording them with challenge_passed=None gives miners free points
        # with zero penalty risk, which is exploitable. Cross-probe results
        # are tracked ONLY for collusion detection (via collusion_detector).
        for result, miner_info in [
            (result_a, miner_a),
            (result_b, miner_b),
        ]:
            # Feed collusion detector with wall-clock-validated timing data.
            # Miners can fabricate ttft_ms/tokens_per_sec — clamp to wall-clock bounds.
            wall_time_ms = result.get("_wall_time_ms", 0)
            reported_ttft = result.get("ttft_ms", 0)
            reported_tps = result.get("tokens_per_sec", 0)
            output_tokens = result.get("output_tokens", 0)
            if isinstance(output_tokens, list):
                output_tokens = len(output_tokens)
            # TTFT can't be less than a fraction of wall time
            validated_ttft = max(reported_ttft, wall_time_ms / max(1 + output_tokens, 1))
            validated_ttft = min(validated_ttft, wall_time_ms)
            # TPS can't exceed wall-clock rate by more than 50%
            wall_tps = output_tokens / max(wall_time_ms / 1000.0, 0.001) if output_tokens > 0 else 0
            validated_tps = min(reported_tps, wall_tps * 1.5) if wall_tps > 0 else reported_tps
            self.collusion_detector.record_timing(
                MinerTimingSample(miner_info.uid, validated_ttft, validated_tps, time.time())
            )

        log.info(
            f"[CROSS_PROBE] Miners {miner_a.uid}↔{miner_b.uid} | "
            f"similarity={resp_sim:.3f}"
        )

    # Follow-up templates for cache probes — generic enough to follow ANY turn-1 topic.
    # These must be indistinguishable from real multi-turn follow-ups.
    _CACHE_PROBE_FOLLOWUPS = [
        "Can you go into more detail on that?",
        "That's interesting. Can you explain the second point more?",
        "What would you recommend as a first step?",
        "Are there any common mistakes to avoid?",
        "Can you give me a concrete example?",
        "How would this work in practice?",
        "What are the trade-offs I should think about?",
        "Could you break that down a bit more?",
        "What's the most important thing to get right?",
        "How does this compare to the alternatives?",
        "Is there anything I'm missing?",
        "What would you do differently if you were starting from scratch?",
        "Can you summarize the key takeaways?",
        "How should I prioritize these steps?",
        "What resources would you recommend for learning more?",
        "Thanks, but I'm still confused about one part.",
        "OK so what about edge cases?",
        "Hmm, that makes sense. What about scalability?",
        "Wait, go back to the first thing you mentioned.",
        "I'm not sure I follow. Can you use an analogy?",
        "And what happens if something goes wrong?",
        "Sure, but how do you handle errors?",
        "Got it. Now walk me through a real scenario.",
        "One more question — how does security factor in?",
        "That helps. What's the debugging process look like?",
        "Interesting. What does the data flow look like end to end?",
        "Let me make sure I understand correctly.",
        "How long would that typically take?",
        "What's the performance impact of that approach?",
        "Would that work for a small team?",
        "OK, and how do you test that?",
        "Does this work on all platforms?",
        "Can you compare those two approaches side by side?",
        "What about backwards compatibility?",
        "Are there any licensing concerns?",
        "I see. Can you elaborate on the architecture?",
        "How does this interact with the rest of the system?",
        "What are the prerequisites?",
        "Could you show me what the output would look like?",
        "And if I wanted to extend this later?",
    ]

    async def run_cache_probe(self):
        """
        Run a single KV cache probe: send turn-1, wait, send turn-2 on same session.

        This verifies miners maintain real KV caches by measuring TTFT speedup.

        ANTI-FINGERPRINTING: Turn-1 uses the same prompt pipeline as synthetic
        challenges (_generate_synthetic_prompt or organic recycling). Turn-2 uses
        generic follow-ups that could follow any topic. This makes probes
        indistinguishable from organic multi-turn conversations.
        """
        # Use the SAME prompt generator as synthetic challenges — miners cannot
        # distinguish cache probes from regular synthetic traffic based on content.
        turn1 = self._generate_synthetic_prompt()
        turn2 = secrets.choice(self._CACHE_PROBE_FOLLOWUPS)
        # Random delay matching real user think time
        delay = PROBE_DELAY_MIN_S + secrets.randbelow(
            int((PROBE_DELAY_MAX_S - PROBE_DELAY_MIN_S) * 1000)
        ) / 1000.0
        session_id = f"sess-{secrets.token_hex(12)}"

        # Turn 1: establish prefix in cache
        # Use varied max_tokens matching synthetic probe distribution to avoid fingerprinting
        probe_max_tokens = secrets.choice([50, 64, 100, 128, 150, 200, 256])
        result1 = await self.process_request(turn1, max_tokens=probe_max_tokens, is_synthetic=True, session_id=session_id)
        if result1 is None:
            return

        # Wait (simulates real user think time)
        await asyncio.sleep(delay)

        # Turn 2: continuation with full conversation context so the miner can
        # reuse the KV cache from turn-1. Without context, turn-2 is a cold start
        # and TTFT ratio provides no signal about cache effectiveness.
        turn2_messages = [
            {"role": "user", "content": turn1},
            {"role": "assistant", "content": result1.get("text", "")},
            {"role": "user", "content": turn2},
        ]
        # Turn 2 also uses varied max_tokens — matching the same distribution
        probe_max_tokens_t2 = secrets.choice([50, 64, 100, 128, 150, 200, 256])
        result2 = await self.process_request(
            turn2, max_tokens=probe_max_tokens_t2, is_synthetic=True,
            session_id=session_id, messages=turn2_messages,
        )
        if result2 is None:
            return

        # Only valid if both went to the same miner (session routing)
        if result1["miner_uid"] != result2["miner_uid"]:
            log.debug(f"[CACHE_PROBE] Session affinity broken: turn1→miner {result1['miner_uid']}, turn2→miner {result2['miner_uid']}")
            return

        ttft1 = result1["ttft_ms"]
        ttft2 = result2["ttft_ms"]

        # Avoid division by zero
        ratio = ttft2 / max(ttft1, 0.1)

        probe_result = CacheProbeResult(
            miner_uid=result1["miner_uid"],
            session_id=session_id,
            turn1_ttft_ms=ttft1,
            turn2_ttft_ms=ttft2,
            ttft_ratio=ratio,
            cache_score=compute_cache_score(ratio),
            challenge_passed=result2["verification"] > 0.0,
            turn1_input_tokens=result1["input_tokens"],
            turn2_input_tokens=result2["input_tokens"],
            probe_delay_s=delay,
        )
        self.cache_prober.record_probe(probe_result)

    async def check_epoch(self):
        """Check if epoch should end and compute/publish weights."""
        if self.scoring.should_end_epoch():
            summary = self.scoring.end_epoch()

            # Collusion/cache analysis is best-effort — failures must not block weight-setting
            try:
                collusion_scores = self.collusion_detector.analyze_all_pairs()
                summary["kv_cache"] = self.cache_prober.summary()
                summary["collusion"] = self.collusion_detector.summary(cached_scores=collusion_scores)

                cache_adj = self.cache_prober.get_cache_weight_adjustments()
                collusion_penalties = self.collusion_detector.get_weight_penalties(cached_scores=collusion_scores)
                if summary["weights"]:
                    adjusted = {}
                    for uid, w in summary["weights"].items():
                        adj = cache_adj.get(uid, 1.0)
                        col = collusion_penalties.get(uid, 1.0)
                        adjusted[uid] = w * adj * col
                    total = sum(adjusted.values())
                    if total > 0:
                        summary["weights"] = {uid: w / total for uid, w in adjusted.items()}
            except Exception as e:
                log.error(f"[EPOCH {summary['epoch']}] Collusion/cache analysis failed (weights unaffected): {e}")

            # Reset prober and detector for next epoch
            try:
                self.cache_prober.reset()
                self.collusion_detector.reset()
            except Exception as e:
                log.error(f"[EPOCH {summary['epoch']}] Prober/detector reset failed: {e}")

            self.epoch_summaries.append(summary)
            if len(self.epoch_summaries) > 100:
                self.epoch_summaries = self.epoch_summaries[-100:]

            try:
                self.r2.publish_epoch_summary(summary)
            except Exception as e:
                log.error(f"[EPOCH {summary['epoch']}] R2 publish failed: {e}")

            log.info(f"\n{'='*60}")
            log.info(f"EPOCH {summary['epoch']} COMPLETE")
            log.info(f"{'='*60}")
            for uid, info in summary["miners"].items():
                suspect = " SUSPECT" if info.get("is_suspect") else ""
                log.info(
                    f"  Miner {uid}: net_pts={info['net_points']:.3f} "
                    f"weight={info['weight']:.4f} "
                    f"pass_rate={info['pass_rate']:.1%} "
                    f"div={info['divergence']:.3f}{suspect}"
                )
            cache_info = summary.get("kv_cache", {})
            log.info(f"  KV Cache: {cache_info.get('total_probes', 0)} probes, {cache_info.get('total_cache_hits', 0)} hits")
            collusion_info = summary.get("collusion", {})
            log.info(f"  Collusion: {collusion_info.get('total_pairs_analyzed', 0)} pairs, {collusion_info.get('flagged_pairs', 0)} flagged")
            log.info(f"{'='*60}\n")

            # Set weights on chain — this is the critical path, must always execute
            if self.chain and summary["weights"]:
                try:
                    success = await self.chain.set_weights(summary["weights"])
                    summary["weights_committed"] = success
                    if success:
                        log.info(f"[EPOCH {summary['epoch']}] Weights committed to chain successfully")
                    else:
                        log.error(f"[EPOCH {summary['epoch']}] Failed to commit weights to chain")
                except Exception as e:
                    log.error(f"[EPOCH {summary['epoch']}] Weight-setting exception: {e}")
                    summary["weights_committed"] = False

            return summary
        return None

    async def synthetic_loop(self):
        """Background loop with randomized timing to prevent detection."""
        while True:
            try:
                # Yield capacity to organic requests — skip probe if users are waiting
                if self.active_organic_requests > 0:
                    await asyncio.sleep(2)
                    continue
                await self.run_synthetic_probe()
                # Randomized interval (±50%) — miners can't predict timing
                base = self.config.SYNTHETIC_INTERVAL_S
                jitter = base * 0.5
                delay = base + (secrets.randbelow(int(jitter * 2000)) - int(jitter * 1000)) / 1000.0
                delay = max(self.config.SYNTHETIC_MIN_INTERVAL_S, min(delay, self.config.SYNTHETIC_MAX_INTERVAL_S))
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Synthetic probe error: {e}")
                await asyncio.sleep(5)

    async def cache_probe_loop(self):
        """Background loop for KV cache probes with randomized timing."""
        while True:
            try:
                # Yield capacity to organic requests
                if self.active_organic_requests > 0:
                    await asyncio.sleep(2)
                    continue
                await self.run_cache_probe()
                # Randomized interval
                base = self.config.CACHE_PROBE_INTERVAL_S
                jitter = base * self.config.CACHE_PROBE_JITTER
                delay = base + (secrets.randbelow(int(jitter * 2000)) - int(jitter * 1000)) / 1000.0
                delay = max(10, min(delay, base * 2))
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Cache probe error: {e}")
                await asyncio.sleep(10)

    async def cross_probe_loop(self):
        """Background loop for active collusion cross-probing."""
        while True:
            try:
                # Yield capacity to organic requests
                if self.active_organic_requests > 0:
                    await asyncio.sleep(2)
                    continue
                await self.run_cross_probe()
                # Cross-probes are resource-intensive (2 inference calls each)
                # Run less frequently than synthetic probes: every 30-90s
                delay = 30 + secrets.randbelow(60)
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Cross probe error: {e}")
                await asyncio.sleep(15)

    async def epoch_loop(self):
        """Background loop for epoch management."""
        while True:
            try:
                await self.check_epoch()
                # Cleanup expired challenges
                self.challenge_engine.cleanup_expired()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Epoch check error: {e}")
                await asyncio.sleep(5)

    async def discovery_loop(self):
        """Background loop for metagraph-based miner discovery."""
        if not self.discovery:
            return
        first_run = True
        while True:
            try:
                miners = await self.discovery.discover_miners()
                if miners:
                    active_uids = set()
                    for m in miners:
                        self.router.add_miner(m["uid"], m["endpoint"], m.get("hotkey", ""))
                        active_uids.add(m["uid"])
                        # Register hotkey with scoring engine for Sybil-resistant tracking
                        hotkey = m.get("hotkey", "")
                        if hotkey:
                            self.scoring.register_hotkey(m["uid"], hotkey)
                    self.router.remove_stale_miners(active_uids)
                    # On first discovery, health-check all miners CONCURRENTLY so dead ones
                    # don't get routed to before the normal health loop catches them.
                    # Sequential checks with 5s timeout × 60 miners = 300s blocking.
                    if first_run:
                        first_run = False
                        session = await self._get_http_session()
                        _disc_sem = asyncio.Semaphore(10)  # Max 10 concurrent checks
                        async def _check_miner(uid, miner):
                            async with _disc_sem:
                                try:
                                    async with session.get(
                                        f"{miner.endpoint}/health",
                                        timeout=aiohttp.ClientTimeout(total=3),
                                    ) as resp:
                                        if resp.status != 200:
                                            raise Exception(f"HTTP {resp.status}")
                                except Exception:
                                    miner.alive = False
                                    miner.reliability_score = 0.0
                                    log.info(f"[DISCOVERY] Miner {uid} ({miner.endpoint}): unreachable on first sync, marked DEAD")
                        await asyncio.gather(*[
                            _check_miner(uid, miner)
                            for uid, miner in list(self.router.miners.items())
                            if miner.alive
                        ])
                await asyncio.sleep(self.discovery.sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Discovery loop error: {e}")
                await asyncio.sleep(30)

    async def health_recovery_loop(self):
        """Background loop that retries dead miners to detect recovery."""
        while True:
            try:
                session = await self._get_http_session()
                await self.router.health_check_dead_miners(session=session)
                await asyncio.sleep(10)  # Fast recovery for agent workloads (was 30s)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Health recovery error: {e}")
                await asyncio.sleep(30)


# ── OpenAI-Compatible FastAPI App ────────────────────────────────────────────

def create_gateway_app(validator: HardenedGatewayValidator) -> FastAPI:
    """Create the gateway FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(
        title="Inference Subnet Gateway",
        version="2.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Force Connection: close on all responses. With h11, this makes the server
    # close the socket after sending the response, preventing CLOSE-WAIT buildup.
    @app.middleware("http")
    async def close_connections(request: Request, call_next):
        response = await call_next(request)
        response.headers["Connection"] = "close"
        return response

    # CORS: restrict by env var, default to open for development
    cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=3600,
    )

    # ── OpenAI-compatible error handler ──────────────────────────────────
    # OpenAI SDKs expect {"error": {"message": ..., "type": ..., "code": ...}}
    _ERROR_TYPE_MAP = {
        401: "authentication_error",
        429: "rate_limit_error",
        501: "invalid_request_error",
        503: "server_error",
    }

    @app.exception_handler(HTTPException)
    async def openai_error_handler(request: Request, exc: HTTPException):
        error_type = _ERROR_TYPE_MAP.get(exc.status_code, "api_error")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": error_type,
                    "param": None,
                    "code": exc.status_code,
                }
            },
            headers=getattr(exc, "headers", None) or {},
        )

    # ── Auth dependency ──────────────────────────────────────────────────
    async def verify_api_key(request: Request):
        """Optional API key verification — uses timing-safe comparison."""
        if not validator.config.API_KEYS:
            return "anonymous"

        auth = request.headers.get("Authorization", "")
        # Case-insensitive "Bearer " prefix per RFC 6750
        if auth[:7].lower() == "bearer ":
            key = auth[7:].strip()
            if _timing_safe_key_in(key, validator.config.API_KEYS):
                return key

        raise HTTPException(status_code=401, detail="Invalid API key")

    async def rate_limit(request: Request, api_key: str = Depends(verify_api_key)):
        """Rate limiting per API key with RFC-compliant headers."""
        if not validator.rate_limiter.check(api_key):
            info = validator.rate_limiter.get_info(api_key)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "Retry-After": str(info["reset"]),
                    "RateLimit-Limit": str(info["limit"]),
                    "RateLimit-Remaining": "0",
                    "RateLimit-Reset": str(info["reset"]),
                },
            )
        return api_key

    # ── Chat Completions (OpenAI-compatible) ─────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        api_key: str = Depends(rate_limit),
    ):
        """OpenAI-compatible chat completions endpoint."""
        # Validate roles (defense in depth — reject unknown roles before forwarding to miners)
        for m in request.messages:
            if m.role not in VALID_ROLES:
                raise HTTPException(status_code=400, detail=f"Invalid role: {m.role!r}")
        # Build messages list for chat template (normalize multimodal content to plain text)
        messages = [{"role": m.role, "content": m.text} for m in request.messages]
        # Fallback prompt for mock model / synthetic probes
        prompt = "\n".join(f"{m.role}: {m.text}" for m in request.messages)

        # Context length guard — prevent miners from getting OOB requests
        max_ctx = validator.config.MAX_CONTEXT_TOKENS
        if hasattr(validator.model, 'tokenizer') and validator.model.tokenizer:
            try:
                if hasattr(validator.model.tokenizer, 'apply_chat_template'):
                    est_text = validator.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    est_text = prompt
                est_prompt_tokens = len(validator.model.tokenizer.encode(est_text))
            except Exception:
                est_prompt_tokens = len(prompt) // 3  # fallback
        else:
            # Conservative heuristic: ~3.5 chars per token for Qwen/Llama
            est_prompt_tokens = len(prompt) // 3
        if est_prompt_tokens >= max_ctx:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long: ~{est_prompt_tokens} tokens exceeds model context of {max_ctx}. Please reduce message length.",
            )
        max_tokens = request.max_tokens
        if est_prompt_tokens + max_tokens > max_ctx:
            max_tokens = max(1, max_ctx - est_prompt_tokens)

        # Use session_id for KV cache routing
        session_id = request.session_id

        # Collect sampling params to forward to miners
        sampling_params = {}
        if request.temperature is not None:
            sampling_params["temperature"] = request.temperature
        if request.top_p is not None:
            sampling_params["top_p"] = request.top_p
        if request.frequency_penalty is not None:
            sampling_params["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty is not None:
            sampling_params["presence_penalty"] = request.presence_penalty
        if request.stop is not None:
            sampling_params["stop"] = request.stop

        if request.stream:
            return StreamingResponse(
                _stream_response(validator, prompt, max_tokens, request.model, session_id, messages=messages, sampling_params=sampling_params),
                media_type="text/event-stream",
            )

        result = await validator.process_request(
            prompt,
            max_tokens=max_tokens,
            is_synthetic=False,
            session_id=session_id,
            messages=messages,
            sampling_params=sampling_params,
        )

        if result is None:
            raise HTTPException(status_code=503, detail="No miners available")

        # Determine finish_reason: "length" if generation hit max_tokens, else "stop"
        out_tokens = result["output_tokens"]
        if isinstance(out_tokens, list):
            out_tokens = len(out_tokens)
        finish = "length" if out_tokens >= max_tokens else "stop"

        return ChatCompletionResponse(
            id=f"chatcmpl-{result['request_id']}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=result["text"]),
                    finish_reason=finish,
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result["input_tokens"],
                completion_tokens=result["output_tokens"] if isinstance(result["output_tokens"], int) else len(result["output_tokens"]),
                total_tokens=result["input_tokens"] + (result["output_tokens"] if isinstance(result["output_tokens"], int) else len(result["output_tokens"])),
            ),
        )

    # ── Anthropic Messages API endpoint ──────────────────────────────────

    class AnthropicContentBlock(BaseModel):
        type: str = "text"
        text: str = ""

    class AnthropicMessage(BaseModel):
        role: str = Field(..., max_length=32)
        content: Union[str, list] = Field(..., max_length=500_000)

    class AnthropicMessagesRequest(BaseModel):
        model: str = Field("default", max_length=128)
        max_tokens: int = Field(4096, ge=1, le=32768)
        messages: list[AnthropicMessage] = Field(..., max_length=256)
        system: Optional[Union[str, list]] = None
        stream: bool = False
        temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
        top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
        stop_sequences: Optional[list[str]] = None
        metadata: Optional[dict] = None

        model_config = {"extra": "ignore"}

    def _extract_text(content) -> str:
        """Extract plain text from Anthropic content (str or list of blocks)."""
        if isinstance(content, str):
            return content
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)

    def _anthropic_to_openai_messages(request: AnthropicMessagesRequest) -> list[ChatMessage]:
        """Convert Anthropic Messages format to OpenAI ChatMessage list."""
        msgs = []
        if request.system:
            sys_text = _extract_text(request.system)
            msgs.append(ChatMessage(role="system", content=sys_text))
        for m in request.messages:
            text = _extract_text(m.content)
            msgs.append(ChatMessage(role=m.role, content=text))
        return msgs

    @app.post("/v1/messages")
    async def anthropic_messages(
        request: AnthropicMessagesRequest,
        raw_request: Request,
    ):
        """Anthropic Messages API endpoint — translates to internal OpenAI format."""
        # Auth: accept both x-api-key (Anthropic) and Authorization: Bearer (OpenAI)
        api_key = raw_request.headers.get("x-api-key") or ""
        auth_header = raw_request.headers.get("authorization", "")
        if auth_header.startswith("Bearer ") and not api_key:
            api_key = auth_header[7:]
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")
        # Rate limit using same mechanism
        await rate_limit(raw_request)

        # Convert to OpenAI message format
        oai_messages = _anthropic_to_openai_messages(request)

        # Validate roles
        for m in oai_messages:
            if m.role not in VALID_ROLES:
                raise HTTPException(status_code=400, detail=f"Invalid role: {m.role!r}")

        messages = [{"role": m.role, "content": m.text} for m in oai_messages]
        prompt = "\n".join(f"{m.role}: {m.text}" for m in oai_messages)

        # Context length guard
        max_ctx = validator.config.MAX_CONTEXT_TOKENS
        if hasattr(validator.model, 'tokenizer') and validator.model.tokenizer:
            try:
                if hasattr(validator.model.tokenizer, 'apply_chat_template'):
                    est_text = validator.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    est_text = prompt
                est_prompt_tokens = len(validator.model.tokenizer.encode(est_text))
            except Exception:
                est_prompt_tokens = len(prompt) // 3
        else:
            est_prompt_tokens = len(prompt) // 3
        if est_prompt_tokens >= max_ctx:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long: ~{est_prompt_tokens} tokens exceeds model context of {max_ctx}.",
            )
        max_tokens = request.max_tokens
        if est_prompt_tokens + max_tokens > max_ctx:
            max_tokens = max(1, max_ctx - est_prompt_tokens)

        # Sampling params
        sampling_params = {}
        if request.temperature is not None:
            sampling_params["temperature"] = request.temperature
        if request.top_p is not None:
            sampling_params["top_p"] = request.top_p
        if request.stop_sequences:
            sampling_params["stop"] = request.stop_sequences

        # Session ID for KV cache affinity (from metadata or header)
        session_id = None
        if request.metadata and isinstance(request.metadata, dict):
            session_id = request.metadata.get("session_id")
        if not session_id:
            session_id = raw_request.headers.get("x-session-id")

        if request.stream:
            return StreamingResponse(
                _stream_anthropic_response(
                    validator, prompt, max_tokens, request.model,
                    messages=messages, sampling_params=sampling_params,
                    session_id=session_id,
                ),
                media_type="text/event-stream",
            )

        # Non-streaming
        result = await validator.process_request(
            prompt,
            max_tokens=max_tokens,
            is_synthetic=False,
            session_id=session_id,
            messages=messages,
            sampling_params=sampling_params,
        )

        if result is None:
            raise HTTPException(status_code=503, detail="No miners available")

        out_tokens = result["output_tokens"]
        if isinstance(out_tokens, list):
            out_tokens = len(out_tokens)
        stop_reason = "max_tokens" if out_tokens >= max_tokens else "end_turn"

        return JSONResponse(content={
            "id": f"msg_{result['request_id'][:24]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": result["text"]}],
            "model": request.model,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": result["input_tokens"],
                "output_tokens": out_tokens,
            },
        })

    # ── Legacy inference endpoint ────────────────────────────────────────

    class LegacyRequest(BaseModel):
        prompt: str = Field(..., max_length=100_000)
        max_tokens: int = Field(64, ge=1, le=32768)
        stream: bool = False

    class LegacyResponse(BaseModel):
        request_id: str
        text: str
        model: str
        input_tokens: int
        output_tokens: int
        ttft_ms: float
        total_ms: float
        tokens_per_sec: float

    @app.post("/v1/inference", response_model=LegacyResponse)
    async def inference(
        request: LegacyRequest,
        api_key: str = Depends(rate_limit),
    ):
        """Legacy inference endpoint."""
        result = await validator.process_request(
            request.prompt, request.max_tokens, is_synthetic=False
        )
        if result is None:
            raise HTTPException(status_code=503, detail="No miners available")

        return LegacyResponse(
            request_id=result["request_id"],
            text=result["text"],
            model=validator.model.config.name,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            ttft_ms=result["ttft_ms"],
            total_ms=result.get("_wall_time_ms", result["ttft_ms"]),
            tokens_per_sec=result["tokens_per_sec"],
        )

    # ── Monitoring auth dependency ────────────────────────────────────────
    async def verify_monitoring_key(request: Request):
        """
        Auth for monitoring endpoints. When no monitoring keys configured,
        scoreboard/epochs require auth (returns 401) to prevent miners from
        reading scoring data. Health endpoint has its own open/restricted logic.
        """
        if not validator.config.MONITORING_KEYS:
            # No monitoring keys = scoreboard/epochs are LOCKED by default.
            # Miners must not read scoring data.
            raise HTTPException(
                status_code=401,
                detail="Monitoring access requires authentication. Set --monitoring-keys."
            )
        auth = request.headers.get("Authorization", "")
        if auth[:7].lower() == "bearer ":
            key = auth[7:].strip()
            if _timing_safe_key_in(key, validator.config.MONITORING_KEYS):
                return key
        raise HTTPException(status_code=401, detail="Monitoring access requires authentication")

    # ── Monitoring endpoints ─────────────────────────────────────────────

    @app.get("/v1/scoreboard")
    async def scoreboard(_key: str = Depends(verify_monitoring_key)):
        """Scoreboard — requires monitoring auth when configured."""
        return {
            "epoch": validator.scoring.epoch_number,
            "epoch_elapsed_s": time.time() - validator.scoring.current_epoch_start,
            "total_organic": validator.total_organic,
            "total_synthetic": validator.total_synthetic,
            "active_organic_requests": validator.active_organic_requests,
            "r2_records": validator.r2.records_published,
            "miners": validator.scoring.get_scoreboard(),
        }

    @app.get("/v1/router/stats")
    async def router_stats(_key: str = Depends(verify_monitoring_key)):
        """Router stats — shows per-miner routing weights, TPS, and stall detection."""
        stats = validator.router.get_router_stats()
        stalled = [s for s in stats if s["stalled"]]
        return {
            "total_alive": len(stats),
            "total_stalled": len(stalled),
            "stalled_uids": [s["uid"] for s in stalled],
            "miners": stats,
        }

    @app.get("/v1/health")
    async def health(request: Request):
        """Health endpoint — always open. Detailed info requires monitoring auth."""
        alive = sum(1 for m in validator.router.miners.values() if m.alive)
        uptime_s = time.time() - GATEWAY_START_TIME
        epoch_elapsed = time.time() - validator.scoring.current_epoch_start
        epoch_length = validator.config.EPOCH_LENGTH_S

        # Get last epoch weights if available
        last_weights = {}
        last_epoch_summary = None
        if validator.epoch_summaries:
            last_epoch_summary = validator.epoch_summaries[-1]
            last_weights = last_epoch_summary.get("weights", {})

        # Check monitoring auth early — scoring data requires it
        is_authed = False
        if validator.config.MONITORING_KEYS:
            auth = request.headers.get("Authorization", "")
            if auth[:7].lower() == "bearer ":
                key = auth[7:].strip()
                if _timing_safe_key_in(key, validator.config.MONITORING_KEYS):
                    is_authed = True

        result = {
            "status": "ok",
            "version": GATEWAY_VERSION,
            "uptime_s": int(uptime_s),
            "model": validator.model.config.name,
            "miners_total": len(validator.router.miners),
            "miners_alive": alive,
            "epoch": validator.scoring.epoch_number,
            "epoch_elapsed_s": int(epoch_elapsed),
            "epoch_length_s": epoch_length,
            "total_organic": validator.total_organic,
            "total_synthetic": validator.total_synthetic,
            "last_weight_set": validator.chain.last_set_time if validator.chain else 0,
            "weights": {str(uid): round(w, 4) for uid, w in last_weights.items()},
            "challenges": {
                "total": validator.challenge_engine.total_challenges,
                "passed": validator.challenge_engine.total_passed,
                "failed": validator.challenge_engine.total_failed,
            },
            "errors": {
                "timeouts": validator.total_timeouts,
                "miner_errors": validator.total_miner_errors,
                "failovers": validator.total_failovers,
            },
            "miners_detail": [],
        }

        # Build per-miner detail — public data (TPS, TTFT, weights, pass rates)
        # Operator policy: full transparency on performance metrics
        # Use cached scoreboard to avoid blocking the event loop with compute_weights()
        try:
            scoreboard = {s["uid"]: s for s in validator.scoring.get_scoreboard()}
        except Exception:
            scoreboard = {}
        for m in validator.router.miners.values():
            detail = {
                "uid": m.uid,
                "alive": m.alive,
                "reliability": round(m.reliability_score, 3),
                "served": m.requests_served,
                "failed": m.requests_failed,
                "avg_ttft_ms": round(m.avg_ttft_ms, 1),
                "avg_tps": round(m.avg_tps, 1),
                "active": m.active_requests,
            }
            # Endpoint URLs only with auth
            if is_authed:
                detail["endpoint"] = m.endpoint
            if m.uid in scoreboard:
                sb = scoreboard[m.uid]
                detail["score"] = round(sb["net_points"], 3)
                detail["weight"] = round(last_weights.get(m.uid, 0), 4)
                detail["pass_rate"] = round(sb["pass_rate"], 3)
                detail["divergence"] = round(sb["divergence"], 3)
                detail["is_suspect"] = sb["is_suspect"]
            result["miners_detail"].append(detail)

        if validator.chain:
            result["chain"] = {
                "enabled": True,
                "netuid": validator.chain.netuid,
                "network": validator.chain.network,
                "total_weight_sets": validator.chain.total_sets,
                "total_failures": validator.chain.total_failures,
            }
        if validator.discovery:
            result["discovery"] = {
                "enabled": True,
                "netuid": validator.discovery.netuid,
                "network": validator.discovery.network,
                "last_sync": validator.discovery.last_sync,
            }

        return result

    # Alias: /health → same handler (standard health check path)
    app.add_api_route("/health", health, methods=["GET"])

    @app.get("/ping")
    async def ping():
        """Public liveness check — no sensitive data."""
        return {
            "status": "ok",
            "uptime_s": int(time.time() - GATEWAY_START_TIME),
            "version": GATEWAY_VERSION,
        }

    # ── Network info endpoint (on-chain data for status dashboard) ──────
    _network_cache: dict = {"data": None, "ts": 0}
    _network_lock = asyncio.Lock()

    @app.get("/v1/network")
    async def network_info():
        """Public on-chain subnet info for the status dashboard. Cached 60s.

        Uses subprocess for bittensor calls (Subtensor/Metagraph spawn websocket
        threads that never die — using asyncio.to_thread would leak 4 threads per call).
        Lock prevents concurrent subprocess spawns while cache is being populated.
        """
        now = time.time()
        if _network_cache["data"] and now - _network_cache["ts"] < 60:
            return _network_cache["data"]

        async with _network_lock:
            # Re-check cache after acquiring lock (another request may have populated it)
            now = time.time()
            if _network_cache["data"] and now - _network_cache["ts"] < 60:
                return _network_cache["data"]
            try:
                result = await _fetch_network_info_subprocess()
                _network_cache["data"] = result
                _network_cache["ts"] = time.time()
                return result
            except Exception as e:
                log.error(f"[NETWORK] Failed to fetch on-chain info: {e}")
                if _network_cache["data"]:
                    return _network_cache["data"]
                return {"error": str(e)}

    async def _fetch_network_info_subprocess():
        """Fetch network info via subprocess to avoid bittensor websocket thread leak."""
        netuid = validator.discovery.netuid if validator.discovery else 97
        network = validator.discovery.network if validator.discovery else "finney"
        script = f"""
import json, sys
try:
    import bittensor as bt
    import numpy as np
    sub = bt.Subtensor(network="{network}")
    meta = bt.Metagraph(netuid={netuid}, network=sub.network, sync=True)

    try:
        cost = sub.recycle(netuid={netuid})
        reg_cost_tao = float(cost) if cost else None
    except Exception:
        reg_cost_tao = None

    try:
        hyper = sub.get_subnet_hyperparameters(netuid={netuid})
        hyperparams = {{
            "tempo": getattr(hyper, "tempo", None),
            "max_validators": getattr(hyper, "max_validators", None),
            "immunity_period": getattr(hyper, "immunity_period", None),
            "weights_rate_limit": getattr(hyper, "weights_rate_limit", None),
            "adjustment_interval": getattr(hyper, "adjustment_interval", None),
            "commit_reveal_weights_enabled": getattr(hyper, "commit_reveal_weights_enabled", None),
            "registration_allowed": getattr(hyper, "registration_allowed", None),
            "max_regs_per_block": getattr(hyper, "max_regs_per_block", None),
            "serving_rate_limit": getattr(hyper, "serving_rate_limit", None),
            "activity_cutoff": getattr(hyper, "activity_cutoff", None),
            "min_allowed_weights": getattr(hyper, "min_allowed_weights", None),
            "max_weight_limit": getattr(hyper, "max_weight_limit", None),
            "liquid_alpha_enabled": getattr(hyper, "liquid_alpha_enabled", None),
        }}
    except Exception:
        hyperparams = {{}}

    n_total = meta.n.item() if hasattr(meta.n, 'item') else int(meta.n)
    dividends = np.array([float(d) for d in meta.dividends])
    stakes = np.array([float(s) for s in meta.stake])
    n_validators = int((dividends > 0).sum())
    n_miners = n_total - n_validators
    total_stake = float(stakes.sum())

    result = {{
        "netuid": {netuid},
        "network": "{network}",
        "n_neurons": n_total,
        "n_validators": n_validators,
        "n_miners": n_miners,
        "total_stake": round(total_stake, 2),
        "registration_cost_tao": reg_cost_tao,
        "hyperparameters": hyperparams,
    }}
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
"""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode == 0 and stdout:
            return json.loads(stdout.decode().strip())
        raise RuntimeError(f"subprocess failed: {stderr.decode().strip()}")

    # ── Internal relay endpoint (auditor → gateway → miner) ──────────────
    @app.post("/internal/relay")
    async def relay_to_miner(request: Request):
        """Forward auditor probes through gateway IP to mask source address."""
        if not validator.config.INTERNAL_RELAY_SECRET:
            return JSONResponse(status_code=404, content={"error": "not found"})
        auth = request.headers.get("Authorization", "")
        if not (auth[:7].lower() == "bearer " and
                hmac.compare_digest(auth[7:].strip().encode(),
                                    validator.config.INTERNAL_RELAY_SECRET.encode())):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})

        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "relay_failed"})
        target_uid = body.get("miner_uid")
        payload = body.get("payload")
        if target_uid is None or payload is None:
            return JSONResponse(status_code=400, content={"error": "relay_failed"})
        if not isinstance(payload, dict):
            return JSONResponse(status_code=400, content={"error": "relay_failed"})
        try:
            target_uid_int = int(target_uid)
        except (ValueError, TypeError):
            return JSONResponse(status_code=400, content={"error": "relay_failed"})

        miner = validator.router.miners.get(target_uid_int)
        if miner is None or not miner.alive:
            return JSONResponse(status_code=404, content={"error": "relay_failed"})

        session = await validator._get_http_session()
        url = f"{miner.endpoint}/inference"
        try:
            async with session.post(
                url, json=payload,
                timeout=aiohttp.ClientTimeout(total=validator.config.INFERENCE_TIMEOUT_S),
            ) as resp:
                if resp.status != 200:
                    return JSONResponse(status_code=resp.status, content={"error": "relay_failed"})
                body = await resp.read()
                # Pass through miner signature headers for auditor verification
                relay_headers = {}
                for h in ("X-Miner-Hotkey", "X-Miner-Signature"):
                    if h in resp.headers:
                        relay_headers[h] = resp.headers[h]
                data = json.loads(body)
                return JSONResponse(content=data, headers=relay_headers)
        except asyncio.TimeoutError:
            log.warning(f"[RELAY] Timeout for UID {target_uid_int}")
            return JSONResponse(status_code=504, content={"error": "relay_failed"})
        except Exception as e:
            log.error(f"[RELAY] Error for UID {target_uid_int}: {e}", exc_info=True)
            return JSONResponse(status_code=502, content={"error": "relay_failed"})

    @app.get("/v1/epochs")
    async def epochs(_key: str = Depends(verify_monitoring_key)):
        """Epoch history with aggregated stats for the dashboard."""
        result = []
        for s in validator.epoch_summaries:
            miners_data = s.get("miners", {})
            total_organic = sum(m.get("organic_count", 0) for m in miners_data.values())
            total_synthetic = sum(m.get("synthetic_count", 0) for m in miners_data.values())
            total_passed = sum(m.get("passed_challenges", 0) for m in miners_data.values())
            total_failed = sum(m.get("failed_challenges", 0) for m in miners_data.values())
            result.append({
                "epoch": s.get("epoch"),
                "duration_s": round(s.get("duration_s", 0), 1),
                "organic": total_organic,
                "synthetic": total_synthetic,
                "challenges_passed": total_passed,
                "challenges_failed": total_failed,
                "miners_alive": len(miners_data),
                "miners_total": len(miners_data),
                "total_requests": s.get("total_requests", 0),
            })
        return result

    @app.get("/metrics")
    async def metrics(_key: str = Depends(verify_monitoring_key)):
        """Prometheus-compatible metrics endpoint."""
        from fastapi.responses import PlainTextResponse
        lines = []

        # Gateway-level metrics
        lines.append("# HELP gateway_requests_total Total requests processed")
        lines.append("# TYPE gateway_requests_total counter")
        lines.append(f'gateway_requests_total{{type="organic"}} {validator.total_organic}')
        lines.append(f'gateway_requests_total{{type="synthetic"}} {validator.total_synthetic}')

        lines.append("# HELP gateway_challenges_total Challenge verification results")
        lines.append("# TYPE gateway_challenges_total counter")
        lines.append(f'gateway_challenges_total{{result="passed"}} {validator.challenge_engine.total_passed}')
        lines.append(f'gateway_challenges_total{{result="failed"}} {validator.challenge_engine.total_failed}')

        lines.append("# HELP gateway_epoch Current epoch number")
        lines.append("# TYPE gateway_epoch gauge")
        lines.append(f"gateway_epoch {validator.scoring.epoch_number}")

        lines.append("# HELP gateway_miners_alive Number of alive miners")
        lines.append("# TYPE gateway_miners_alive gauge")
        alive = sum(1 for m in validator.router.miners.values() if m.alive)
        lines.append(f"gateway_miners_alive {alive}")

        lines.append("# HELP gateway_miners_total Total registered miners")
        lines.append("# TYPE gateway_miners_total gauge")
        lines.append(f"gateway_miners_total {len(validator.router.miners)}")

        # Per-miner metrics
        def _sanitize_label(s: str) -> str:
            """Sanitize a string for use in Prometheus labels."""
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")

        lines.append("# HELP miner_requests_served Requests served per miner")
        lines.append("# TYPE miner_requests_served counter")
        for m in validator.router.miners.values():
            lines.append(f'miner_requests_served{{uid="{m.uid}"}} {m.requests_served}')

        lines.append("# HELP miner_requests_failed Requests failed per miner")
        lines.append("# TYPE miner_requests_failed counter")
        for m in validator.router.miners.values():
            lines.append(f'miner_requests_failed{{uid="{m.uid}"}} {m.requests_failed}')

        lines.append("# HELP miner_reliability Reliability score per miner")
        lines.append("# TYPE miner_reliability gauge")
        for m in validator.router.miners.values():
            lines.append(f'miner_reliability{{uid="{m.uid}"}} {m.reliability_score:.4f}')

        lines.append("# HELP miner_avg_ttft_ms Average TTFT per miner")
        lines.append("# TYPE miner_avg_ttft_ms gauge")
        for m in validator.router.miners.values():
            lines.append(f'miner_avg_ttft_ms{{uid="{m.uid}"}} {m.avg_ttft_ms:.2f}')

        lines.append("# HELP miner_avg_tps Average tokens per second per miner")
        lines.append("# TYPE miner_avg_tps gauge")
        for m in validator.router.miners.values():
            lines.append(f'miner_avg_tps{{uid="{m.uid}"}} {m.avg_tps:.1f}')

        lines.append("# HELP miner_active_requests In-flight requests per miner")
        lines.append("# TYPE miner_active_requests gauge")
        for m in validator.router.miners.values():
            lines.append(f'miner_active_requests{{uid="{m.uid}"}} {m.active_requests}')

        # Chain metrics (if enabled)
        if validator.chain:
            lines.append("# HELP chain_weight_sets_total Successful weight sets")
            lines.append("# TYPE chain_weight_sets_total counter")
            lines.append(f"chain_weight_sets_total {validator.chain.total_sets}")
            lines.append("# HELP chain_weight_failures_total Failed weight sets")
            lines.append("# TYPE chain_weight_failures_total counter")
            lines.append(f"chain_weight_failures_total {validator.chain.total_failures}")

        # R2 audit metrics
        lines.append("# HELP r2_records_published Total audit records published")
        lines.append("# TYPE r2_records_published counter")
        lines.append(f"r2_records_published {validator.r2.records_published}")

        # Error / reliability metrics
        lines.append("# HELP gateway_timeouts_total Total miner timeouts")
        lines.append("# TYPE gateway_timeouts_total counter")
        lines.append(f"gateway_timeouts_total {validator.total_timeouts}")

        lines.append("# HELP gateway_miner_errors_total Total miner errors (non-timeout)")
        lines.append("# TYPE gateway_miner_errors_total counter")
        lines.append(f"gateway_miner_errors_total {validator.total_miner_errors}")

        lines.append("# HELP gateway_failovers_total Total failover events")
        lines.append("# TYPE gateway_failovers_total counter")
        lines.append(f"gateway_failovers_total {validator.total_failovers}")

        # Quality metrics
        if validator._quality_scores:
            avg_q = sum(validator._quality_scores) / len(validator._quality_scores)
            lines.append("# HELP gateway_avg_quality_score Average output quality score")
            lines.append("# TYPE gateway_avg_quality_score gauge")
            lines.append(f"gateway_avg_quality_score {avg_q:.4f}")

        # Challenge latency summary
        if validator._challenge_latencies:
            lats = sorted(validator._challenge_latencies)
            p50 = lats[len(lats) // 2]
            p95 = lats[int(len(lats) * 0.95)]
            p99 = lats[int(len(lats) * 0.99)]
            lines.append("# HELP gateway_challenge_latency_ms Challenge latency percentiles")
            lines.append("# TYPE gateway_challenge_latency_ms gauge")
            lines.append(f'gateway_challenge_latency_ms{{quantile="0.5"}} {p50:.2f}')
            lines.append(f'gateway_challenge_latency_ms{{quantile="0.95"}} {p95:.2f}')
            lines.append(f'gateway_challenge_latency_ms{{quantile="0.99"}} {p99:.2f}')

        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    # ── Text Completions (legacy OpenAI-compatible) ────────────────────

    class CompletionRequest(BaseModel):
        model: str = "default"
        prompt: str = Field(..., max_length=100_000)
        max_tokens: int = Field(64, ge=1, le=32768)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        stream: bool = False

    @app.post("/v1/completions")
    async def completions(
        request: CompletionRequest,
        api_key: str = Depends(rate_limit),
    ):
        """OpenAI-compatible legacy text completions endpoint."""
        # Context length guard
        max_ctx = validator.config.MAX_CONTEXT_TOKENS
        est_prompt_tokens = len(request.prompt) // 3
        if est_prompt_tokens >= max_ctx:
            raise HTTPException(
                status_code=400,
                detail=f"Prompt too long: ~{est_prompt_tokens} tokens exceeds model context of {max_ctx}. Please reduce prompt length.",
            )
        max_tokens = request.max_tokens
        if est_prompt_tokens + max_tokens > max_ctx:
            max_tokens = max(1, max_ctx - est_prompt_tokens)
        result = await validator.process_request(
            request.prompt, max_tokens=max_tokens, is_synthetic=False,
        )
        if result is None:
            raise HTTPException(status_code=503, detail="No miners available")

        return {
            "id": f"cmpl-{result['request_id']}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": result["text"],
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": result["input_tokens"],
                "completion_tokens": result["output_tokens"],
                "total_tokens": result["input_tokens"] + result["output_tokens"],
            },
        }

    # ── Embeddings stub ──────────────────────────────────────────────

    @app.post("/v1/embeddings")
    async def embeddings():
        """Embeddings not supported — return helpful error."""
        raise HTTPException(
            status_code=501,
            detail="Embeddings not available. This endpoint serves inference only via /v1/chat/completions.",
        )

    @app.get("/v1/models")
    async def models():
        """OpenAI-compatible models listing."""
        return {
            "object": "list",
            "data": [{
                "id": validator.model.config.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "inference-subnet",
            }]
        }

    # ── Live Dashboard ──────────────────────────────────────────────

    @app.get("/dashboard")
    async def dashboard():
        """Self-contained HTML monitoring dashboard. No external deps."""
        from fastapi.responses import HTMLResponse
        html = _DASHBOARD_HTML.replace("{{MODEL_NAME}}", html_mod.escape(validator.model.config.name))
        return HTMLResponse(html)

    return app


async def _stream_response(
    validator: HardenedGatewayValidator,
    prompt: str,
    max_tokens: int,
    model: str,
    session_id: Optional[str] = None,
    messages: list = None,
    sampling_params: dict = None,
) -> AsyncGenerator[str, None]:
    """
    Stream response in OpenAI SSE format.

    Pipes real token-by-token streaming from the miner when available,
    falling back to word-splitting for non-streaming miners.
    """
    # Track in-flight organic request for synthetic throttling
    validator.active_organic_requests += 1
    try:
        async for chunk in _stream_response_inner(validator, prompt, max_tokens, model, session_id, messages, sampling_params):
            yield chunk
    finally:
        validator.active_organic_requests = max(0, validator.active_organic_requests - 1)


async def _stream_response_inner(
    validator: HardenedGatewayValidator,
    prompt: str,
    max_tokens: int,
    model: str,
    session_id: Optional[str] = None,
    messages: list = None,
    sampling_params: dict = None,
) -> AsyncGenerator[str, None]:
    """Inner streaming implementation."""
    # Select miner
    miner = validator.router.select_miner(session_id=session_id)
    if not miner:
        yield f"data: {json.dumps({'error': 'No miners available'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    request_id = str(uuid.uuid4())
    if session_id:
        validator.router.session_router.set_affinity(session_id, miner.uid)

    # Pre-compute challenge BEFORE streaming (same as non-streaming path).
    # This allows bundling challenge params in the streaming request so miners
    # cannot distinguish challenge vs organic traffic by watching for a separate
    # /hidden_state call after streaming completes.
    should_challenge = secrets.randbelow(1000) < int(validator.config.CHALLENGE_RATE * 1000)
    challenge_params = None
    if should_challenge:
        if hasattr(validator.model, 'tokenizer'):
            if messages and hasattr(validator.model.tokenizer, 'apply_chat_template'):
                st_est_prompt = validator.model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                st_est_prompt = prompt
            prompt_tokens = validator.model.tokenizer.encode(st_est_prompt)
            estimated_seq_len = len(prompt_tokens) + max_tokens
        else:
            gen_result = validator.model.generate(prompt, max_tokens)
            estimated_seq_len = len(gen_result["all_tokens"])

        # Add ±10% noise to prevent miners from predicting challenge positions
        noise_range = max(1, estimated_seq_len // 10)
        estimated_seq_len += secrets.randbelow(2 * noise_range + 1) - noise_range
        estimated_seq_len = max(2, estimated_seq_len)

        num_layers = validator.model.config.num_layers
        challenge = validator.challenge_engine.create_challenge(
            request_id=request_id,
            num_layers=num_layers,
            seq_len=estimated_seq_len,
        )
        challenge_params = {
            "layer_index": challenge.layer_index,
            "token_index": challenge.token_index,
            "challenge": challenge,
        }
        if challenge.extra_points:
            challenge_params["extra_points"] = [
                [lyr, tok] for lyr, tok in challenge.extra_points
            ]

    # Try real SSE streaming from miner
    stream_url = f"{miner.endpoint}/inference/stream"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "request_id": request_id,
    }
    if messages:
        payload["messages"] = messages
    # Forward OpenAI sampling params to miner
    if sampling_params:
        for key in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "stop"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]
    # ALWAYS send challenge fields — prevents miners from fingerprinting by
    # presence/absence of challenge_layer. Same logic as _send_inference.
    if challenge_params:
        payload["challenge_layer"] = challenge_params["layer_index"]
        payload["challenge_token"] = challenge_params["token_index"]
        if challenge_params.get("extra_points"):
            payload["challenge_extra"] = challenge_params["extra_points"]
    else:
        dummy = validator._generate_dummy_challenge_fields(max_tokens, prompt=prompt, messages=messages)
        payload["challenge_layer"] = dummy["challenge_layer"]
        payload["challenge_token"] = dummy["challenge_token"]
        if "challenge_extra" in dummy:
            payload["challenge_extra"] = dummy["challenge_extra"]

    miner.active_requests += 1
    _counter_decremented = False  # Track whether report_success/failure handled the decrement
    streamed_ok = False
    all_text = ""
    final_meta = None
    max_stream_chars = 200_000  # Cap accumulated text to prevent memory exhaustion
    stream_wall_start = time.perf_counter()  # Wall-clock timing for metric validation
    first_token_wall_time = None  # When first token arrived (for TTFT)
    second_token_wall_time = None  # Anti-manipulation: confirm TTFT with 2nd token
    streamed_token_count = 0  # Count tokens as they arrive

    try:
        session = await validator._get_http_session()
        try:
            async with session.post(
                stream_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=validator.config.INFERENCE_TIMEOUT_S),
            ) as resp:
                if resp.status == 200 and resp.content_type == "text/event-stream":
                    streamed_ok = True
                    sse_buffer = ""
                    done_seen = False
                    async for raw_chunk in resp.content.iter_any():
                        sse_buffer += raw_chunk.decode("utf-8", errors="replace")
                        # Process complete SSE events (terminated by double newline)
                        while "\n\n" in sse_buffer:
                            event, sse_buffer = sse_buffer.split("\n\n", 1)
                            line = event.strip()
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                done_seen = True
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            token_text = data.get("token", "")
                            finish = data.get("finish_reason")

                            if finish == "stop":
                                final_meta = data
                            elif token_text:
                                streamed_token_count += 1
                                if first_token_wall_time is None:
                                    first_token_wall_time = time.perf_counter()
                                elif second_token_wall_time is None:
                                    second_token_wall_time = time.perf_counter()
                                if len(all_text) < max_stream_chars:
                                    all_text += token_text
                                # First chunk includes role per OpenAI spec
                                delta = {"content": token_text}
                                if streamed_token_count == 1:
                                    delta["role"] = "assistant"
                                chunk = {
                                    "id": f"chatcmpl-{request_id}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None,
                                    }],
                                }
                                # First chunk: include miner UID for debug transparency
                                if streamed_token_count == 1:
                                    chunk["miner_uid"] = miner.uid
                                yield f"data: {json.dumps(chunk)}\n\n"
                        if done_seen:
                            break
        except asyncio.TimeoutError:
            log.warning(f"Miner {miner.uid}: streaming timeout ({validator.config.INFERENCE_TIMEOUT_S}s)")
            validator.total_timeouts += 1
            validator.router.report_failure(miner, timeout=True)
            _counter_decremented = True
            streamed_ok = False
            # Send error chunk to client if we already started streaming tokens
            if streamed_token_count > 0:
                err_chunk = {"error": {"message": "upstream timeout", "type": "server_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        except aiohttp.ClientError as e:
            log.warning(f"Miner {miner.uid}: streaming connection error: {e}")
            validator.total_miner_errors += 1
            validator.router.report_failure(miner)
            _counter_decremented = True
            streamed_ok = False
            if streamed_token_count > 0:
                err_chunk = {"error": {"message": "upstream connection error", "type": "server_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return
        except Exception as e:
            log.warning(f"Miner {miner.uid}: streaming error: {type(e).__name__}: {e}")
            validator.total_miner_errors += 1
            validator.router.report_failure(miner)
            _counter_decremented = True
            streamed_ok = False
            if streamed_token_count > 0:
                err_chunk = {"error": {"message": "upstream error", "type": "server_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

        # Record metrics from streamed response — wall-clock validated
        if streamed_ok and final_meta:
            stream_wall_end = time.perf_counter()
            wall_time_s = stream_wall_end - stream_wall_start
            wall_time_ms = wall_time_s * 1000.0

            # Wall-clock TTFT: time from request start to first token
            # Anti-manipulation: if the gap between 1st and 2nd token is >5× the
            # TTFT, the miner likely sent a fake first token to game TTFT.
            # In that case, use the 2nd token's arrival as the real TTFT.
            reported_ttft = final_meta.get("ttft_ms", 0)
            if first_token_wall_time is not None:
                wall_ttft_ms = (first_token_wall_time - stream_wall_start) * 1000.0
                # Detect fake first token: large gap between token 1 and token 2
                if second_token_wall_time is not None:
                    gap_ms = (second_token_wall_time - first_token_wall_time) * 1000.0
                    if wall_ttft_ms > 0 and gap_ms > wall_ttft_ms * 5.0:
                        # Suspicious: 2nd token took >5× longer than TTFT
                        # Use 2nd token arrival as the true TTFT
                        wall_ttft_ms = (second_token_wall_time - stream_wall_start) * 1000.0
                        log.debug(f"Miner {miner.uid}: fake-first-token detected, using 2nd token for TTFT")
                # Miner-reported TTFT must not be lower than wall-clock measurement
                # (allow 20% slack for network overhead)
                ttft_ms = max(reported_ttft, wall_ttft_ms * 0.8)
                # Also cap at total wall time
                ttft_ms = min(ttft_ms, wall_time_ms)
            else:
                ttft_ms = wall_time_ms  # No tokens received, TTFT = full time

            # Wall-clock TPS: actual tokens observed / wall time
            reported_tps = final_meta.get("tokens_per_sec", 0)
            if wall_time_s > 0 and streamed_token_count > 0:
                wall_tps = streamed_token_count / wall_time_s
                # Miner cannot claim faster than 1.5× wall-clock TPS
                tps = min(reported_tps, wall_tps * 1.5)
            else:
                tps = 0.0  # No tokens = no TPS credit

            validator.router.report_success(miner, ttft_ms=ttft_ms, tps=tps)
            _counter_decremented = True
            validator.total_organic += 1

            # Verify inline challenge result (bundled with streaming response)
            all_token_ids = final_meta.get("all_token_ids")
            challenge_passed = None  # None = no challenge, True/False = result
            cos_sim = 0.0
            challenge_latency = 0.0
            challenge_result = None

            if should_challenge and challenge_params:
                challenge = challenge_params["challenge"]
                inline_result = final_meta.get("challenge_result")

                # Get tokens for reference computation
                # When messages are provided, use chat template for correct token alignment
                if messages and hasattr(validator.model, 'tokenizer') and hasattr(validator.model.tokenizer, 'apply_chat_template'):
                    st_eff_prompt = validator.model.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                else:
                    st_eff_prompt = prompt

                miner_token_ids = all_token_ids
                if miner_token_ids:
                    valid, reason = validator._validate_token_ids(miner_token_ids, st_eff_prompt, all_text)
                    if valid:
                        all_tokens = miner_token_ids
                    else:
                        log.warning(f"Miner {miner.uid} streaming token_ids rejected: {reason}")
                        all_tokens = None
                else:
                    all_tokens = None

                # Fallback: validator produces token IDs if miner didn't (or they were rejected)
                if not all_tokens:
                    if hasattr(validator.model, 'tokenizer'):
                        full_text = st_eff_prompt + all_text
                        all_tokens = validator.model.tokenizer.encode(full_text)
                    elif all_text:
                        gen_result = validator.model.generate(prompt, max_tokens)
                        all_tokens = gen_result.get("all_tokens")

                if all_tokens:
                    if inline_result and "error" not in inline_result and "hidden_state" in inline_result:
                        # Verify inline challenge result — use gateway wall time,
                        # not miner's self-reported latency_ms
                        challenge_result = await validator._verify_inline_challenge(
                            challenge, inline_result, all_tokens,
                            gateway_wall_time_ms=wall_time_ms,
                        )
                    elif inline_result is None:
                        # Miner doesn't support inline challenges — auto-fail.
                        # Legacy /hidden_state fallback reveals challenge classification.
                        log.warning(f"Miner {miner.uid}: no inline streaming challenge support — auto-fail")
                        challenge_result = {
                            "passed": False,
                            "reason": "miner does not support inline challenges",
                            "latency_ms": 0.0,
                            "cosine_sim": 0.0,
                            "layer": challenge.layer_index,
                            "token_pos": challenge.token_index,
                        }
                    else:
                        # Miner returned an error (cache_miss, etc.)
                        challenge_result = {
                            "passed": False,
                            "reason": inline_result.get("error", "unknown error"),
                            "latency_ms": inline_result.get("latency_ms", 0),
                            "cosine_sim": 0.0,
                            "layer": challenge.layer_index,
                            "token_pos": challenge.token_index,
                        }

                    challenge_passed = challenge_result["passed"]
                    cos_sim = challenge_result["cosine_sim"]
                    challenge_latency = challenge_result["latency_ms"]

            medians_ttft, medians_tps = validator.scoring.get_miner_medians()
            speed = compute_speed_score(ttft_ms, tps, miner_medians_ttft=medians_ttft, miner_medians_tps=medians_tps)
            verification = compute_verification_score(challenge_passed, cos_sim, challenge_latency)
            quality = compute_output_quality(all_text)
            score = RequestScore(
                request_id=request_id, miner_uid=miner.uid,
                timestamp=time.time(), is_synthetic=False,
                speed_score=speed, verification_score=verification,
                quality_score=quality, ttft_ms=ttft_ms, tokens_per_sec=tps,
                cosine_sim=cos_sim,
                challenge_latency_ms=challenge_latency,
                challenge_passed=challenge_passed,
            )
            validator.scoring.record_request(score)

            # Update router blocked UIDs after each scored request
            validator.router.update_blocked_uids(validator.scoring)

            # Publish audit record (parity with non-streaming path)
            audit = AuditRecord(
                request_id=request_id,
                miner_uid=miner.uid,
                miner_hotkey=miner.hotkey,
                is_synthetic=False,
                prompt=prompt,
                response_text=all_text,
                ttft_ms=ttft_ms,
                tokens_per_sec=tps,
                input_tokens=0,
                output_tokens=0,
                challenge_layer=challenge_result["layer"] if challenge_result else None,
                challenge_token_pos=challenge_result["token_pos"] if challenge_result else None,
                cosine_sim=cos_sim if challenge_result else None,
                challenge_latency_ms=challenge_latency if challenge_result else None,
                challenge_passed=challenge_passed if challenge_result else None,
                speed_score=speed,
                verification_score=verification,
                points_awarded=score.points,
            )
            validator.r2.publish(audit)

            # Feed collusion detector (parity with non-streaming path)
            validator.collusion_detector.record_timing(
                MinerTimingSample(miner.uid, ttft_ms, tps, time.time())
            )
            # Only record error events when a challenge was actually attempted
            if challenge_passed is not None:
                validator.collusion_detector.record_error(
                    MinerErrorEvent(miner.uid, challenge_passed, time.time())
                )

        if not streamed_ok:
            # Fallback: use regular inference + word-splitting
            log.info(f"Miner {miner.uid}: streaming failed, falling back to non-streaming inference")
            # Note: report_failure already decremented active_requests in the error handlers above.
            # Clean up orphaned challenge from the failed streaming attempt.
            # process_request will create its own challenge for the fallback miner.
            if challenge_params and "challenge" in challenge_params:
                orphaned = challenge_params["challenge"]
                validator.challenge_engine._pending.pop(orphaned.challenge_id, None)
            result = await validator.process_request(
                prompt, max_tokens=max_tokens, is_synthetic=False,
                session_id=session_id, messages=messages,
            )
            if result is None:
                yield f"data: {json.dumps({'error': 'No miners available'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            text = result["text"]
            words = text.split()
            for i, word in enumerate(words):
                delta = {"content": word + (" " if i < len(words) - 1 else "")}
                if i == 0:
                    delta["role"] = "assistant"
                chunk = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

    finally:
        # Safety net: if neither report_success nor report_failure ran
        # (e.g. scoring code raised after streaming), decrement here.
        if not _counter_decremented:
            miner.active_requests = max(0, miner.active_requests - 1)

    # Final chunk with usage stats (OpenAI convention)
    prompt_tok_count = final_meta.get("prompt_tokens", 0) if final_meta else 0
    completion_tok_count = streamed_token_count
    final = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tok_count,
            "completion_tokens": completion_tok_count,
            "total_tokens": prompt_tok_count + completion_tok_count,
        },
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_anthropic_response(
    validator,
    prompt: str,
    max_tokens: int,
    model: str,
    messages: list = None,
    sampling_params: dict = None,
    session_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream response in Anthropic Messages SSE format.

    Internally uses the same miner routing as OpenAI streaming, but emits
    Anthropic-format SSE events (message_start, content_block_start/delta/stop,
    message_delta, message_stop).
    """
    request_id = str(uuid.uuid4())
    msg_id = f"msg_{request_id[:24]}"

    # Select miner (with session affinity for KV cache reuse)
    miner = validator.router.select_miner(session_id=session_id)
    if not miner:
        # Emit an error as an Anthropic-style error event
        err = {"type": "error", "error": {"type": "overloaded_error", "message": "No miners available"}}
        yield f"event: error\ndata: {json.dumps(err)}\n\n"
        return

    # Register session affinity for future requests (KV cache reuse)
    if session_id:
        validator.router.session_router.set_affinity(session_id, miner.uid)

    # Build payload for miner (same as OpenAI streaming path)
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "request_id": request_id,
    }
    if messages:
        payload["messages"] = messages
    if sampling_params:
        for key in ("temperature", "top_p", "stop"):
            if sampling_params.get(key) is not None:
                payload[key] = sampling_params[key]

    # Dummy challenge fields (no verification on Anthropic path for now)
    dummy = validator._generate_dummy_challenge_fields(max_tokens, prompt=prompt, messages=messages)
    payload["challenge_layer"] = dummy["challenge_layer"]
    payload["challenge_token"] = dummy["challenge_token"]
    if "challenge_extra" in dummy:
        payload["challenge_extra"] = dummy["challenge_extra"]

    # Emit message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    # Emit content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    # Emit ping
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    miner.active_requests += 1
    all_text = ""
    streamed_ok = False
    input_tokens = 0
    output_tokens = 0

    try:
        stream_url = f"{miner.endpoint}/inference/stream"
        session = await validator._get_http_session()
        try:
            async with session.post(
                stream_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=validator.config.INFERENCE_TIMEOUT_S),
            ) as resp:
                if resp.status == 200 and resp.content_type == "text/event-stream":
                    streamed_ok = True
                    sse_buffer = ""
                    done_seen = False
                    async for raw_chunk in resp.content.iter_any():
                        sse_buffer += raw_chunk.decode("utf-8", errors="replace")
                        while "\n\n" in sse_buffer:
                            event, sse_buffer = sse_buffer.split("\n\n", 1)
                            line = event.strip()
                            if not line.startswith("data: "):
                                continue
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                done_seen = True
                                break
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            token_text = data.get("token", "")
                            finish = data.get("finish_reason")

                            if finish == "stop":
                                input_tokens = data.get("input_tokens", 0)
                                output_tokens = data.get("output_tokens", 0)
                                if isinstance(output_tokens, list):
                                    output_tokens = len(output_tokens)
                            elif token_text:
                                all_text += token_text
                                # Emit content_block_delta
                                delta_event = {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": token_text},
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                        if done_seen:
                            break
        except (asyncio.TimeoutError, aiohttp.ClientError, Exception) as e:
            log.warning(f"Anthropic stream: miner {miner.uid} error: {e}")
            validator.router.report_failure(miner)
            streamed_ok = False

        if not streamed_ok:
            # Fallback: non-streaming request
            result = await validator.process_request(
                prompt,
                max_tokens=max_tokens,
                is_synthetic=False,
                messages=messages,
                sampling_params=sampling_params,
            )
            if result:
                all_text = result.get("text", "")
                input_tokens = result.get("input_tokens", 0)
                out = result.get("output_tokens", 0)
                output_tokens = out if isinstance(out, int) else len(out)
                # Emit all text as a single delta
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": all_text},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
            else:
                all_text = "Error: no miners available"
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": all_text},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

    finally:
        miner.active_requests = max(0, miner.active_requests - 1)

    if streamed_ok:
        try:
            validator.router.report_success(miner, ttft_ms=0.0, tps=0.0)
        except Exception:
            pass
        validator.total_organic += 1

    # Estimate tokens if not reported
    if not output_tokens:
        output_tokens = len(all_text.split())
    if not input_tokens:
        input_tokens = len(prompt) // 4

    stop_reason = "max_tokens" if output_tokens >= max_tokens else "end_turn"

    # Emit closing events
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ── CLI Runner ───────────────────────────────────────────────────────────────

async def run_gateway(args):
    """Run the hardened gateway validator."""
    # Cap the default ThreadPoolExecutor to prevent unbounded thread growth.
    # asyncio's create_connection() calls getaddrinfo() in this pool — with 60+
    # miners being probed concurrently, the default pool grows unboundedly.
    # 8 threads is enough for concurrent getaddrinfo (instant for IP addresses).
    import concurrent.futures
    _bounded_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    asyncio.get_event_loop().set_default_executor(_bounded_pool)

    config = GatewayConfig()
    config.GATEWAY_PORT = args.port
    config.EPOCH_LENGTH_S = args.epoch_length
    config.SYNTHETIC_INTERVAL_S = args.synthetic_interval
    config.MAX_CONTEXT_TOKENS = getattr(args, 'max_context_tokens', 4096)

    if args.api_keys:
        config.API_KEYS = set(args.api_keys.split(","))
    if args.monitoring_keys:
        config.MONITORING_KEYS = set(args.monitoring_keys.split(","))
    relay_secret = getattr(args, 'relay_secret', '') or os.environ.get("INTERNAL_RELAY_SECRET", "")
    if relay_secret:
        config.INTERNAL_RELAY_SECRET = relay_secret
    miner_secret = getattr(args, 'miner_secret', '') or os.environ.get("MINER_VALIDATOR_SECRET", "")
    if miner_secret:
        config.MINER_VALIDATOR_SECRET = miner_secret

    model = load_validator_model(getattr(args, 'model', None))

    # Auto-disable challenges when using mock model (hidden states won't match real miners)
    model_name = getattr(args, 'model', None)
    if not model_name or model_name == "mock":
        log.warning("Using mock model — disabling hidden state challenges (speed-only scoring)")
        config.CHALLENGE_RATE = 0.0

    # Override challenge rate if specified
    if args.challenge_rate is not None:
        config.CHALLENGE_RATE = max(0.0, min(1.0, args.challenge_rate))
        log.info(f"Challenge rate set to {config.CHALLENGE_RATE:.1%}")

    # Chain integration (optional)
    chain = None
    if args.wallet:
        chain = ChainWeightSetter(
            wallet_name=args.wallet,
            hotkey=args.hotkey,
            netuid=args.netuid,
            network=args.network,
            wallet_path=args.wallet_path,
        )
        log.info(f"Chain integration: wallet={args.wallet} netuid={args.netuid} network={args.network}")

    # Metagraph discovery (optional — replaces or supplements static --miners)
    discovery = None
    if args.discover:
        discovery = MetagraphDiscovery(
            netuid=args.netuid,
            network=args.network,
            validator_hotkey=None,  # Will be set from wallet if available
        )
        log.info(f"Metagraph discovery: netuid={args.netuid} network={args.network}")

    validator = HardenedGatewayValidator(
        miner_endpoints=args.miners or [],
        config=config,
        r2_local_dir=args.r2_dir,
        model=model,
        chain_weight_setter=chain,
        metagraph_discovery=discovery,
    )

    # Set up auditor polling for external challenge data (split architecture)
    auditor_url = getattr(args, 'auditor_url', None)

    # C4-3: HMAC-authenticate auditor↔gateway communication
    _AUDITOR_SECRET = os.environ.get("AUDITOR_SECRET", "")

    def _auditor_auth_headers() -> dict:
        """Build HMAC auth headers for auditor requests."""
        if not _AUDITOR_SECRET:
            return {}
        ts = str(int(time.time()))
        sig = hmac.new(_AUDITOR_SECRET.encode(), ts.encode(), hashlib.sha256).hexdigest()
        return {"Authorization": f"Bearer {ts}:{sig}"}

    async def auditor_poll_loop():
        """Periodically fetch challenge data from external audit_validator.

        In split architecture, the gateway defers all challenges to the auditor.
        This loop syncs challenge pass/fail counts back into the gateway's scorer
        so weight computation doesn't penalize miners for 'zero challenges'.
        """
        if not auditor_url:
            return
        import aiohttp
        # Single reusable session with AsyncResolver — avoids ThreadedResolver thread leak
        # (creating new ClientSession() each iteration spawns OS threads that never die)
        _poll_connector = aiohttp.TCPConnector(limit=10, resolver=aiohttp.AsyncResolver())
        _poll_session = aiohttp.ClientSession(
            connector=_poll_connector,
            timeout=aiohttp.ClientTimeout(total=5),
        )
        log.info(f"[AUDITOR-POLL] Polling {auditor_url}/v1/scoreboard for challenge data")
        try:
            while True:
                # Poll scoreboard and health separately so one failing doesn't block the other
                try:
                    async with _poll_session.get(f"{auditor_url}/v1/scoreboard", headers=_auditor_auth_headers()) as resp:
                        if resp.status == 200:
                            # C5-8: Verify response signature to prevent MITM score injection
                            resp_body = await resp.read()
                            if _AUDITOR_SECRET:
                                resp_sig = resp.headers.get("X-Signature", "")
                                expected_sig = hmac.new(
                                    _AUDITOR_SECRET.encode(), resp_body, hashlib.sha256
                                ).hexdigest()
                                if not hmac.compare_digest(expected_sig, resp_sig):
                                    log.warning("[AUDITOR-POLL] Response signature mismatch — possible MITM")
                                    await asyncio.sleep(30)
                                    continue
                            data = json.loads(resp_body)
                            blocked = set()
                            synced = 0
                            for m in data.get("miners", []):
                                uid = m["uid"]
                                net_points = m.get("net_points", 0)
                                pass_rate = m.get("pass_rate", 1.0)
                                requests = m.get("requests", 0)
                                # Block miners with negative net_points and enough samples
                                if requests >= 3 and net_points < 0:
                                    blocked.add(uid)
                                # Block miners with very low pass_rate AND negative score
                                elif requests >= 8 and pass_rate < 0.3 and net_points <= 0:
                                    blocked.add(uid)

                                # Sync challenge counts from auditor into gateway scorer.
                                # The gateway's own scorer sees challenge_passed=None for
                                # all deferred requests, so without this sync, every miner
                                # hits the "zero challenges" 0.05x weight penalty.
                                auditor_passed = m.get("passed_challenges", 0)
                                auditor_failed = m.get("failed_challenges", 0)
                                if auditor_passed + auditor_failed > 0:
                                    stats = validator.scoring._get_stats(uid)
                                    stats.passed_challenges = max(stats.passed_challenges, auditor_passed)
                                    stats.failed_challenges = max(stats.failed_challenges, auditor_failed)
                                # C4 H4-1: Sync void counts from auditor
                                auditor_voided = m.get("voided_challenges", 0)
                                if auditor_voided > 0:
                                    stats = validator.scoring._get_stats(uid)
                                    stats.voided_challenges = max(stats.voided_challenges, auditor_voided)
                                    # VOID-only miners: auditor challenged them but all results
                                    # were inconclusive (CPU/GPU divergence). This is NOT the same
                                    # as "never challenged" — seed cosine so TPS bonus works.
                                    if auditor_passed == 0 and auditor_failed == 0 and auditor_voided >= 3:
                                        if not stats.cosine_values:
                                            stats.cosine_values = [0.95]  # Conservative seed
                                        if stats.passed_challenges == 0:
                                            stats.passed_challenges = 1  # Virtual pass from void evidence
                                    synced += 1
                                # Sync cosine data from auditor for TPS bonus damping
                                auditor_cosine = m.get("avg_cosine", 0.0)
                                if auditor_cosine > 0 and (auditor_passed + auditor_failed > 0):
                                    stats = validator.scoring._get_stats(uid)
                                    if not stats.cosine_values or auditor_cosine > stats.avg_cosine:
                                        stats.cosine_values = [auditor_cosine]
                            validator.router.update_auditor_blocked(blocked)
                            log.info(f"[AUDITOR-POLL] Synced {len(data.get('miners', []))} miners ({synced} with voids), blocked={len(blocked)}")
                except Exception as e:
                    log.warning(f"[AUDITOR-POLL] Scoreboard error: {type(e).__name__}: {e}")

                # Separate try block for health poll — don't let it crash the scoreboard sync
                try:
                    async with _poll_session.get(f"{auditor_url}/health") as health_resp:
                        if health_resp.status == 200:
                            health_data = await health_resp.json()
                            challenge_rates = health_data.get("challenge_rates", {})
                            for uid_str, rate in challenge_rates.items():
                                uid = int(uid_str)
                                rate = float(rate)
                                stats = validator.scoring._get_stats(uid)
                                if rate <= 0.30:
                                    if not stats.cosine_values:
                                        stats.cosine_values = [0.99]
                                    if stats.passed_challenges == 0 and stats.failed_challenges == 0:
                                        stats.passed_challenges = 1  # Virtual pass from trust
                except Exception as e:
                    log.warning(f"[AUDITOR-POLL] Health error: {type(e).__name__}: {e}")

                await asyncio.sleep(30)
        finally:
            await _poll_session.close()

    # ── Pre-flight: poll auditor + health-check miners BEFORE accepting traffic ──
    if auditor_url:
        try:
            import aiohttp as _pf_aiohttp
            _pf_conn = _pf_aiohttp.TCPConnector(limit=10, resolver=_pf_aiohttp.AsyncResolver())
            async with _pf_aiohttp.ClientSession(connector=_pf_conn, timeout=_pf_aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(
                    f"{auditor_url}/v1/scoreboard",
                    headers=_auditor_auth_headers(),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        blocked = set()
                        for m in data.get("miners", []):
                            uid = m["uid"]
                            net_pts = m.get("net_points", 0)
                            pr = m.get("pass_rate", 1.0)
                            reqs = m.get("requests", 0)
                            if reqs >= 3 and net_pts < 0:
                                blocked.add(uid)
                            elif reqs >= 8 and pr < 0.3 and net_pts <= 0:
                                blocked.add(uid)
                        validator.router.update_auditor_blocked(blocked)
                        log.info(f"[STARTUP] Pre-loaded auditor blocked UIDs: {blocked}")
        except Exception as e:
            log.warning(f"[STARTUP] Could not pre-load auditor data: {e}")

    # Quick health-check all static miners CONCURRENTLY — mark unreachable ones dead immediately
    # Use the validator's shared session (AsyncResolver) to avoid spawning ThreadedResolver threads
    async def _startup_health_check():
        session = await validator._get_http_session()
        _sem = asyncio.Semaphore(10)
        async def _check(uid, miner):
            async with _sem:
                try:
                    async with session.get(
                        f"{miner.endpoint}/health",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as resp:
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}")
                except Exception:
                    miner.alive = False
                    miner.reliability_score = 0.0
                    log.info(f"[STARTUP] Miner {uid} ({miner.endpoint}): unreachable, marked DEAD")
        await asyncio.gather(*[_check(uid, m) for uid, m in list(validator.router.miners.items())])
    await _startup_health_check()

    app = create_gateway_app(validator)

    uvi_config = uvicorn.Config(
        app, host="0.0.0.0", port=args.port, log_level="warning", log_config=None,
        timeout_keep_alive=2,    # Aggressively close idle keep-alive connections
        limit_concurrency=2000,  # Raised from 200 — stress testing at scale
        http="h11",              # h11 handles connection_lost properly (httptools leaks CLOSE-WAIT)
        limit_max_requests=10000, # Recycle process to clear any leaked sockets
    )
    server = uvicorn.Server(uvi_config)
    # Disable uvicorn's signal handling — it re-raises captured signals via
    # signal.raise_signal() which bypasses Python try/except and kills the process.
    # We handle shutdown ourselves via server.should_exit.
    import contextlib
    server.capture_signals = contextlib.contextmanager(lambda: (yield))  # type: ignore[assignment]

    async def server_task():
        try:
            await server.serve()
        except (KeyboardInterrupt, SystemExit):
            pass

    async def main_loop():
        await asyncio.sleep(1)
        # Re-ensure our logger has exactly one stdout handler after uvicorn startup.
        # Note: PM2 routes stdout to the "entire log" path (arbos.log), not gateway.log.
        # Use `pm2 logs proxy_gateway` or check arbos.log for gateway output.
        log.handlers.clear()
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        log.addHandler(_h)
        log.propagate = False
        log.info(f"Hardened Gateway v{GATEWAY_VERSION} running on port {args.port}")
        log.info(f"Miners: {args.miners}")
        log.info(f"Epoch: {args.epoch_length}s | Synthetic interval: ~{args.synthetic_interval}s (with jitter)")
        if config.API_KEYS:
            log.info(f"Auth: {len(config.API_KEYS)} API keys configured")
        else:
            log.info("Auth: disabled (no API keys)")
        if config.MONITORING_KEYS:
            log.info(f"Monitoring auth: {len(config.MONITORING_KEYS)} keys configured")
        else:
            log.info(
                "Monitoring auth: locked (no --monitoring-keys set, endpoints return 401)"
            )
        if chain:
            log.info(f"Chain: wallet={args.wallet} netuid={args.netuid} network={args.network}")
        else:
            log.info("Chain: disabled (no --wallet specified)")

        async def event_loop_watchdog():
            """Detect event loop stalls and log them."""
            while True:
                t0 = time.time()
                await asyncio.sleep(1)
                stall = time.time() - t0 - 1.0
                if stall > 0.5:
                    log.warning(f"[STALL] Event loop blocked for {stall:.1f}s")

        synth_task = asyncio.create_task(validator.synthetic_loop())
        epoch_task = asyncio.create_task(validator.epoch_loop())
        cache_task = asyncio.create_task(validator.cache_probe_loop())
        cross_task = asyncio.create_task(validator.cross_probe_loop())
        health_task = asyncio.create_task(validator.health_recovery_loop())
        discovery_task = asyncio.create_task(validator.discovery_loop())
        auditor_task = asyncio.create_task(auditor_poll_loop())
        watchdog_task = asyncio.create_task(event_loop_watchdog())

        tasks = [synth_task, epoch_task, cache_task, cross_task, health_task, discovery_task, auditor_task, watchdog_task]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await validator.close()

    server.should_exit = False
    try:
        await asyncio.gather(server_task(), main_loop())
    except (KeyboardInterrupt, asyncio.CancelledError):
        server.should_exit = True
        await validator.close()


# ── Dashboard HTML (self-contained, no external deps) ─────────────────────

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inference Subnet Dashboard</title>
<style>
:root { --bg:#0d1117; --card:#161b22; --border:#30363d; --text:#c9d1d9; --dim:#8b949e;
        --green:#3fb950; --red:#f85149; --blue:#58a6ff; --orange:#d29922; --purple:#bc8cff;
        --gold:#ffd700; --silver:#c0c0c0; --bronze:#cd7f32; }
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
     background:var(--bg);color:var(--text);line-height:1.5;padding:16px;max-width:1400px;margin:0 auto}
h1{font-size:22px;font-weight:600;margin-bottom:4px}
.subtitle{color:var(--dim);font-size:13px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:20px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}
.stat .label{font-size:11px;text-transform:uppercase;letter-spacing:.5px;color:var(--dim)}
.stat .value{font-size:26px;font-weight:700;margin-top:2px}
.stat .sub{font-size:11px;color:var(--dim);margin-top:2px}
.green{color:var(--green)} .red{color:var(--red)} .blue{color:var(--blue)}
.orange{color:var(--orange)} .purple{color:var(--purple)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--border);border-radius:8px;overflow:hidden}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--border);font-size:13px}
th{background:#1c2128;color:var(--dim);font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:.5px}
tr:last-child td{border-bottom:none}
.health{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.health.up{background:var(--green)} .health.down{background:var(--red)}
.section{margin-bottom:20px}
.section h2{font-size:14px;font-weight:600;margin-bottom:8px;color:var(--dim)}
canvas{background:var(--card);border:1px solid var(--border);border-radius:8px;width:100%!important;height:180px!important}
.chart-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
@media(max-width:700px){.chart-row{grid-template-columns:1fr}}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600}
.badge.ok{background:#0d2818;color:var(--green)} .badge.warn{background:#2d1b00;color:var(--orange)}
.badge.err{background:#2d0000;color:var(--red)}
.rank{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;font-size:12px;font-weight:700}
.rank-1{background:var(--gold);color:#000} .rank-2{background:var(--silver);color:#000} .rank-3{background:var(--bronze);color:#fff}
.rank-n{background:var(--border);color:var(--dim)}
.bar{height:6px;border-radius:3px;background:var(--border);overflow:hidden;margin-top:4px}
.bar-fill{height:100%;border-radius:3px;transition:width 0.5s}
.tabs{display:flex;gap:0;margin-bottom:16px;border-bottom:1px solid var(--border)}
.tab{padding:8px 16px;cursor:pointer;color:var(--dim);font-size:13px;font-weight:500;border-bottom:2px solid transparent;transition:all 0.2s}
.tab:hover{color:var(--text)} .tab.active{color:var(--blue);border-bottom-color:var(--blue)}
.tab-content{display:none} .tab-content.active{display:block}
#updated{color:var(--dim);font-size:11px;text-align:right;margin-top:8px}
.uptime-bar{display:flex;gap:1px;height:18px;align-items:flex-end}
.uptime-dot{flex:1;min-width:3px;border-radius:1px}
</style>
</head>
<body>
<h1>Inference Subnet &mdash; {{MODEL_NAME}}</h1>
<p class="subtitle">Live monitoring dashboard &bull; auto-refreshes every 5s</p>

<div class="grid" id="stats"></div>

<div class="tabs">
  <div class="tab active" data-tab="overview">Overview</div>
  <div class="tab" data-tab="leaderboard">Leaderboard</div>
  <div class="tab" data-tab="miners">Miners</div>
  <div class="tab" data-tab="epochs">Epoch History</div>
</div>

<!-- Overview Tab -->
<div class="tab-content active" id="tab-overview">
<div class="chart-row">
<div><div class="section"><h2>Throughput (requests/tick)</h2></div><canvas id="chartReqs"></canvas></div>
<div><div class="section"><h2>Challenge Verification</h2></div><canvas id="chartChal"></canvas></div>
</div>
<div class="chart-row">
<div><div class="section"><h2>Cumulative Requests</h2></div><canvas id="chartCumulative"></canvas></div>
<div><div class="section"><h2>Latency (TTFT ms)</h2></div><canvas id="chartLatency"></canvas></div>
</div>
</div>

<!-- Leaderboard Tab -->
<div class="tab-content" id="tab-leaderboard">
<div class="section"><h2>Miner Leaderboard &mdash; ranked by net points</h2></div>
<table><thead><tr>
<th>Rank</th><th>UID</th><th>Net Points</th><th>Score Bar</th>
<th>Requests</th><th>Pass Rate</th><th>Avg TPS</th><th>Reliability</th><th>Status</th>
</tr></thead><tbody id="leaderRows"></tbody></table>
</div>

<!-- Miners Tab -->
<div class="tab-content" id="tab-miners">
<div class="section"><h2>Miner Details</h2></div>
<table><thead><tr>
<th>Status</th><th>UID</th><th>Endpoint</th><th>Reliability</th>
<th>Served</th><th>Failed</th><th>TTFT (ms)</th><th>TPS</th><th>Active</th>
</tr></thead><tbody id="minerRows"></tbody></table>

<div style="margin-top:20px">
<div class="section"><h2>Scoreboard (current epoch)</h2></div>
<table><thead><tr>
<th>UID</th><th>Net Pts</th><th>Organic</th><th>Synthetic</th>
<th>Pass Rate</th><th>Consistency</th><th>Divergence</th><th>Suspect</th>
</tr></thead><tbody id="scoreRows"></tbody></table>
</div>
</div>

<!-- Epochs Tab -->
<div class="tab-content" id="tab-epochs">
<div class="section"><h2>Epoch History</h2></div>
<table><thead><tr>
<th>Epoch</th><th>Duration</th><th>Organic</th><th>Synthetic</th>
<th>Challenges</th><th>Pass Rate</th><th>Miners</th>
</tr></thead><tbody id="epochRows"></tbody></table>
</div>

<div id="updated"></div>

<script>
// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

function miniChart(canvasId, datasets, opts) {
  const c = document.getElementById(canvasId);
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.width = c.offsetWidth * (window.devicePixelRatio||1);
  const H = c.height = c.offsetHeight * (window.devicePixelRatio||1);
  ctx.scale(window.devicePixelRatio||1, window.devicePixelRatio||1);
  const w = c.offsetWidth, h = c.offsetHeight;
  const pad = {t:20,r:12,b:24,l:44};
  const pw = w-pad.l-pad.r, ph = h-pad.t-pad.b;
  ctx.clearRect(0,0,w,h);

  let allVals = datasets.flatMap(d=>d.data);
  let maxV = Math.max(...allVals, 1);
  let n = datasets[0].data.length;
  if (n === 0) return;

  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  for (let i=0;i<=4;i++){
    let y = pad.t + ph - (ph*i/4);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+pw,y); ctx.stroke();
    ctx.fillStyle='#8b949e'; ctx.font='10px sans-serif'; ctx.textAlign='right';
    ctx.fillText(Math.round(maxV*i/4), pad.l-4, y+3);
  }

  if (opts && opts.line) {
    // Line chart mode
    datasets.forEach(ds => {
      ctx.strokeStyle = ds.color || '#58a6ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ds.data.forEach((v,i) => {
        let x = pad.l + (i/(n-1||1))*pw;
        let y = pad.t + ph - (v/maxV)*ph;
        i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
      });
      ctx.stroke();
      // Fill area under line
      ctx.globalAlpha = 0.1;
      ctx.fillStyle = ds.color;
      ctx.lineTo(pad.l + pw, pad.t + ph);
      ctx.lineTo(pad.l, pad.t + ph);
      ctx.fill();
      ctx.globalAlpha = 1;
    });
  } else {
    // Bar chart mode
    let barW = pw / n;
    datasets.forEach((ds,di) => {
      ctx.fillStyle = ds.color || '#58a6ff';
      ds.data.forEach((v,i) => {
        let x = pad.l + i*barW + di*(barW/(datasets.length+1));
        let bw = barW/(datasets.length+1);
        let bh = (v/maxV)*ph;
        ctx.fillRect(x, pad.t+ph-bh, bw-1, bh);
      });
    });
  }

  ctx.fillStyle='#8b949e'; ctx.font='10px sans-serif'; ctx.textAlign='center';
  datasets[0].labels?.forEach((lbl,i) => {
    let x = (opts && opts.line) ? pad.l + (i/(n-1||1))*pw : pad.l + i*(pw/n) + (pw/n)/2;
    ctx.fillText(lbl, x, h-4);
  });

  let lx = pad.l;
  datasets.forEach(ds => {
    ctx.fillStyle = ds.color; ctx.fillRect(lx, 4, 10, 10);
    ctx.fillStyle = '#c9d1d9'; ctx.font = '10px sans-serif'; ctx.textAlign='left';
    ctx.fillText(ds.label, lx+13, 13);
    lx += ctx.measureText(ds.label).width + 26;
  });
}

const history = {organic:[], synthetic:[], passed:[], failed:[], ttft:[]};
const MAX_HIST = 60;

function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function rankClass(i) {
  if (i===0) return 'rank-1'; if (i===1) return 'rank-2'; if (i===2) return 'rank-3'; return 'rank-n';
}

async function refresh() {
  try {
    const [hRes, sRes, eRes] = await Promise.all([
      fetch('/v1/health').then(r=>r.json()),
      fetch('/v1/scoreboard').then(r=>r.json()).catch(()=>null),
      fetch('/v1/epochs').then(r=>r.json()).catch(()=>null)
    ]);

    // Stats cards
    const chal = hRes.challenges || {};
    const passRate = chal.total > 0
      ? ((chal.passed/chal.total)*100).toFixed(1) : '–';
    const passClass = parseFloat(passRate) >= 99 ? 'green' : parseFloat(passRate) >= 90 ? 'orange' : 'red';
    const totalReqs = (hRes.total_organic||0) + (hRes.total_synthetic||0);
    document.getElementById('stats').innerHTML = `
      <div class="stat"><div class="label">Status</div><div class="value"><span class="badge ${hRes.miners_alive>0?'ok':'err'}">${hRes.miners_alive>0?'ONLINE':'DOWN'}</span></div><div class="sub">${hRes.miners_alive}/${hRes.miners_total} miners</div></div>
      <div class="stat"><div class="label">Epoch</div><div class="value blue">${hRes.epoch}</div></div>
      <div class="stat"><div class="label">Total Requests</div><div class="value green">${totalReqs}</div><div class="sub">${hRes.total_organic||0} organic / ${hRes.total_synthetic||0} synthetic</div></div>
      <div class="stat"><div class="label">Challenge Pass</div><div class="value ${passClass}">${passRate}%</div><div class="sub">${chal.passed||0}/${chal.total||0}</div></div>
      <div class="stat"><div class="label">Errors</div><div class="value ${(chal.failed||0)>0?'red':'green'}">${chal.failed||0}</div></div>
    `;

    // Build combined miner data
    const miners = hRes.miners_detail || [];
    const scores = (sRes && sRes.miners) || [];
    const scoreMap = {};
    scores.forEach(s => { scoreMap[s.uid] = s; });

    // Track latency history
    if (miners.length > 0) {
      const avgTtft = miners.reduce((s,m) => s + (m.avg_ttft_ms||0), 0) / miners.length;
      history.ttft.push(Math.round(avgTtft));
    }

    // Chart history
    history.organic.push(hRes.total_organic||0);
    history.synthetic.push(hRes.total_synthetic||0);
    history.passed.push(chal.passed||0);
    history.failed.push(chal.failed||0);
    if (history.organic.length > MAX_HIST) {
      Object.values(history).forEach(a => { if (a.length > MAX_HIST) a.shift(); });
    }

    function diffs(arr){ return arr.map((v,i)=>i===0?0:v-arr[i-1]); }
    let labels = history.organic.map(()=>'');

    miniChart('chartReqs', [
      {data:diffs(history.organic), color:'#3fb950', label:'Organic', labels},
      {data:diffs(history.synthetic), color:'#bc8cff', label:'Synthetic', labels}
    ]);
    miniChart('chartChal', [
      {data:diffs(history.passed), color:'#3fb950', label:'Passed', labels},
      {data:diffs(history.failed), color:'#f85149', label:'Failed', labels}
    ]);
    miniChart('chartCumulative', [
      {data:history.organic, color:'#3fb950', label:'Organic', labels},
      {data:history.synthetic, color:'#bc8cff', label:'Synthetic', labels}
    ], {line:true});
    miniChart('chartLatency', [
      {data:history.ttft, color:'#58a6ff', label:'Avg TTFT (ms)', labels: history.ttft.map(()=>'')}
    ], {line:true});

    // Leaderboard tab
    const leaderData = miners.map(m => {
      const s = scoreMap[m.uid] || {};
      return { ...m, net_points: s.net_points||0, pass_rate: s.pass_rate||0, organic: s.organic||0, synthetic: s.synthetic||0 };
    }).sort((a,b) => b.net_points - a.net_points);
    const maxPts = Math.max(...leaderData.map(m=>m.net_points), 1);

    document.getElementById('leaderRows').innerHTML = leaderData.map((m,i) => `
      <tr>
        <td><span class="rank ${rankClass(i)}">${i+1}</span></td>
        <td>${esc(m.uid)}</td>
        <td class="blue" style="font-weight:700">${m.net_points.toFixed(1)}</td>
        <td style="min-width:120px"><div class="bar"><div class="bar-fill" style="width:${(m.net_points/maxPts*100).toFixed(1)}%;background:${i===0?'var(--gold)':i===1?'var(--silver)':i===2?'var(--bronze)':'var(--blue)'}"></div></div></td>
        <td>${m.organic + m.synthetic}</td>
        <td class="${m.pass_rate>=0.99?'green':m.pass_rate>=0.9?'orange':'red'}">${(m.pass_rate*100).toFixed(0)}%</td>
        <td class="green">${m.avg_tps.toFixed(1)}</td>
        <td>${(m.reliability*100).toFixed(1)}%</td>
        <td><span class="health ${m.alive?'up':'down'}"></span>${m.alive?'Online':'Offline'}</td>
      </tr>
    `).join('');

    // Miner detail table
    document.getElementById('minerRows').innerHTML = miners.map(m => `
      <tr>
        <td><span class="health ${m.alive?'up':'down'}"></span>${m.alive?'Alive':'Dead'}</td>
        <td>${esc(m.uid)}</td>
        <td style="font-size:12px;color:var(--dim)">${esc(m.endpoint||'')}</td>
        <td>${(m.reliability*100).toFixed(1)}%</td>
        <td>${esc(m.served)}</td>
        <td class="${m.failed>0?'red':''}">${esc(m.failed)}</td>
        <td>${m.avg_ttft_ms.toFixed(1)}</td>
        <td class="green">${m.avg_tps.toFixed(1)}</td>
        <td>${esc(m.active)}</td>
      </tr>
    `).join('');

    // Scoreboard table
    if (scores.length > 0) {
      document.getElementById('scoreRows').innerHTML = scores.map(m => `
        <tr>
          <td>${esc(m.uid)}</td>
          <td class="blue">${m.net_points.toFixed(1)}</td>
          <td>${esc(m.organic)}</td>
          <td>${esc(m.synthetic)}</td>
          <td class="${m.pass_rate>=0.99?'green':m.pass_rate>=0.9?'orange':'red'}">${(m.pass_rate*100).toFixed(0)}%</td>
          <td>${m.consistency.toFixed(3)}</td>
          <td>${m.divergence.toFixed(3)}</td>
          <td>${m.is_suspect?'<span class="badge err">YES</span>':'<span class="badge ok">NO</span>'}</td>
        </tr>
      `).join('');
    }

    // Epoch history tab
    if (Array.isArray(eRes) && eRes.length > 0) {
      document.getElementById('epochRows').innerHTML = eRes.slice().reverse().map(e => {
        const dur = e.duration_s ? (e.duration_s/60).toFixed(1)+'m' : '–';
        const chalTotal = (e.challenges_passed||0) + (e.challenges_failed||0);
        const ePassRate = chalTotal > 0 ? ((e.challenges_passed/chalTotal)*100).toFixed(0)+'%' : '–';
        return `<tr>
          <td class="blue">${e.epoch}</td>
          <td>${dur}</td>
          <td class="green">${e.organic||0}</td>
          <td class="purple">${e.synthetic||0}</td>
          <td>${chalTotal}</td>
          <td class="${parseFloat(ePassRate)>=99?'green':parseFloat(ePassRate)>=90?'orange':'red'}">${ePassRate}</td>
          <td>${e.miners_alive||'–'}/${e.miners_total||'–'}</td>
        </tr>`;
      }).join('');
    }

    document.getElementById('updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('updated').textContent = 'Error: ' + e.message;
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Hardened Inference Subnet Gateway")
    parser.add_argument("--miners", nargs="+", default=[], help="Miner endpoint URLs (static)")
    parser.add_argument("--discover", action="store_true", help="Enable metagraph-based miner discovery")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port")
    parser.add_argument("--epoch-length", type=int, default=60, help="Epoch length (seconds)")
    parser.add_argument("--synthetic-interval", type=float, default=8, help="Base synthetic interval (seconds)")
    parser.add_argument("--r2-dir", default="/tmp/r2-audit", help="Local R2 audit directory")
    parser.add_argument("--api-keys", default="", help="Comma-separated API keys (empty = no auth)")
    parser.add_argument("--monitoring-keys", default="", help="Comma-separated monitoring API keys (empty = open)")
    parser.add_argument("--relay-secret", default="", help="Shared secret for /internal/relay (auditor IP masking)")
    parser.add_argument("--miner-secret", default="", help="Shared secret for miner auth (X-Validator-Key header)")
    parser.add_argument("--model", default=None, help="HuggingFace model name for verification (default: mock)")
    parser.add_argument("--challenge-rate", type=float, default=None, help="Challenge rate 0.0-1.0 (default: 1.0 with model, 0.0 without)")
    # Chain integration (optional — omit --wallet to run without chain)
    parser.add_argument("--wallet", default=None, help="Bittensor wallet name (enables chain weight-setting)")
    parser.add_argument("--hotkey", default="default", help="Bittensor hotkey name")
    parser.add_argument("--netuid", type=int, default=1, help="Subnet UID")
    parser.add_argument("--network", default="finney", help="Bittensor network (finney/test/local/ws://...)")
    parser.add_argument("--wallet-path", default=None, help="Wallet directory path")
    parser.add_argument("--auditor-url", default=None, help="External audit_validator URL to fetch challenge data for routing")
    parser.add_argument("--max-context-tokens", type=int, default=32768, help="Max context window (prompt+output tokens) — prevents OOB errors on miners")
    args = parser.parse_args()

    if not args.miners and not args.discover:
        parser.error("Must specify --miners or --discover (or both)")

    asyncio.run(run_gateway(args))


if __name__ == "__main__":
    main()
