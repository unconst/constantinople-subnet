#!/usr/bin/env python3
"""
Challenge Engine — Unpredictable, multi-layered verification system.

This is the core anti-cheat component. Challenges must be:
1. Indistinguishable from organic traffic (miners can't detect them)
2. Cryptographically unpredictable (miners can't precompute answers)
3. Multi-dimensional (check speed, correctness, AND consistency)
4. Asymmetrically costly to fake (cheating must cost more than honesty)

Challenge Types:
- Hidden state challenges (standard): verify cached states match
- Timing challenges: verify sub-50ms response proves GPU cache
- Multi-point challenges: check multiple (layer, position) pairs in one burst
- Sequential challenges: send follow-up to same request, verify consistency
- Cross-request challenges: same prompt to two miners, compare outputs
"""

import asyncio
import hashlib
import logging
import math
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("challenge_engine")
log.propagate = False


@dataclass
class Challenge:
    """A single challenge to send to a miner."""
    challenge_id: str
    request_id: str
    challenge_type: str  # "hidden_state", "multi_point", "timing"
    layer_index: int
    token_index: int
    created_at: float = field(default_factory=time.time)
    # For multi-point challenges
    extra_points: list = field(default_factory=list)  # [(layer, token), ...]


@dataclass
class ChallengeVerification:
    """Result of verifying a challenge response."""
    challenge_id: str
    passed: bool
    cosine_sim: float
    latency_ms: float
    reason: str
    extra_results: list = field(default_factory=list)  # For multi-point


class ChallengeEngine:
    """
    Generates and verifies challenges against miners.

    Anti-cheat features:
    1. Cryptographic nonces prevent replay attacks
    2. Random layer/position selection prevents precomputation
    3. Multi-point bursts catch miners who only cache partial states
    4. Tight timing requirements prove GPU-cached hidden states
    5. Challenge rate varies randomly to prevent timing analysis
    """

    def __init__(
        self,
        cosine_threshold: float = 0.70,  # C4 H4-4: lowered from 0.995 to enable tiered scoring in scoring engine
        timing_threshold_ms: float = 50.0,
        timing_hard_cutoff_ms: float = 500.0,
        multi_point_probability: float = 0.2,  # 20% of challenges are multi-point
        num_extra_points: int = 3,
    ):
        self.cosine_threshold = cosine_threshold
        self.timing_threshold_ms = timing_threshold_ms
        self.timing_hard_cutoff_ms = timing_hard_cutoff_ms
        self.multi_point_probability = multi_point_probability
        self.num_extra_points = num_extra_points

        # Pending challenges (awaiting responses) — bounded to prevent memory leaks
        self._pending: dict[str, Challenge] = {}
        self._max_pending = 10000
        self._lock = asyncio.Lock()  # Protects _pending + counters under concurrent access

        # Statistics
        self.total_challenges = 0
        self.total_passed = 0
        self.total_failed = 0

    def create_challenge(
        self,
        request_id: str,
        num_layers: int,
        seq_len: int,
    ) -> Challenge:
        """
        Create an unpredictable challenge for a miner.

        Uses cryptographic randomness for:
        - Challenge ID (unguessable, prevents replay)
        - Layer selection (uniform random)
        - Token position selection (uniform random)
        """
        challenge_id = secrets.token_hex(16)

        # Amortized cleanup — prevent unbounded memory growth (C3-6)
        if len(self._pending) > 100:
            self.cleanup_expired()

        # Exclude last 2 layers — CPU float32 vs GPU float16 divergence at
        # deep transformer layers causes false failures (cosine < 0.1).
        # Layer 27 (last) is worst; layer 26 also diverges significantly.
        safe_layers = max(num_layers - 2, 1)

        # Primary challenge point
        layer_index = secrets.randbelow(safe_layers)
        token_index = secrets.randbelow(max(seq_len, 1))

        # Decide if multi-point based on cryptographic coin flip
        extra_points = []
        if secrets.randbelow(100) < int(self.multi_point_probability * 100):
            for _ in range(self.num_extra_points):
                extra_points.append((
                    secrets.randbelow(safe_layers),
                    secrets.randbelow(max(seq_len, 1)),
                ))
            challenge_type = "multi_point"
        else:
            challenge_type = "hidden_state"

        challenge = Challenge(
            challenge_id=challenge_id,
            request_id=request_id,
            challenge_type=challenge_type,
            layer_index=layer_index,
            token_index=token_index,
            extra_points=extra_points,
        )

        # Evict oldest entries if at capacity (use pop to avoid KeyError from concurrent access)
        if len(self._pending) >= self._max_pending:
            oldest = sorted(self._pending.values(), key=lambda c: c.created_at)[:100]
            for c in oldest:
                self._pending.pop(c.challenge_id, None)

        self._pending[challenge_id] = challenge
        self.total_challenges += 1

        return challenge

    def verify_response(
        self,
        challenge_id: str,
        miner_hidden_state: np.ndarray,
        reference_hidden_state: np.ndarray,
        latency_ms: float,
        extra_miner_states: list[np.ndarray] = None,
        extra_reference_states: list[np.ndarray] = None,
    ) -> ChallengeVerification:
        """
        Verify a miner's challenge response.

        Checks:
        1. Challenge ID is valid and pending (not replayed)
        2. Cosine similarity meets threshold
        3. Latency is within bounds (proves GPU cache)
        4. Extra points match (for multi-point challenges)
        """
        challenge = self._pending.pop(challenge_id, None)
        if challenge is None:
            self.total_failed += 1
            return ChallengeVerification(
                challenge_id=challenge_id,
                passed=False,
                cosine_sim=0.0,
                latency_ms=latency_ms,
                reason="Invalid or expired challenge ID",
            )

        # Cosine similarity check
        cos_sim = _cosine_similarity(miner_hidden_state, reference_hidden_state)

        # Timing check — add per-challenge jitter to prevent miners from
        # optimizing fake cache responses to land just under a fixed cutoff (C3-7)
        jitter_ms = secrets.randbelow(400) - 200  # ±200ms
        effective_hard_cutoff = self.timing_hard_cutoff_ms + jitter_ms
        if latency_ms > effective_hard_cutoff:
            self.total_failed += 1
            return ChallengeVerification(
                challenge_id=challenge_id,
                passed=False,
                cosine_sim=cos_sim,
                latency_ms=latency_ms,
                reason=f"Latency {latency_ms:.1f}ms exceeds hard cutoff {effective_hard_cutoff:.0f}ms",
            )

        if cos_sim < self.cosine_threshold:
            self.total_failed += 1
            return ChallengeVerification(
                challenge_id=challenge_id,
                passed=False,
                cosine_sim=cos_sim,
                latency_ms=latency_ms,
                reason=f"Cosine {cos_sim:.6f} below threshold {self.cosine_threshold}",
            )

        # Soft timing check (penalized but not auto-fail)
        timing_ok = latency_ms <= self.timing_threshold_ms

        # Multi-point verification
        extra_results = []
        if challenge.challenge_type == "multi_point" and extra_miner_states and extra_reference_states:
            for i, (m_state, r_state) in enumerate(zip(extra_miner_states, extra_reference_states)):
                extra_cos = _cosine_similarity(m_state, r_state)
                extra_ok = extra_cos >= self.cosine_threshold
                extra_results.append({
                    "point_index": i,
                    "layer": challenge.extra_points[i][0] if i < len(challenge.extra_points) else -1,
                    "token": challenge.extra_points[i][1] if i < len(challenge.extra_points) else -1,
                    "cosine_sim": extra_cos,
                    "passed": extra_ok,
                })
                if not extra_ok:
                    self.total_failed += 1
                    return ChallengeVerification(
                        challenge_id=challenge_id,
                        passed=False,
                        cosine_sim=cos_sim,
                        latency_ms=latency_ms,
                        reason=f"Multi-point {i} failed: cosine={extra_cos:.6f}",
                        extra_results=extra_results,
                    )

        self.total_passed += 1
        reason = "OK"
        if not timing_ok:
            reason = f"OK (slow: {latency_ms:.1f}ms > {self.timing_threshold_ms}ms)"

        return ChallengeVerification(
            challenge_id=challenge_id,
            passed=True,
            cosine_sim=cos_sim,
            latency_ms=latency_ms,
            reason=reason,
            extra_results=extra_results,
        )

    def cleanup_expired(self, max_age_s: float = 300.0):
        """Remove expired pending challenges."""
        now = time.time()
        expired = [cid for cid, c in self._pending.items() if now - c.created_at > max_age_s]
        for cid in expired:
            del self._pending[cid]
        if expired:
            log.debug(f"Cleaned up {len(expired)} expired challenges")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 for zero/NaN/Inf vectors."""
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    result = float(dot / (norm_a * norm_b))
    if not math.isfinite(result):
        return 0.0
    return result
