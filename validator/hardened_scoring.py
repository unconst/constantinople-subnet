#!/usr/bin/env python3
"""
Hardened Scoring Engine — Exploit-resistant miner evaluation.

Attack vectors defended against:
1. Selective honesty (organic vs synthetic performance divergence)
2. Score manipulation via edge cases in formulas
3. Latency gaming (fast on challenges, slow on organic)
4. Collusion detection via statistical correlation
5. Sybil resistance via diminishing returns per hotkey cluster
6. Three-state challenge tracking (pass/fail/unchallenged) to prevent score inflation
7. Minimum sample requirements to prevent early-exit gaming

Design principle: Every formula is designed so that the optimal strategy
for a rational miner is to serve fast, honest inference.
"""

import asyncio
import logging
import math
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("hardened_scoring")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(_handler)
log.propagate = False


# ── Constants ────────────────────────────────────────────────────────────────

# Scoring weights
SPEED_WEIGHT = 0.40       # 40% for speed (TTFT + throughput)
VERIFICATION_WEIGHT = 0.40  # 40% for hidden state verification
CONSISTENCY_WEIGHT = 0.20  # 20% for consistency across requests

# Divergence detection
DIVERGENCE_THRESHOLD = 0.12       # 12% gap triggers investigation
DIVERGENCE_PENALTY_MILD = 0.30    # -30% weight for mild divergence
DIVERGENCE_PENALTY_SEVERE = 0.70  # -70% weight for severe divergence (>25%)
MIN_ORGANIC_SAMPLES = 3           # Need this many organic scores (lowered from 5 to close C2-5 evasion)
MIN_SYNTHETIC_SAMPLES = 5         # Need this many synthetic scores

# Challenge verification — relaxed to allow optimization
# Tiered cosine: speculative decoding/quantization produce lower but directionally
# correct hidden states. We reward fidelity on a gradient rather than a hard cliff.
COSINE_THRESHOLD = 0.70           # Minimum to pass (cheaters produce < 0.3)
COSINE_FULL_CREDIT = 0.99         # Standard inference (bit-exact)
COSINE_HIGH_CREDIT = 0.90         # Quantized / TP-sharded inference
# Timing thresholds — these apply to RTT-corrected extraction time when available.
# The audit_validator subtracts network RTT baseline from raw latency before scoring.
# VRAM cache lookup: <5ms. CPU HF extraction: 50-500ms. Full re-inference: 2-10s.
CHALLENGE_TIMEOUT_MS = 50         # Ideal: proves VRAM cache (soft limit, full credit)
CHALLENGE_TIMEOUT_HARD_MS = 2000  # Hard cutoff — auto-fail above this (raised from 500ms
                                  # to accommodate RTT-corrected deep-layer CPU extraction)
REINFERENCE_THRESHOLD_MS = 1000   # Above this, strong evidence of re-inference from scratch

# Speed scoring (relative to population)
TTFT_EXCELLENT_MS = 30
TTFT_POOR_MS = 500
TPS_EXCELLENT = 150
TPS_POOR = 10

# Throughput incentive — absolute TPS bonus multiplier on weight.
# Purpose: incentivize miners to optimize for raw throughput (speculative decoding,
# better batching, faster hardware) beyond just being "fast enough" relative to fleet.
# Miners above TPS_BONUS_THRESHOLD get a multiplicative weight bonus up to TPS_BONUS_MAX.
# This makes it rational to invest in speculative decoding, continuous batching, etc.
TPS_BONUS_THRESHOLD = 50   # TPS above which bonus kicks in
TPS_BONUS_CEILING = 200    # TPS at which max bonus is reached
TPS_BONUS_MAX = 2.5        # Maximum bonus multiplier (150% more weight) — speed is king

# Anti-gaming
MAX_POINTS_PER_REQUEST = 1.0      # Cap to prevent any single request from dominating
MIN_REQUESTS_FOR_WEIGHT = 1       # Minimum requests to receive any weight (hard floor)
FULL_WEIGHT_REQUESTS = 3          # Requests at which miner gets full weight (gradual ramp below this)
CHALLENGE_FAIL_STRIKE_MULTIPLIER = 3.0  # Failing costs 3x what passing earns
MAX_CONSECUTIVE_FAILS = 3         # 3 consecutive fails → miner marked suspect

# Rate limiting
MAX_REQUESTS_PER_MINER_PER_EPOCH = 10000  # Prevent flooding

# Availability incentive — rewards miners that serve more requests per epoch.
# Uses a log-ratio: miners serving 2x more requests than the median get a small
# boost. This makes uptime and reliability matter for weight, not just speed.
# A miner online 100% of the time will naturally serve ~2x a miner at 50%.
AVAILABILITY_BONUS_MAX = 1.3     # Up to 30% bonus for high-availability miners
AVAILABILITY_BONUS_FLOOR = 0.5   # Minimum multiplier for very low availability

# Concurrency bonus — rewards miners that maintain throughput under concurrent load.
# Measured via periodic bandwidth probes (4 simultaneous requests).
# A miner with good batching/scheduling maintains ~80%+ of single-request TPS.
# This directly incentivizes continuous batching, PagedAttention, etc.
CONCURRENCY_BONUS_THRESHOLD = 0.7  # Concurrent/single TPS ratio above which bonus kicks in
CONCURRENCY_BONUS_MAX = 1.25       # Up to 25% bonus for excellent concurrency handling

# Epoch
DEFAULT_EPOCH_LENGTH_S = 4320     # ~72 minutes (360 blocks * 12s)

# Adaptive challenge rate (validator cost scaling)
BASE_CHALLENGE_RATE = 0.3         # Default per-miner challenge rate
NEW_MINER_CHALLENGE_RATE = 0.9    # Rate for miners with < MIN_CHALLENGES_FOR_TRUST challenges
MIN_CHALLENGES_FOR_TRUST = 10     # Need this many passes before rate decays
SUSPECT_CHALLENGE_RATE = 1.0      # Always challenge suspects
POST_FAIL_BOOST_COUNT = 5         # Challenge at 100% for this many requests after a fail
CHALLENGE_RATE_FLOOR = 0.10       # Never go below 10% even for highly trusted miners
MIN_CHALLENGES_PER_EPOCH = 5      # Every miner must face at least this many challenges per epoch


class RollingMeanTracker:
    """Exponentially-weighted rolling mean with infinite memory and decay.

    Replaces fixed-size deques for cross-epoch divergence tracking (C3-10).
    Old samples fade with exponential decay but never disappear entirely,
    preventing rotation attacks where miners flush honest data by waiting.
    """
    __slots__ = ("mean", "count", "alpha")

    def __init__(self, alpha: float = 0.05):
        self.mean = 0.0
        self.count = 0
        self.alpha = alpha

    def update(self, value: float):
        self.count += 1
        if self.count == 1:
            self.mean = value
        else:
            self.mean = (1 - self.alpha) * self.mean + self.alpha * value


@dataclass
class ChallengeResult:
    """Result of a single hidden state challenge."""
    request_id: str
    layer_index: int
    token_index: int
    cosine_sim: float
    latency_ms: float
    passed: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RequestScore:
    """Score for a single request."""
    request_id: str
    miner_uid: int
    timestamp: float
    is_synthetic: bool
    speed_score: float
    verification_score: float
    quality_score: float
    ttft_ms: float
    tokens_per_sec: float
    cosine_sim: float
    challenge_latency_ms: float
    challenge_passed: bool = None  # None = no challenge performed, True/False = challenge result

    @property
    def points(self) -> float:
        raw = self.speed_score * self.verification_score * self.quality_score
        if not math.isfinite(raw):
            return 0.0
        return min(raw, MAX_POINTS_PER_REQUEST)


def compute_verification_score(challenge_passed, cosine_sim: float,
                                challenge_latency_ms: float) -> float:
    """
    Tiered verification score that rewards optimization while catching cheaters.

    Cosine tiers (optimization-friendly):
    - >= 0.99: full credit (1.0)  — standard float16/float32 inference
    - >= 0.90: high credit (0.9)  — quantized (INT8/FP8) or TP-sharded
    - >= 0.70: base credit (0.7)  — speculative decoding, aggressive optimization
    - <  0.70: fail (0.0)         — cheater / completely wrong model

    This allows miners to use speculative decoding, quantization, tensor
    parallelism, and KV compression without being penalized to zero, while
    still catching miners running the wrong model (cosine < 0.3 typically).

    Unchallenged requests get 0.3 (well below the worst passing score of 0.7),
    incentivizing challenge participation. RED-TEAM FIX (step 13): Reduced from
    0.5 to 0.3 — at 0.5, miners could serve 2 unchallenged for every 1 challenged
    and still earn net positive points even when failing challenges.
    """
    if challenge_passed is None:
        return 0.3
    if challenge_passed is False:
        return 0.0

    # Tiered cosine scoring
    if cosine_sim >= COSINE_FULL_CREDIT:
        # Near-exact match: linear gradient from 0.95 to 1.0
        cos_factor = 0.95 + 0.05 * min(1.0, (cosine_sim - COSINE_FULL_CREDIT) / max(1.0 - COSINE_FULL_CREDIT, 1e-9))
    elif cosine_sim >= COSINE_HIGH_CREDIT:
        # Quantized/TP: gradient from 0.85 to 0.95
        cos_factor = 0.85 + 0.10 * (cosine_sim - COSINE_HIGH_CREDIT) / max(COSINE_FULL_CREDIT - COSINE_HIGH_CREDIT, 1e-9)
    elif cosine_sim >= COSINE_THRESHOLD:
        # Speculative/aggressive opt: gradient from 0.70 to 0.85
        cos_factor = 0.70 + 0.15 * (cosine_sim - COSINE_THRESHOLD) / max(COSINE_HIGH_CREDIT - COSINE_THRESHOLD, 1e-9)
    else:
        return 0.0  # Below threshold — fail

    # Timing defense: latency_ms should be RTT-corrected (net extraction time) when
    # the audit_validator has enough RTT baseline data. This makes the timing check
    # meaningful even over WAN — it measures actual server-side work, not network latency.
    #
    # <50ms → 1.0 (VRAM cache proven — extraction from cached KV states)
    # 50-200ms → decay to 0.90 (slow cache or CPU-side extraction, acceptable)
    # 200-1000ms → decay to 0.70 (suspicious — might be partial re-computation)
    # >1000ms → 0.50 (strong evidence of full re-inference, Qwen 7B takes 2-10s)
    # >2000ms → auto-fail in challenge_engine (CHALLENGE_TIMEOUT_HARD_MS)
    if challenge_latency_ms <= CHALLENGE_TIMEOUT_MS:
        latency_factor = 1.0
    elif challenge_latency_ms <= 200:
        # Slow but plausible cache extraction (CPU HF model, large sequence)
        latency_factor = 1.0 - 0.10 * (
            (challenge_latency_ms - CHALLENGE_TIMEOUT_MS) / (200 - CHALLENGE_TIMEOUT_MS)
        )
    elif challenge_latency_ms <= REINFERENCE_THRESHOLD_MS:
        # Suspicious range — possible partial re-computation
        latency_factor = 0.90 - 0.20 * (
            (challenge_latency_ms - 200) / (REINFERENCE_THRESHOLD_MS - 200)
        )
    else:
        # >1000ms: strong evidence of re-inference from scratch
        latency_factor = 0.50

    return cos_factor * latency_factor


@dataclass
class MinerEpochStats:
    """Hardened per-miner stats for an epoch."""
    uid: int
    total_points: float = 0.0
    penalty_points: float = 0.0  # Points deducted for failures
    organic_scores: list = field(default_factory=list)
    synthetic_scores: list = field(default_factory=list)
    organic_latencies: list = field(default_factory=list)
    synthetic_latencies: list = field(default_factory=list)
    challenge_results: list = field(default_factory=list)
    total_requests: int = 0
    passed_challenges: int = 0
    failed_challenges: int = 0
    voided_challenges: int = 0  # C4 H4-1: Track voids for penalty system
    consecutive_fails: int = 0
    is_suspect: bool = False
    ttft_values: list = field(default_factory=list)
    tps_values: list = field(default_factory=list)
    cosine_values: list = field(default_factory=list)
    concurrent_tps_ratios: list = field(default_factory=list)  # TPS ratio under concurrent load

    @property
    def net_points(self) -> float:
        """Points after deducting penalties. Can go negative — miners with
        negative net_points are excluded from weight calculation entirely."""
        return self.total_points - self.penalty_points

    # C5 H5-4: Count voided challenges in recent_pass_rate denominator
    voided_results: int = 0

    @property
    def recent_pass_rate(self) -> float:
        """Pass rate over the most recent challenges only (last 20).
        C5 H5-4: Includes voided_results in denominator so high-void miners
        can't show artificially high recent_pass_rate."""
        if not self.challenge_results:
            return self.pass_rate  # Fall back to overall
        recent = self.challenge_results[-20:]
        passed = sum(1 for r in recent if r.passed)
        # Add capped void count to denominator — voids dilute pass rate
        total_with_voids = len(recent) + min(self.voided_results, 20)
        return passed / max(total_with_voids, 1)

    @property
    def organic_mean(self) -> float:
        return sum(self.organic_scores) / len(self.organic_scores) if self.organic_scores else 0.0

    @property
    def synthetic_mean(self) -> float:
        return sum(self.synthetic_scores) / len(self.synthetic_scores) if self.synthetic_scores else 0.0

    @property
    def organic_std(self) -> float:
        if len(self.organic_scores) < 2:
            return 0.0
        return float(np.std(self.organic_scores))

    @property
    def synthetic_std(self) -> float:
        if len(self.synthetic_scores) < 2:
            return 0.0
        return float(np.std(self.synthetic_scores))

    @property
    def divergence(self) -> float:
        """
        Robust divergence metric. Uses both mean and distribution comparison.
        Returns 0 if insufficient data (favors the miner — innocent until proven guilty).
        """
        if len(self.organic_scores) < MIN_ORGANIC_SAMPLES or len(self.synthetic_scores) < MIN_SYNTHETIC_SAMPLES:
            return 0.0

        org_mean = self.organic_mean
        syn_mean = self.synthetic_mean

        if syn_mean == 0 and org_mean == 0:
            return 0.0
        if syn_mean == 0:
            return 1.0  # All synthetic scores are 0 but organic aren't — suspicious

        # Mean-based divergence
        mean_div = abs(org_mean - syn_mean) / max(syn_mean, 0.001)

        # Also check latency divergence (miners gaming speed on synthetics)
        latency_div = 0.0
        if self.organic_latencies and self.synthetic_latencies:
            org_lat = np.median(self.organic_latencies)
            syn_lat = np.median(self.synthetic_latencies)
            if syn_lat > 0:
                latency_div = abs(org_lat - syn_lat) / max(syn_lat, 1.0)

        # Combined divergence: max of mean and latency divergence
        return max(mean_div, latency_div)

    @property
    def pass_rate(self) -> float:
        total = self.passed_challenges + self.failed_challenges
        return self.passed_challenges / max(total, 1)

    @property
    def avg_ttft_ms(self) -> float:
        return sum(self.ttft_values) / len(self.ttft_values) if self.ttft_values else 0.0

    @property
    def avg_tps(self) -> float:
        return sum(self.tps_values) / len(self.tps_values) if self.tps_values else 0.0

    @property
    def avg_cosine(self) -> float:
        return sum(self.cosine_values) / len(self.cosine_values) if self.cosine_values else 0.0

    @property
    def consistency_score(self) -> float:
        """
        Measure how consistent the miner's performance is.
        Inconsistent performance suggests gaming or unreliable service.
        Suspiciously LOW variance also penalized with a GRADIENT —
        real GPU inference always has natural jitter (cv typically 0.05-0.15).

        Gradient low-variance penalty prevents miners from injecting just
        enough jitter (cv≈0.03) to clear a binary threshold while still
        being suspiciously uniform.

        Returns 0-1 where 1 = consistent with natural variance.
        """
        if len(self.tps_values) < 3:
            return 0.5  # Not enough data, neutral
        cv = np.std(self.tps_values) / max(np.mean(self.tps_values), 0.001)
        # High variance penalty: cv=0 → 1.0, cv≥1 → 0.0
        high_var_score = max(0.0, 1.0 - cv)
        # Gradient low-variance penalty: real GPU inference has CV typically
        # in the 0.05-0.15 range. Suspiciously low CV is penalized on a
        # gradient so miners can't circumvent with minimal jitter injection.
        #   cv=0.00 → 0.3 (heavily penalized)
        #   cv=0.01 → 0.44 (still suspicious)
        #   cv=0.03 → 0.72 (marginal)
        #   cv≥0.05 → 1.0 (no low-variance penalty)
        LOW_CV_FLOOR = 0.07  # Below this, apply gradient penalty (raised from 0.05 to close C2-6 gaming)
        if cv < LOW_CV_FLOOR:
            low_var_factor = 0.3 + 0.7 * (cv / LOW_CV_FLOOR)
        else:
            low_var_factor = 1.0
        return high_var_score * low_var_factor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
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


def compute_output_quality(text: str, expected_min_tokens: int = 4) -> float:
    """
    Score output quality. Detects garbage, repetition, and degenerate outputs.

    Returns 0.0-1.0 where 1.0 = high quality output.

    Checks:
    1. Non-empty and meets minimum length
    2. Repetition ratio (repeated n-grams signal degenerate decoding)
    3. Character entropy (gibberish has abnormal entropy)
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    if len(words) < expected_min_tokens:
        # Very short output — scale linearly from 0.05 to threshold
        # 1 word = ~0.05, 2 words = ~0.25 for expected_min=4
        return max(0.05, len(words) / (expected_min_tokens * 2))

    # --- Repetition detection ---
    # Check trigram repetition: what fraction of trigrams are unique?
    if len(words) >= 6:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        # unique_ratio < 0.3 → heavy repetition (looping)
        if unique_ratio < 0.2:
            return 0.1  # Severely repetitive
        repetition_score = min(1.0, unique_ratio / 0.5)  # 0.5+ unique → 1.0
    else:
        repetition_score = 1.0

    # --- Character entropy ---
    # Normal English text: ~4.0-4.5 bits/char. Gibberish or single-char spam diverges.
    char_counts = {}
    clean = text.strip().lower()
    for c in clean:
        char_counts[c] = char_counts.get(c, 0) + 1
    total = len(clean)
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Very low entropy (< 2.0) → single char repeated or very uniform
    # Very high entropy (> 6.0) → random bytes / base64 etc
    if entropy < 1.5:
        entropy_score = 0.2
    elif entropy < 2.5:
        entropy_score = 0.6
    elif entropy > 6.5:
        entropy_score = 0.5
    else:
        entropy_score = 1.0

    return repetition_score * entropy_score


def compute_speed_score(ttft_ms: float, tokens_per_sec: float,
                        population_ttft: list[float] = None,
                        population_tps: list[float] = None,
                        miner_medians_ttft: list[float] = None,
                        miner_medians_tps: list[float] = None) -> float:
    """
    Score miner speed. Uses population-relative scoring when available,
    falls back to absolute thresholds.

    Sybil-resistant: when miner_medians_* are provided, ranks against
    per-miner median values instead of raw request pools. This prevents
    Sybil miners from flooding the population with fabricated values to
    shift the distribution. Each miner UID contributes exactly one data
    point regardless of request volume.
    """
    # Prefer per-miner medians (Sybil-resistant) over raw request pool
    pool_ttft = miner_medians_ttft if miner_medians_ttft and len(miner_medians_ttft) >= 3 else population_ttft
    pool_tps = miner_medians_tps if miner_medians_tps and len(miner_medians_tps) >= 3 else population_tps

    if pool_ttft and len(pool_ttft) >= 3:
        ttft_rank = sum(1 for t in pool_ttft if t >= ttft_ms) / len(pool_ttft)
        tps_rank = sum(1 for t in pool_tps if t <= tokens_per_sec) / len(pool_tps)
    else:
        # Absolute fallback
        ttft_rank = max(0.0, min(1.0, 1.0 - (ttft_ms - TTFT_EXCELLENT_MS) / (TTFT_POOR_MS - TTFT_EXCELLENT_MS)))
        tps_rank = max(0.0, min(1.0, (tokens_per_sec - TPS_POOR) / (TPS_EXCELLENT - TPS_POOR)))

    return 0.4 * ttft_rank + 0.6 * tps_rank


class HardenedScoringEngine:
    """
    Exploit-resistant scoring engine.

    Key hardening features:
    1. Asymmetric penalties: failing a challenge costs more than passing earns
    2. Consecutive failure tracking with suspect flagging
    3. Population-relative speed scoring (can't game absolute thresholds)
    4. Dual divergence detection (score + latency)
    5. Minimum sample requirements
    6. Net points (gross - penalties) prevents "win some lose some" strategies
    7. Request rate limiting per miner
    8. Consistency scoring rewards reliable service
    """

    # Max samples to keep for population-relative scoring per epoch
    _MAX_POPULATION_SAMPLES = 5000
    # Max request log entries per epoch (older entries discarded)
    _MAX_REQUEST_LOG = 2000
    # Max weight history entries to retain
    _MAX_WEIGHT_HISTORY = 100

    def __init__(self, epoch_length_s: float = DEFAULT_EPOCH_LENGTH_S):
        self.epoch_length_s = epoch_length_s
        self.current_epoch_start = time.time()
        self._current_epoch_target_s = self._randomize_epoch_length()
        self.epoch_number = 0
        self.miner_stats: dict[int, MinerEpochStats] = {}
        self.request_log: deque[RequestScore] = deque(maxlen=self._MAX_REQUEST_LOG)
        self.weight_history: deque[dict] = deque(maxlen=self._MAX_WEIGHT_HISTORY)
        self._epoch_lock = threading.Lock()  # Protects miner_stats during epoch rollover

        # Population-level metrics for relative scoring (bounded)
        self._population_ttft: deque[float] = deque(maxlen=self._MAX_POPULATION_SAMPLES)
        self._population_tps: deque[float] = deque(maxlen=self._MAX_POPULATION_SAMPLES)

        # Cross-epoch suspect tracking: uid → number of epochs flagged suspect
        self._suspect_history: dict[int, int] = {}

        # Hotkey-based suspect tracking: survives UID re-registrations.
        # hotkey → suspect count. When a miner registers with a known hotkey,
        # its suspect history carries over even if the UID changes.
        self._hotkey_suspect_history: dict[str, int] = {}

        # UID ↔ hotkey mapping for current epoch
        self._uid_to_hotkey: dict[int, str] = {}

        # Cross-epoch divergence tracking: accumulates organic/synthetic score samples
        # across epochs so miners can't reset detection by staying under per-epoch minimums.
        # Uses rolling mean trackers instead of fixed-size deques (C3-10) so old data
        # decays exponentially but never disappears — prevents rotation attacks.
        # uid → {"organic": RollingMeanTracker, "synthetic": RollingMeanTracker}
        self._cross_epoch_scores: dict[int, dict[str, RollingMeanTracker]] = {}

        # Adaptive challenge rate tracking (cross-epoch, survives epoch resets)
        # uid → consecutive clean challenges (resets on any failure)
        self._clean_streak: dict[int, int] = {}
        # uid → remaining boosted challenges after a failure
        self._post_fail_boost: dict[int, int] = {}
        # uid → challenges issued this epoch (resets each epoch)
        self._epoch_challenge_count: dict[int, int] = {}
        # uid → requests seen this epoch (resets each epoch)
        self._epoch_request_count: dict[int, int] = {}

        # Cache miss rates (set by auditor) — uid → (misses, total_attempts)
        self._cache_miss_rates: dict[int, tuple[int, int]] = {}

        # Log-spam guards: only warn about divergence once per miner per epoch
        self._divergence_warned: set[int] = set()
        self._cross_div_warned: set[int] = set()

    def set_cache_miss_rate(self, uid: int, misses: int, total: int):
        """Called by auditor to report cache miss rates for weight penalty."""
        self._cache_miss_rates[uid] = (misses, total)

    def record_bandwidth_probe(self, uid: int, concurrent_tps: float, baseline_tps: float):
        """Record a bandwidth probe result. Called by audit_validator after burst test.

        concurrent_tps: aggregate TPS under concurrent load (4 simultaneous requests)
        baseline_tps: this miner's single-request median TPS
        """
        if baseline_tps <= 0:
            return
        ratio = min(1.5, concurrent_tps / baseline_tps)  # Cap at 1.5 to prevent inflation
        stats = self._get_stats(uid)
        stats.concurrent_tps_ratios.append(ratio)

    def _get_stats(self, uid: int) -> MinerEpochStats:
        if uid not in self.miner_stats:
            self.miner_stats[uid] = MinerEpochStats(uid=uid)
        return self.miner_stats[uid]

    def register_hotkey(self, uid: int, hotkey: str):
        """Register a UID↔hotkey mapping. Must be called when miners are discovered.
        Transfers any hotkey-based suspect history to the UID-based tracker so
        re-registered miners inherit their penalty history."""
        self._uid_to_hotkey[uid] = hotkey
        if hotkey in self._hotkey_suspect_history:
            # Transfer hotkey history to UID history — re-registering doesn't reset
            existing_uid_history = self._suspect_history.get(uid, 0)
            hotkey_history = self._hotkey_suspect_history[hotkey]
            self._suspect_history[uid] = max(existing_uid_history, hotkey_history)

    def get_challenge_rate(self, uid: int) -> float:
        """
        Adaptive per-miner challenge rate for validator cost scaling.

        The idea: trusted miners with long clean streaks get challenged less,
        freeing validator GPU budget for new/suspicious miners. Total validator
        compute stays roughly constant as the fleet grows.

        Rates:
        - Suspect miners: 100% (always challenge)
        - Post-failure boost: 100% for POST_FAIL_BOOST_COUNT challenges
        - New miners (< MIN_CHALLENGES_FOR_TRUST passes): 90%
        - Trusted miners: base_rate / sqrt(clean_streak / MIN_CHALLENGES_FOR_TRUST)
          decaying from 30% down to floor of 10%
        - Epoch minimum enforcement: if a miner has had fewer than
          MIN_CHALLENGES_PER_EPOCH challenges this epoch, rate is boosted to
          ensure the minimum is met.

        Returns a float in [CHALLENGE_RATE_FLOOR, 1.0].
        """
        # Suspect miners always get challenged
        if uid in self._suspect_history and self._suspect_history[uid] > 0:
            return SUSPECT_CHALLENGE_RATE
        stats = self.miner_stats.get(uid)
        if stats and stats.is_suspect:
            return SUSPECT_CHALLENGE_RATE

        # Post-failure boost: 100% for several challenges after a failure
        if self._post_fail_boost.get(uid, 0) > 0:
            return SUSPECT_CHALLENGE_RATE

        # New miners: high rate until they build a track record
        clean = self._clean_streak.get(uid, 0)
        if clean < MIN_CHALLENGES_FOR_TRUST:
            return NEW_MINER_CHALLENGE_RATE

        # Trusted miners: decaying rate based on clean streak
        # rate = base / sqrt(streak / min_trust) — e.g.:
        #   10 passes → 0.30, 40 passes → 0.15, 90 passes → 0.10
        rate = BASE_CHALLENGE_RATE / math.sqrt(clean / MIN_CHALLENGES_FOR_TRUST)
        rate = max(CHALLENGE_RATE_FLOOR, min(1.0, rate))

        # Epoch minimum enforcement: if we haven't hit the per-epoch minimum
        # for this miner, boost the rate to catch up. This prevents a miner
        # from exploiting a low adaptive rate to avoid most challenges.
        epoch_challenges = self._epoch_challenge_count.get(uid, 0)
        epoch_requests = self._epoch_request_count.get(uid, 0)
        if epoch_challenges < MIN_CHALLENGES_PER_EPOCH and epoch_requests > 0:
            # How many requests remain (estimate based on epoch progress)?
            # Conservative: assume at least 20 more requests will come.
            remaining_est = max(20, epoch_requests)
            needed = MIN_CHALLENGES_PER_EPOCH - epoch_challenges
            boost_rate = needed / remaining_est
            rate = max(rate, boost_rate)

        return rate

    def record_challenge_outcome(self, uid: int, passed: bool, voided: bool = False):
        """
        Update cross-epoch clean streak tracking for adaptive challenge rates.
        Call this after every challenge (pass, fail, or void).
        C5 H5-4: Voided challenges increment voided_results counter.
        """
        if voided:
            stats = self._get_stats(uid)
            stats.voided_results += 1
            return  # Voids don't affect clean_streak
        self._epoch_challenge_count[uid] = self._epoch_challenge_count.get(uid, 0) + 1
        if passed:
            self._clean_streak[uid] = self._clean_streak.get(uid, 0) + 1
            # Decrement post-fail boost counter
            if uid in self._post_fail_boost and self._post_fail_boost[uid] > 0:
                self._post_fail_boost[uid] -= 1
                if self._post_fail_boost[uid] == 0:
                    del self._post_fail_boost[uid]
        else:
            # Failure resets clean streak and activates boost
            self._clean_streak[uid] = 0
            self._post_fail_boost[uid] = POST_FAIL_BOOST_COUNT

    def record_request_seen(self, uid: int):
        """Track requests per miner per epoch for minimum challenge enforcement."""
        self._epoch_request_count[uid] = self._epoch_request_count.get(uid, 0) + 1

    def get_all_challenge_rates(self) -> dict[int, float]:
        """Get challenge rates for all known miners (for monitoring/logging)."""
        rates = {}
        # Include all miners we've ever seen
        all_uids = set(self.miner_stats.keys()) | set(self._clean_streak.keys())
        for uid in all_uids:
            rates[uid] = self.get_challenge_rate(uid)
        return rates

    def _randomize_epoch_length(self) -> float:
        """Add ±20% jitter to epoch length so miners cannot predict boundaries."""
        jitter_range = int(self.epoch_length_s * 0.2)
        if jitter_range < 1:
            return self.epoch_length_s
        offset = secrets.randbelow(2 * jitter_range + 1) - jitter_range
        return self.epoch_length_s + offset

    def get_miner_medians(self) -> tuple[list[float], list[float]]:
        """
        Compute per-miner median TTFT and TPS for Sybil-resistant population ranking.

        Each miner UID contributes exactly one data point (its median), regardless
        of how many requests it has served. This prevents Sybil miners from flooding
        the population pool with fabricated values to shift the distribution.
        """
        medians_ttft = []
        medians_tps = []
        for uid, stats in self.miner_stats.items():
            if stats.ttft_values:
                medians_ttft.append(float(np.median(stats.ttft_values)))
            if stats.tps_values:
                medians_tps.append(float(np.median(stats.tps_values)))
        return medians_ttft, medians_tps

    def record_request(self, score: RequestScore):
        """Record a scored request with hardened accounting."""
        with self._epoch_lock:
            self._record_request_inner(score)

    def _record_request_inner(self, score: RequestScore):
        """Inner implementation — must be called with _epoch_lock held."""
        stats = self._get_stats(score.miner_uid)

        # Rate limiting
        if stats.total_requests >= MAX_REQUESTS_PER_MINER_PER_EPOCH:
            log.warning(f"Miner {score.miner_uid}: rate limited (>{MAX_REQUESTS_PER_MINER_PER_EPOCH} requests)")
            return

        # Record speed_score for divergence comparison (NOT points, which includes
        # verification_score that differs structurally between organic=0.3 and
        # synthetic≥0.7, causing false-positive divergence on honest miners).
        if score.is_synthetic:
            stats.synthetic_scores.append(score.speed_score)
            stats.synthetic_latencies.append(score.ttft_ms)
        else:
            stats.organic_scores.append(score.speed_score)
            stats.organic_latencies.append(score.ttft_ms)

        # Challenge tracking — three states:
        #   None  = no challenge performed (cross-probes, unchallenged requests)
        #   True  = challenge passed
        #   False = challenge failed
        if score.challenge_passed is True:
            stats.passed_challenges += 1
            stats.consecutive_fails = 0
            stats.total_points += score.points
            stats.challenge_results.append(ChallengeResult(
                request_id=score.request_id, layer_index=0, token_index=0,
                cosine_sim=score.cosine_sim, latency_ms=score.challenge_latency_ms,
                passed=True, reason="pass",
            ))
        elif score.challenge_passed is False:
            stats.failed_challenges += 1
            stats.consecutive_fails += 1
            stats.challenge_results.append(ChallengeResult(
                request_id=score.request_id, layer_index=0, token_index=0,
                cosine_sim=score.cosine_sim, latency_ms=score.challenge_latency_ms,
                passed=False, reason="fail",
            ))
            # Asymmetric penalty: failing costs MORE than passing earns.
            # Use speed_score as the base (NOT quality_score — garbage output miners
            # shouldn't get LOWER penalties than honest miners).
            # Floor of 0.3 ensures even slow/garbage miners face meaningful penalties.
            penalty_base = max(score.speed_score, 0.3)
            stats.penalty_points += penalty_base * CHALLENGE_FAIL_STRIKE_MULTIPLIER

            if stats.consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                stats.is_suspect = True
                log.warning(f"Miner {score.miner_uid}: SUSPECT — {stats.consecutive_fails} consecutive challenge failures")
        else:
            # No challenge performed — add points at reduced rate (0.3 verification)
            # RED-TEAM FIX (step 13): Points still accrue but verification_score=0.3
            # makes unchallenged requests worth less than even the worst passing challenge (0.7)
            stats.total_points += score.points

        stats.total_requests += 1
        stats.ttft_values.append(score.ttft_ms)
        stats.tps_values.append(score.tokens_per_sec)
        # Only append cosine values when a challenge was actually performed
        # to prevent 0.0 from failed/unchallenged requests diluting avg_cosine
        if score.challenge_passed is not None and score.cosine_sim > 0.0:
            stats.cosine_values.append(score.cosine_sim)

        # Update population metrics
        self._population_ttft.append(score.ttft_ms)
        self._population_tps.append(score.tokens_per_sec)

        self.request_log.append(score)

        # Feed cross-epoch divergence tracker (rolling mean, C3-10)
        uid = score.miner_uid
        if uid not in self._cross_epoch_scores:
            self._cross_epoch_scores[uid] = {
                "organic": RollingMeanTracker(alpha=0.05),
                "synthetic": RollingMeanTracker(alpha=0.05),
            }
        bucket = "synthetic" if score.is_synthetic else "organic"
        self._cross_epoch_scores[uid][bucket].update(score.speed_score)

    def record_challenge(self, miner_uid: int, result: ChallengeResult):
        """Record a standalone challenge result."""
        stats = self._get_stats(miner_uid)
        stats.challenge_results.append(result)

    def should_end_epoch(self) -> bool:
        return time.time() - self.current_epoch_start >= self._current_epoch_target_s

    def compute_weights(self) -> dict[int, float]:
        """Public API — compute weights from current live miner_stats."""
        return self._compute_weights_from(self.miner_stats)

    def _compute_weights_from(self, stats_dict: dict) -> dict[int, float]:
        """
        Compute final weights with layered anti-gaming defenses.

        Weight = net_points × consistency × divergence_factor × suspect_factor
        All normalized to sum to 1.0.

        Accepts an explicit stats_dict so end_epoch can pass a snapshot
        without racing with concurrent record_request() calls.
        """
        raw_weights: dict[int, float] = {}

        for uid, stats in stats_dict.items():
            # Minimum request threshold (hard floor)
            if stats.total_requests < MIN_REQUESTS_FOR_WEIGHT:
                log.debug(f"Miner {uid}: below minimum requests ({stats.total_requests}/{MIN_REQUESTS_FOR_WEIGHT})")
                continue

            # Gradual ramp: miners with few requests get partial weight.
            # At MIN_REQUESTS_FOR_WEIGHT requests → 40% weight, scaling linearly to 100% at FULL_WEIGHT_REQUESTS.
            # This prevents sampling luck from zeroing out legitimate miners while still
            # making Sybil attacks expensive (need 5 requests for full weight).
            request_ramp = min(1.0, stats.total_requests / FULL_WEIGHT_REQUESTS)

            # Start with AVERAGE points per request (not sum).
            # Using the sum caused volume to dominate: a slow miner serving 600 requests
            # outweighed a fast miner serving 100 requests. Normalizing to average ensures
            # weight reflects per-request QUALITY (speed + verification + consistency).
            # The availability bonus (applied later) separately rewards uptime/volume.
            weight = stats.net_points / stats.total_requests
            if weight <= 0:
                continue

            # Apply request ramp — partial weight for miners with few samples
            weight *= request_ramp

            base_weight = weight  # Track pre-bonus weight for combined cap

            # Consistency bonus/penalty (0.5-1.5x multiplier)
            consistency = stats.consistency_score
            weight *= (0.5 + consistency)  # Range: 0.5x to 1.5x

            # Cosine fidelity bonus (0.85-1.1x multiplier)
            # Rewards miners with consistently high cosine similarity
            # Uses tiered thresholds: optimized miners (0.7-0.9) get less bonus
            # but aren't penalized
            if stats.cosine_values:
                avg_cos = stats.avg_cosine
                # Map [COSINE_THRESHOLD..1.0] → [0.85..1.1]
                cos_range = 1.0 - COSINE_THRESHOLD
                cos_pos = min(1.0, max(0.0, (avg_cos - COSINE_THRESHOLD) / max(cos_range, 1e-9)))
                weight *= 0.85 + 0.25 * cos_pos

            # Absolute throughput bonus (1.0-1.5x multiplier)
            # Incentivizes miners to maximize TPS through optimization:
            # speculative decoding, better batching, faster hardware.
            # Kicks in above TPS_BONUS_THRESHOLD, maxes at TPS_BONUS_CEILING.
            # IMPORTANT: Scale TPS bonus by cosine fidelity so a distilled/quantized
            # model can't compensate verification penalty with raw speed.
            if stats.tps_values and len(stats.tps_values) >= 3:
                median_tps = float(np.median(stats.tps_values))
                if median_tps > TPS_BONUS_THRESHOLD:
                    tps_progress = min(1.0, (median_tps - TPS_BONUS_THRESHOLD) /
                                       max(TPS_BONUS_CEILING - TPS_BONUS_THRESHOLD, 1))
                    tps_bonus = 1.0 + (TPS_BONUS_MAX - 1.0) * tps_progress
                    # Dampen TPS bonus for low-fidelity miners: if cosine < 0.90,
                    # reduce the bonus proportionally. Full bonus only at cosine >= 0.95.
                    # RED-TEAM FIX (step 14): When cosine_values is EMPTY (no challenge
                    # data at all), apply maximum damping (cos_damper=0). A miner with
                    # zero verification evidence should NOT get speed bonuses — this
                    # closes the exploit where 100% cache-miss miners bypass the damper.
                    if stats.cosine_values:
                        cos_damper = min(1.0, max(0.0, (stats.avg_cosine - 0.80) / 0.15))
                    else:
                        cos_damper = 0.0  # No verification data = no speed bonus
                    tps_bonus = 1.0 + (tps_bonus - 1.0) * cos_damper
                    weight *= tps_bonus

            # Concurrency bonus (1.0-1.25x multiplier)
            # Incentivizes miners to optimize for concurrent request handling:
            # continuous batching, PagedAttention, efficient scheduling.
            # Measured via periodic bandwidth probes. Ratio = concurrent_tps / baseline_tps.
            if stats.concurrent_tps_ratios and len(stats.concurrent_tps_ratios) >= 3:
                median_ratio = float(np.median(stats.concurrent_tps_ratios))
                if median_ratio > CONCURRENCY_BONUS_THRESHOLD:
                    conc_progress = min(1.0, (median_ratio - CONCURRENCY_BONUS_THRESHOLD) /
                                        (1.0 - CONCURRENCY_BONUS_THRESHOLD))
                    conc_bonus = 1.0 + (CONCURRENCY_BONUS_MAX - 1.0) * conc_progress
                    weight *= conc_bonus

            # Cap combined bonus multiplier to prevent multiplicative stacking.
            # Raised from 2.5 to 4.0 to allow the higher TPS bonus (2.5x) to combine
            # with consistency and cosine bonuses. Speed differentiation is the priority.
            if base_weight > 0:
                bonus_ratio = weight / base_weight
                MAX_COMBINED_BONUS = 4.0
                if bonus_ratio > MAX_COMBINED_BONUS:
                    weight = base_weight * MAX_COMBINED_BONUS

            # Availability bonus applied in second pass (after main loop, before normalization)

            # Cache miss penalty — penalizes miners that dodge inline commitments.
            # A honest miner returns commitments on every request (<5% miss rate).
            # 100% miss = deliberate evasion (dropping commitment fields to save compute).
            # Tiered: >15% triggers penalty, 100% evasion = 90% weight reduction.
            miss_data = self._cache_miss_rates.get(uid)
            if miss_data:
                misses, total_attempts = miss_data
                if total_attempts >= 2:  # Lowered from 3 — catch evasion faster
                    miss_rate = misses / total_attempts
                    MISS_PENALTY_THRESHOLD = 0.15
                    MISS_MAX_PENALTY = 0.90  # Up from 0.60 — near-zero weight for full evasion
                    if miss_rate > MISS_PENALTY_THRESHOLD:
                        penalty_frac = min(1.0, (miss_rate - MISS_PENALTY_THRESHOLD) /
                                           (1.0 - MISS_PENALTY_THRESHOLD))
                        miss_factor = 1.0 - MISS_MAX_PENALTY * penalty_frac
                        weight *= miss_factor
                        log.warning(
                            f"Miner {uid}: cache miss rate {miss_rate:.0%} "
                            f"({misses}/{total_attempts}) → {miss_factor:.2f}x weight"
                        )

            # Divergence penalty (per-epoch)
            # Compares organic vs synthetic speed scores. Infrastructure noise causes
            # high divergence for ALL miners (6-14x) regardless of honesty. Use a
            # graduated penalty that scales with failure rate rather than a hard cliff.
            # At pass_rate>=0.90, no penalty (noise). At pass_rate<0.80, full penalty.
            # Between 0.80-0.90, linearly interpolated penalty fraction.
            # (Tightened from 0.97 to remove easy immunity window for gaming miners)
            # For divergence scaling: VOID-only miners (CPU/GPU divergence, no actual fails)
            # should not get the full penalty. Use effective_pr that treats VOIDs as neutral.
            total_ch = stats.passed_challenges + stats.failed_challenges
            if total_ch == 0 and stats.voided_challenges > 0:
                # Auditor challenged this miner but all results were inconclusive (CPU/GPU divergence).
                # Treat as pass_rate=0.90 for divergence scaling — not suspicious.
                pr = 0.90
            else:
                pr = stats.pass_rate
            div = stats.divergence
            # Graduated: penalty_scale goes from 0 at pr>=0.95 to 1.0 at pr<=0.80
            # Fixed: removed max(0.20) floor — miners with pr>=0.95 should get NO penalty.
            # The floor caused 14% weight loss on 100% pass-rate miners due to CPU/GPU divergence.
            div_penalty_scale = max(0.0, min(1.0, (0.95 - pr) / 0.15)) if div > DIVERGENCE_THRESHOLD else 0.0
            if div_penalty_scale > 0:
                if div > 0.25:
                    actual_penalty = DIVERGENCE_PENALTY_SEVERE * div_penalty_scale
                    weight *= (1.0 - actual_penalty)
                    if uid not in self._divergence_warned:
                        log.warning(f"Miner {uid}: SEVERE divergence={div:.3f} pr={pr:.0%} → -{actual_penalty:.0%} weight (scale={div_penalty_scale:.2f})")
                        self._divergence_warned.add(uid)
                elif div > DIVERGENCE_THRESHOLD:
                    actual_penalty = DIVERGENCE_PENALTY_MILD * div_penalty_scale
                    weight *= (1.0 - actual_penalty)
                    if uid not in self._divergence_warned:
                        log.warning(f"Miner {uid}: mild divergence={div:.3f} pr={pr:.0%} → -{actual_penalty:.0%} weight (scale={div_penalty_scale:.2f})")
                        self._divergence_warned.add(uid)
            elif div > DIVERGENCE_THRESHOLD:
                # pass_rate>=0.90: miner overwhelmingly passes challenges, divergence is
                # infrastructure noise. Only log once, no weight penalty.
                if uid not in self._divergence_warned:
                    log.info(f"Miner {uid}: divergence={div:.3f} but pass_rate={pr:.0%} — monitoring only")
                    self._divergence_warned.add(uid)

            # Cross-epoch divergence: catches miners staying under per-epoch sample minimums
            # Same graduated policy as per-epoch divergence.
            cross = self._cross_epoch_scores.get(uid)
            if cross and cross["organic"].count >= MIN_ORGANIC_SAMPLES and cross["synthetic"].count >= MIN_SYNTHETIC_SAMPLES:
                cross_org_mean = cross["organic"].mean
                cross_syn_mean = cross["synthetic"].mean
                if cross_syn_mean > 0:
                    cross_div = abs(cross_org_mean - cross_syn_mean) / max(cross_syn_mean, 0.001)
                    # Cross-epoch divergence: same graduated scale, no floor for high pass-rate
                    cross_div_scale = max(0.0, min(1.0, (0.95 - pr) / 0.15)) if cross_div > DIVERGENCE_THRESHOLD else 0.0
                    if cross_div_scale > 0:
                        if cross_div > 0.25:
                            actual_penalty = DIVERGENCE_PENALTY_SEVERE * cross_div_scale
                            weight *= (1.0 - actual_penalty)
                            if uid not in self._cross_div_warned:
                                log.warning(f"Miner {uid}: SEVERE cross-epoch divergence={cross_div:.3f} pr={pr:.0%} → -{actual_penalty:.0%} weight")
                                self._cross_div_warned.add(uid)
                        elif cross_div > DIVERGENCE_THRESHOLD:
                            actual_penalty = DIVERGENCE_PENALTY_MILD * cross_div_scale
                            weight *= (1.0 - actual_penalty)
                            if uid not in self._cross_div_warned:
                                log.warning(f"Miner {uid}: mild cross-epoch divergence={cross_div:.3f} pr={pr:.0%} → -{actual_penalty:.0%} weight")
                                self._cross_div_warned.add(uid)
                    elif cross_div > DIVERGENCE_THRESHOLD:
                        if uid not in self._cross_div_warned:
                            log.info(f"Miner {uid}: cross-epoch divergence={cross_div:.3f} but pass_rate={pr:.0%} — monitoring only")
                            self._cross_div_warned.add(uid)

            # Suspect penalty (current epoch)
            if stats.is_suspect:
                weight *= 0.1  # 90% penalty for suspected cheaters
                log.warning(f"Miner {uid}: SUSPECT → -90% weight")

            # Cross-epoch suspect history penalty
            prior_suspect_epochs = self._suspect_history.get(uid, 0)
            if prior_suspect_epochs > 0:
                # Each prior suspect epoch adds a 20% cumulative penalty (min 0.2x)
                history_factor = max(0.2, 1.0 - 0.2 * prior_suspect_epochs)
                weight *= history_factor
                log.warning(f"Miner {uid}: suspect history ({prior_suspect_epochs} epochs) → {history_factor:.1f}x weight")

            # Pass rate factor: weighted average of overall and recent pass rates.
            # RED-TEAM FIX (step 15): Changed from min(overall, recent) to weighted
            # average. The min() approach was exploitable: a miner could fail early,
            # then pass recent challenges to reset recent_pass_rate to 1.0, making
            # pr = min(0.85, 1.0) = 0.85 — only a 28% penalty. The reverse order
            # (pass early, fail late) gave min(0.94, 0.75) = 0.75 — a 44% penalty.
            # Weighted average (0.6 overall + 0.4 recent) treats both orderings
            # similarly and prevents recovery-ordering abuse.
            # Bayesian smoothing: add 3 virtual passes (increased from 2 for better
            # small-sample stability).
            total_challenged = stats.passed_challenges + stats.failed_challenges
            if total_challenged > 0:
                N_PRIOR = 1  # C5 H5-3: Minimal smoothing — 0/3 fails → pr=0.25 (was 0.50)
                pr_overall = (stats.passed_challenges + N_PRIOR) / (total_challenged + N_PRIOR)
                pr_recent = stats.recent_pass_rate
                # Weighted average: overall dominates early, recent gains influence later
                if total_challenged >= 5:
                    pr = 0.6 * pr_overall + 0.4 * pr_recent
                else:
                    pr = pr_overall  # Trust smoothed overall for small samples
                weight *= pr ** 2  # C5 H5-3: Convex curve — punishes low PR harder (0.5→0.25)

            # C4 H4-1/H4-2: Void rate penalty — DISABLED.
            # CPU/GPU divergence (CPU float32 validator vs GPU float16 miners) causes
            # 70-90% VOID rate for ALL miners, including our best performers (UID 11: 78%,
            # UID 38: 85%). The 0.30 threshold penalized every single miner, defeating its
            # purpose (catching connection-reset exploiters). The pass_rate factor already
            # handles cheaters (they get 0 passes). Re-enable when CPU/GPU divergence is
            # resolved (e.g. GPU-based validator, or float16 reference model).
            voided = stats.voided_challenges
            total_audited = stats.passed_challenges + stats.failed_challenges + voided
            if total_audited >= 5:
                void_rate = voided / total_audited
                if void_rate > 0.95 and stats.passed_challenges == 0:
                    # Only penalize if EXTREME void rate with zero passes — likely connection-reset exploit
                    weight *= 0.10
                    log.warning(f"Miner {uid}: void_rate={void_rate:.0%} ({voided}/{total_audited}) zero passes → 0.10x weight")

            # Challenge participation factor: penalize miners with zero challenges.
            # In split architecture (gateway defers challenges to auditor), challenge
            # counts are inherently low relative to request counts because the auditor
            # uses adaptive rates (10-100%). The old 25% ratio threshold punished fast
            # miners that served many requests but happened to have few auditor challenges.
            # Fix: only apply penalty for truly ZERO challenges (the auditor hasn't
            # verified them at all). Any non-zero challenge count means the auditor has
            # evaluated the miner — the pass_rate penalty handles quality from there.
            # VOIDs count as participation: the auditor DID challenge the miner, the
            # result was inconclusive due to CPU/GPU divergence — not the same as
            # "never challenged". A miner with many VOIDs and zero FAILs is legitimate.
            total_challenged = stats.passed_challenges + stats.failed_challenges
            total_audited = total_challenged + stats.voided_challenges
            if stats.total_requests >= 30 and total_audited == 0:
                # Raised threshold from 15→30 and softened penalty from 0.05→0.30.
                # With 30s auditor sync interval, fast miners hit 20+ requests before
                # first sync. 0.05x was too harsh for what may be a sync delay.
                weight *= 0.30
                log.warning(f"Miner {uid}: zero audits out of {stats.total_requests} requests → 0.30x weight")

            # Organic participation factor: penalize miners with suspiciously low
            # organic request counts relative to synthetic. Miners that avoid organic
            # traffic can evade divergence detection (which needs MIN_ORGANIC_SAMPLES).
            # A healthy miner should serve a comparable ratio of organic to synthetic.
            # This uses cross-epoch accumulated data to catch drip-feeders who stay
            # under per-epoch minimums.
            cross = self._cross_epoch_scores.get(uid)
            if cross:
                cross_org_count = cross["organic"].count
                cross_syn_count = cross["synthetic"].count
                cross_total = cross_org_count + cross_syn_count
                if cross_syn_count >= MIN_ORGANIC_SAMPLES and cross_org_count < MIN_ORGANIC_SAMPLES:
                    # Miner has enough synthetic samples but suspiciously few organic.
                    # Lowered from 10 to MIN_ORGANIC_SAMPLES to close the gap where
                    # a miner with 5-9 synthetic + <5 organic evaded both checks.
                    # Penalty: 0.6x for zero organic, scaling to 1.0x at MIN_ORGANIC_SAMPLES
                    organic_factor = 0.6 + 0.4 * (cross_org_count / max(MIN_ORGANIC_SAMPLES, 1))
                    weight *= organic_factor
                elif cross_total < MIN_ORGANIC_SAMPLES and stats.total_requests >= 3:
                    # Drip-feed evasion: miner keeps BOTH organic and synthetic counts
                    # below minimums to avoid all divergence checks. Lowered from 10 to
                    # 3 requests to match the challenge participation threshold.
                    drip_factor = 0.7 + 0.3 * (cross_total / max(MIN_ORGANIC_SAMPLES, 1))
                    weight *= drip_factor
                elif cross_total >= MIN_ORGANIC_SAMPLES and (cross_org_count < MIN_ORGANIC_SAMPLES or cross_syn_count < MIN_ORGANIC_SAMPLES):
                    # Imbalanced participation: enough total samples but insufficient
                    # diversity. A healthy miner should have both organic and synthetic.
                    # This catches the 9+4 or 4+9 gap that evaded the above conditions.
                    min_count = min(cross_org_count, cross_syn_count)
                    balance_factor = 0.7 + 0.3 * (min_count / max(MIN_ORGANIC_SAMPLES, 1))
                    weight *= balance_factor

            # Guard against NaN/Inf/subnormal contamination — a single bad value
            # would corrupt the entire normalization step. Subnormal floats
            # (< 1e-15) from chained penalty multipliers distort normalization.
            if not math.isfinite(weight) or weight < 1e-15:
                if weight != 0.0:
                    log.warning(f"Miner {uid}: non-finite or subnormal weight {weight} → 0.0")
                weight = 0.0
            # Trace weight factors — info level to diagnose weight=0 issues
            log.info(
                f"[WEIGHT_TRACE] Miner {uid}: net_pts={stats.net_points:.3f} "
                f"reqs={stats.total_requests} consistency={stats.consistency_score:.3f} "
                f"cosine_avg={stats.avg_cosine:.3f} "
                f"passed={stats.passed_challenges} failed={stats.failed_challenges} voided={stats.voided_challenges} "
                f"pr={stats.pass_rate:.3f} "
                f"raw_weight={weight:.9f}"
            )
            raw_weights[uid] = weight

        # Availability bonus: second pass using fleet-relative request counts.
        # Miners that serve more requests (higher uptime/reliability) get a bonus.
        # Uses log-ratio vs fleet median: serving 2x the median -> ~1.3x, 0.5x -> ~0.7x.
        # RED-TEAM FIX (step 13): Skip availability bonus for suspect miners or those
        # with suspect history — prevents cheaters from recouping weight via request flooding.
        if len(raw_weights) >= 2:
            request_counts = {uid: stats_dict[uid].total_requests for uid in raw_weights if uid in stats_dict}
            if request_counts:
                median_requests = float(np.median(list(request_counts.values())))
                if median_requests > 0:
                    for uid in list(raw_weights):
                        stats = stats_dict.get(uid)
                        if stats and (stats.is_suspect or self._suspect_history.get(uid, 0) > 0):
                            continue  # No availability bonus for suspect miners
                        count = request_counts.get(uid, 0)
                        if count > 0 and median_requests > 0:
                            ratio = count / median_requests
                            log_ratio = math.log2(max(ratio, 0.01))
                            log_ratio = max(-2.0, min(2.0, log_ratio))
                            if log_ratio >= 0:
                                avail_factor = 1.0 + (AVAILABILITY_BONUS_MAX - 1.0) * (log_ratio / 2.0)
                            else:
                                avail_factor = 1.0 + (1.0 - AVAILABILITY_BONUS_FLOOR) * (log_ratio / 2.0)
                            raw_weights[uid] *= avail_factor

        # Normalize
        total = sum(raw_weights.values())
        if total == 0 or not math.isfinite(total):
            return {}

        return {uid: w / total for uid, w in raw_weights.items()}

    def end_epoch(self) -> dict:
        """End current epoch, compute weights, return summary.

        Atomically swaps miner_stats to a fresh dict before computing weights,
        so concurrent record_request() calls go to the new epoch's dict and
        don't corrupt the snapshot being scored.
        """
        # Lock the swap to prevent record_request() from writing to stale dicts.
        # The lock scope is minimal — only covers the snapshot+swap, not computation.
        with self._epoch_lock:
            old_stats = self.miner_stats
            old_request_log = self.request_log
            old_population_ttft = list(self._population_ttft)
            old_population_tps = list(self._population_tps)

            # New epoch starts immediately — concurrent record_request() writes here
            self.miner_stats = {}
            self.request_log = deque(maxlen=self._MAX_REQUEST_LOG)
            self._population_ttft = deque(maxlen=self._MAX_POPULATION_SAMPLES)
            self._population_tps = deque(maxlen=self._MAX_POPULATION_SAMPLES)
            self._cache_miss_rates = {}  # Fresh per-epoch cache miss evaluation
            self._divergence_warned = set()   # Reset log-spam guards
            self._cross_div_warned = set()
            self._epoch_challenge_count = {}  # Reset per-epoch challenge counts
            self._epoch_request_count = {}    # Reset per-epoch request counts

            # C5 H5-1: Decay clean streaks at epoch boundary — prevent permanent low challenge rate
            for uid in list(self._clean_streak):
                self._clean_streak[uid] = int(self._clean_streak[uid] * 0.7)
                if self._clean_streak[uid] == 0:
                    del self._clean_streak[uid]

        # Update suspect history BEFORE computing weights so that first-time
        # suspects get the history penalty in the same epoch they're flagged.
        # This closes the 1-epoch amnesty window where a newly-suspect miner
        # only got the current-epoch 0.1x penalty without the history multiplier.
        for uid, stats in old_stats.items():
            if stats.is_suspect:
                prior = self._suspect_history.get(uid, 0)
                increment = 3 * (2 ** min(prior // 3, 4))
                self._suspect_history[uid] = prior + increment
                hotkey = self._uid_to_hotkey.get(uid)
                if hotkey:
                    self._hotkey_suspect_history[hotkey] = self._suspect_history[uid]
            else:
                if uid in self._suspect_history:
                    self._suspect_history[uid] = max(0, self._suspect_history[uid] - 1)
                    if self._suspect_history[uid] == 0:
                        del self._suspect_history[uid]
                hotkey = self._uid_to_hotkey.get(uid)
                if hotkey and hotkey in self._hotkey_suspect_history:
                    self._hotkey_suspect_history[hotkey] = max(0, self._hotkey_suspect_history[hotkey] - 1)
                    if self._hotkey_suspect_history[hotkey] == 0:
                        del self._hotkey_suspect_history[hotkey]

        # Compute weights from the snapshot (no concurrent mutation possible)
        weights = self._compute_weights_from(old_stats)

        summary = {
            "epoch": self.epoch_number,
            "duration_s": time.time() - self.current_epoch_start,
            "total_requests": sum(s.total_requests for s in old_stats.values()),
            "miners": {},
            "weights": weights,
        }

        for uid, stats in old_stats.items():
            summary["miners"][uid] = {
                "total_points": stats.total_points,
                "penalty_points": stats.penalty_points,
                "net_points": stats.net_points,
                "total_requests": stats.total_requests,
                "organic_count": len(stats.organic_scores),
                "synthetic_count": len(stats.synthetic_scores),
                "organic_mean": stats.organic_mean,
                "synthetic_mean": stats.synthetic_mean,
                "divergence": stats.divergence,
                "consistency": stats.consistency_score,
                "passed_challenges": stats.passed_challenges,
                "failed_challenges": stats.failed_challenges,
                "voided_challenges": stats.voided_challenges,
                "pass_rate": stats.pass_rate,
                "is_suspect": stats.is_suspect,
                "avg_ttft_ms": stats.avg_ttft_ms,
                "avg_tps": stats.avg_tps,
                "avg_cosine": stats.avg_cosine,
                "weight": weights.get(uid, 0.0),
            }

        self.weight_history.append(summary)

        # Suspect history already updated above (before weight computation).
        # Advance epoch counter (miner_stats already swapped at top of method)
        self.epoch_number += 1
        self.current_epoch_start = time.time()
        self._current_epoch_target_s = self._randomize_epoch_length()

        log.info(
            f"Epoch {summary['epoch']} complete: "
            f"{summary['total_requests']} requests, "
            f"{len(weights)} miners weighted"
        )

        return summary

    def get_scoreboard(self) -> list[dict]:
        """Get current epoch scoreboard with computed weights.

        Includes fully-penalized weights (cache miss, divergence, suspect, etc.)
        so consumers don't need to recompute penalties from raw net_points.
        Cached for 60s. NEVER blocks longer than the cache lookup — returns stale
        data (or empty) rather than computing on the hot path.
        """
        now = time.time()
        if hasattr(self, '_scoreboard_cache') and now - self._scoreboard_cache_time < 10:
            return self._scoreboard_cache
        # If cache is stale but exists, return it rather than blocking.
        # The epoch_loop will refresh the cache when it calls compute_weights().
        if hasattr(self, '_scoreboard_cache'):
            # Mark as recently refreshed to prevent re-trying for 60s
            self._scoreboard_cache_time = now
            return self._scoreboard_cache
        # First call — compute synchronously (only happens once at startup)
        live_weights = self.compute_weights()
        board = []
        for uid, stats in self.miner_stats.items():
            board.append({
                "uid": uid,
                "net_points": stats.net_points,
                "total_points": stats.total_points,
                "penalty_points": stats.penalty_points,
                "requests": stats.total_requests,
                "organic": len(stats.organic_scores),
                "synthetic": len(stats.synthetic_scores),
                "divergence": stats.divergence,
                "consistency": stats.consistency_score,
                "pass_rate": stats.pass_rate,
                "passed_challenges": stats.passed_challenges,
                "failed_challenges": stats.failed_challenges,
                "voided_challenges": stats.voided_challenges,
                "is_suspect": stats.is_suspect,
                "avg_ttft_ms": stats.avg_ttft_ms,
                "avg_tps": stats.avg_tps,
                "avg_cosine": stats.avg_cosine,
                "computed_weight": live_weights.get(uid, 0.0),
            })
        result = sorted(board, key=lambda x: x["computed_weight"], reverse=True)
        self._scoreboard_cache = result
        self._scoreboard_cache_time = now
        return result

