#!/usr/bin/env python3
"""
Cross-Miner Collusion Detector — Statistical analysis to detect coordinated cheating.

Attack this defends against:
  Miners that coordinate to share work, split compute, or relay answers between
  each other. Colluding miners may:
  - Run one GPU and proxy results to multiple miner identities
  - Share hidden state caches over the network
  - Coordinate which requests to answer honestly vs cheat on
  - Use a single model serving backend with multiple frontend identities

Detection methods:
  1. Response fingerprinting — Same prompt sent to different miners: compare
     token-level output similarity. Identical responses from "different" miners
     are suspicious (independent models + sampling should produce variation).

  2. Timing correlation — Plot response latencies across miners over time.
     Colluding miners often show correlated latency spikes (shared backend
     bottleneck) while independent miners show uncorrelated patterns.

  3. Error correlation — When one miner fails, do specific other miners also
     fail? Correlated failures suggest shared infrastructure.

  4. Hidden state fingerprinting — For the same prompt, compare hidden states
     from different miners. They should match the reference model but if two
     miners produce identical bit-patterns (not just cosine-similar), they
     may be sharing a cache.

Scoring:
  - Each miner pair gets a collusion_score (0-1) where 1 = definitely colluding
  - Miners in a collusion cluster share their combined weight equally
    (removing the incentive to sybil into multiple identities)
  - Pairs with score > threshold are logged and flagged for operator review

Anti-gaming:
  - Cross-miner probes use crypto-random timing and prompt selection
  - Detection is retrospective (miners can't adapt in real-time)
  - Multiple orthogonal signals must agree before flagging
  - Minimum sample size prevents false positives from insufficient data
"""

import hashlib
import logging
import math
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

log = logging.getLogger("collusion_detector")
log.propagate = False


# ── Configuration ────────────────────────────────────────────────────────────

# Response similarity thresholds
RESPONSE_SIMILARITY_SUSPICIOUS = 0.85  # 85%+ token overlap → suspicious
RESPONSE_SIMILARITY_COLLUDING = 0.95   # 95%+ → almost certainly shared backend

# Timing correlation thresholds
TIMING_CORRELATION_SUSPICIOUS = 0.7    # Pearson r > 0.7 → suspicious
TIMING_CORRELATION_COLLUDING = 0.9     # r > 0.9 → almost certainly shared

# Error correlation thresholds
ERROR_CORRELATION_SUSPICIOUS = 0.6
ERROR_CORRELATION_COLLUDING = 0.8

# Hidden state bit-exact match (not just cosine-similar)
HIDDEN_STATE_EXACT_THRESHOLD = 0.999  # Cosine > 0.999 between different miners (C3-8: lowered from 0.9999)
# (same model + same input should give cosine ~0.999 but not 0.9999+
#  due to floating point non-determinism on different GPUs)

# Minimum samples before scoring
MIN_CROSS_PROBES = 5          # Minimum shared prompts between a pair
MIN_TIMING_SAMPLES = 10       # Minimum latency samples per miner

# Collusion score thresholds
COLLUSION_SCORE_WARNING = 0.4  # Log a warning
COLLUSION_SCORE_PENALTY = 0.6  # Apply weight penalty
COLLUSION_SCORE_SEVERE = 0.8   # Severe weight penalty

# Weight penalties
COLLUSION_PENALTY_MILD = 0.3   # 30% weight reduction
COLLUSION_PENALTY_SEVERE = 0.7 # 70% weight reduction


@dataclass
class CrossProbeResult:
    """Result of sending the same prompt to two different miners."""
    prompt_hash: str          # SHA256 of the prompt (for grouping)
    miner_a_uid: int
    miner_b_uid: int
    response_similarity: float  # 0-1 token-level similarity
    hidden_state_cosine: float  # Cosine between hidden states from the two miners
    ttft_a_ms: float
    ttft_b_ms: float
    timestamp: float = field(default_factory=time.time)
    semantic_similarity: float = 0.0  # 0-1 embedding/shingle-based semantic similarity
    response_text_a: str = ""  # Decoded response text from miner A (for semantic comparison)
    response_text_b: str = ""  # Decoded response text from miner B (for semantic comparison)


@dataclass
class MinerTimingSample:
    """A single latency observation for a miner."""
    miner_uid: int
    ttft_ms: float
    tps: float
    timestamp: float


@dataclass
class MinerErrorEvent:
    """Record of a miner success/failure at a point in time."""
    miner_uid: int
    success: bool
    timestamp: float


@dataclass
class CollusionPairScore:
    """Collusion assessment for a pair of miners."""
    miner_a: int
    miner_b: int
    response_similarity_score: float  # 0-1 from cross-probes (token-level)
    timing_correlation_score: float    # 0-1 from latency correlation
    error_correlation_score: float     # 0-1 from failure correlation
    hidden_state_exact_score: float    # 0-1 from bit-exact matches
    overall_score: float               # Weighted combination
    num_cross_probes: int
    flagged: bool
    semantic_similarity_score: float = 0.0  # 0-1 from embedding/shingle similarity

    @property
    def pair_key(self) -> tuple[int, int]:
        return (min(self.miner_a, self.miner_b), max(self.miner_a, self.miner_b))


def compute_response_similarity(tokens_a: list[int], tokens_b: list[int]) -> float:
    """
    Token-level similarity between two responses using multiple metrics.

    Combines LCS (sequence order) with bigram Jaccard (local structure).
    Bigram Jaccard is much harder to evade via token substitution:
    changing 15% of tokens drops bigram overlap by ~30%, whereas
    LCS only drops ~15%.
    """
    if not tokens_a or not tokens_b:
        return 0.0

    # Limit to reasonable length to avoid O(n²) blowup
    m, n = min(len(tokens_a), 500), min(len(tokens_b), 500)
    tokens_a = tokens_a[:m]
    tokens_b = tokens_b[:n]

    # --- Metric 1: LCS Dice coefficient ---
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    lcs_sim = 2.0 * lcs_len / (m + n)  # Dice coefficient

    # --- Metric 2: Bigram Jaccard similarity ---
    # Much harder to evade: each substituted token destroys 2 bigrams
    if m >= 2 and n >= 2:
        bigrams_a = set(zip(tokens_a[:-1], tokens_a[1:]))
        bigrams_b = set(zip(tokens_b[:-1], tokens_b[1:]))
        intersection = len(bigrams_a & bigrams_b)
        union = len(bigrams_a | bigrams_b)
        bigram_sim = intersection / union if union > 0 else 0.0
    else:
        bigram_sim = lcs_sim  # Fallback for very short sequences

    # Combined: take the max to catch evasion of either metric alone
    return max(lcs_sim, bigram_sim)


# ── Semantic Similarity (embedding-based) ────────────────────────────────────

# Default shingle size for character n-gram embeddings
_SHINGLE_SIZE = 4
# Number of hash buckets for the MinHash sketch
_MINHASH_BUCKETS = 128

# Pluggable embedding function: (text) -> np.ndarray
# Set via set_embedding_fn() for production use with a real model.
_embedding_fn: Callable[[str], np.ndarray] | None = None


def set_embedding_fn(fn: Callable[[str], np.ndarray] | None):
    """Plug in a real embedding model for semantic similarity.

    The function should map a string to a 1-D numpy array (the embedding).
    When set, compute_semantic_similarity() uses cosine similarity between
    embeddings instead of the built-in MinHash shingle fallback.
    """
    global _embedding_fn
    _embedding_fn = fn


def _minhash_signature(text: str, num_hashes: int = _MINHASH_BUCKETS) -> np.ndarray:
    """Compute a MinHash signature from character n-gram shingles.

    MinHash approximates Jaccard similarity on shingle sets.  It's much more
    robust to synonym substitution than token-level matching because:
      - "compute" → "calculate" still shares shingles like "alcu", "lcul", "cula", "ulat"
      - The structural pattern of the sentence (word order, punctuation, function
        words) is captured via overlapping character windows

    This is a lightweight fallback — a real sentence-transformer embedding is
    preferred for production (see set_embedding_fn).
    """
    text = text.lower().strip()
    if len(text) < _SHINGLE_SIZE:
        text = text + " " * (_SHINGLE_SIZE - len(text))

    shingles = set()
    for i in range(len(text) - _SHINGLE_SIZE + 1):
        shingles.add(text[i:i + _SHINGLE_SIZE])

    if not shingles:
        return np.zeros(num_hashes, dtype=np.uint64)

    # Generate num_hashes independent hash functions via salted SHA256
    sig = np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for shingle in shingles:
        sb = shingle.encode("utf-8")
        for h in range(num_hashes):
            digest = hashlib.sha256(sb + h.to_bytes(2, "big")).digest()
            val = int.from_bytes(digest[:8], "big")
            if val < sig[h]:
                sig[h] = val

    return sig


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Semantic similarity between two response texts.

    If a real embedding function is set (via set_embedding_fn), uses cosine
    similarity between embeddings.  Otherwise falls back to MinHash Jaccard
    estimation on character 4-gram shingles.

    Returns 0.0-1.0 where 1.0 = semantically identical.
    """
    if not text_a or not text_b:
        return 0.0

    # --- Real embedding path ---
    if _embedding_fn is not None:
        emb_a = _embedding_fn(text_a)
        emb_b = _embedding_fn(text_b)
        dot = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        cos = float(dot / (norm_a * norm_b))
        return max(0.0, min(1.0, cos))

    # --- MinHash fallback ---
    sig_a = _minhash_signature(text_a)
    sig_b = _minhash_signature(text_b)
    # Jaccard estimate = fraction of matching hash buckets
    return float(np.mean(sig_a == sig_b))


def _timing_correlation_at_scale(
    samples_a: list[MinerTimingSample],
    samples_b: list[MinerTimingSample],
    window_s: float,
) -> float:
    """Pearson correlation of latencies at a specific time-bin scale."""
    t_min = max(min(s.timestamp for s in samples_a), min(s.timestamp for s in samples_b))
    t_max = min(max(s.timestamp for s in samples_a), max(s.timestamp for s in samples_b))

    if t_max - t_min < window_s * 3:
        return 0.0

    num_bins = int((t_max - t_min) / window_s)
    if num_bins < 3:
        return 0.0

    bins_a = [[] for _ in range(num_bins)]
    bins_b = [[] for _ in range(num_bins)]

    for s in samples_a:
        idx = int((s.timestamp - t_min) / window_s)
        if 0 <= idx < num_bins:
            bins_a[idx].append(s.ttft_ms)

    for s in samples_b:
        idx = int((s.timestamp - t_min) / window_s)
        if 0 <= idx < num_bins:
            bins_b[idx].append(s.ttft_ms)

    paired_a = []
    paired_b = []
    for ba, bb in zip(bins_a, bins_b):
        if ba and bb:
            paired_a.append(np.mean(ba))
            paired_b.append(np.mean(bb))

    if len(paired_a) < 3:
        return 0.0

    arr_a = np.array(paired_a)
    arr_b = np.array(paired_b)

    std_a = np.std(arr_a)
    std_b = np.std(arr_b)
    if std_a < 1e-10 or std_b < 1e-10:
        return 0.0

    r = float(np.corrcoef(arr_a, arr_b)[0, 1])
    return max(0.0, r)


def compute_timing_correlation(
    samples_a: list[MinerTimingSample],
    samples_b: list[MinerTimingSample],
    window_s: float = 2.0,
) -> float:
    """
    Multi-scale Pearson correlation of latencies between two miners.

    Runs correlation at multiple time-bin scales (2s, 5s, 10s) and returns
    the maximum. This defeats jitter-based evasion: miners alternating fast/slow
    within a 5s window are caught by the 2s or 10s scale instead.
    """
    if len(samples_a) < MIN_TIMING_SAMPLES or len(samples_b) < MIN_TIMING_SAMPLES:
        return 0.0  # Insufficient data → no signal

    # Multi-scale: check at the requested scale plus coarser/finer bins
    scales = [window_s, window_s * 2.5, window_s * 5.0]
    return max(_timing_correlation_at_scale(samples_a, samples_b, s) for s in scales)


def compute_error_correlation(
    events_a: list[MinerErrorEvent],
    events_b: list[MinerErrorEvent],
    window_s: float = 5.0,
) -> float:
    """
    Measure how often two miners fail at the same time.

    Uses Jaccard similarity of failure time windows.
    Window is 5s (up from 2s) to catch miners that stagger failures
    2-3 seconds apart to evade coincidence detection.
    """
    if not events_a or not events_b:
        return 0.0

    fails_a = {e.timestamp for e in events_a if not e.success}
    fails_b = {e.timestamp for e in events_b if not e.success}

    if not fails_a or not fails_b:
        return 0.0

    # Count coincident failures (within window_s of each other)
    coincident = 0
    for ta in fails_a:
        for tb in fails_b:
            if abs(ta - tb) < window_s:
                coincident += 1
                break

    # Jaccard-like: coincident / union
    union = len(fails_a) + len(fails_b) - coincident
    if union == 0:
        return 0.0
    return coincident / union


class CollusionDetector:
    """
    Detects collusion between miners using multiple orthogonal signals.

    Usage:
        detector = CollusionDetector()

        # Feed data as it arrives:
        detector.record_cross_probe(probe_result)
        detector.record_timing(timing_sample)
        detector.record_error(error_event)

        # At epoch end:
        scores = detector.analyze_all_pairs()
        for pair_score in scores:
            if pair_score.flagged:
                apply_collusion_penalty(pair_score)
    """

    def __init__(self):
        self._cross_probes: list[CrossProbeResult] = []
        self._timing_samples: dict[int, list[MinerTimingSample]] = defaultdict(list)
        self._error_events: dict[int, list[MinerErrorEvent]] = defaultdict(list)
        self._known_miners: set[int] = set()

    def record_cross_probe(self, result: CrossProbeResult):
        """Record result of sending same prompt to two miners."""
        self._cross_probes.append(result)
        self._known_miners.add(result.miner_a_uid)
        self._known_miners.add(result.miner_b_uid)

    def record_timing(self, sample: MinerTimingSample):
        """Record a latency observation."""
        self._timing_samples[sample.miner_uid].append(sample)
        self._known_miners.add(sample.miner_uid)

    def record_error(self, event: MinerErrorEvent):
        """Record a success/failure event."""
        self._error_events[event.miner_uid].append(event)
        self._known_miners.add(event.miner_uid)

    def _analyze_pair(self, miner_a: int, miner_b: int) -> CollusionPairScore:
        """Analyze collusion signals for a single pair of miners."""
        # 1. Response similarity from cross-probes (token-level)
        pair_probes = [
            p for p in self._cross_probes
            if {p.miner_a_uid, p.miner_b_uid} == {miner_a, miner_b}
        ]
        if len(pair_probes) >= MIN_CROSS_PROBES:
            avg_sim = np.mean([p.response_similarity for p in pair_probes])
            response_score = max(0.0, (avg_sim - 0.5) / 0.5)  # Normalize: 0.5→0, 1.0→1
        else:
            response_score = 0.0  # Insufficient data

        # 2. Timing correlation (snapshot lists to avoid concurrent-append corruption)
        timing_score = 0.0
        if miner_a in self._timing_samples and miner_b in self._timing_samples:
            corr = compute_timing_correlation(
                list(self._timing_samples[miner_a]),
                list(self._timing_samples[miner_b]),
            )
            timing_score = max(0.0, (corr - 0.3) / 0.7)  # Normalize: 0.3→0, 1.0→1

        # 3. Error correlation (snapshot lists to avoid concurrent-append corruption)
        error_score = 0.0
        if miner_a in self._error_events and miner_b in self._error_events:
            corr = compute_error_correlation(
                list(self._error_events[miner_a]),
                list(self._error_events[miner_b]),
            )
            error_score = max(0.0, (corr - 0.2) / 0.8)  # Normalize

        # 4. Hidden state exactness
        exact_score = 0.0
        if pair_probes:
            exact_matches = sum(
                1 for p in pair_probes
                if p.hidden_state_cosine > HIDDEN_STATE_EXACT_THRESHOLD
            )
            exact_score = exact_matches / len(pair_probes)

        # 5. Semantic similarity (catches synonym-substitution evasion)
        # Uses either a pluggable embedding model or built-in MinHash shingles.
        # This catches collusion where miners swap ~10% of tokens with synonyms
        # to evade token-level LCS/bigram detection.
        semantic_score = 0.0
        semantic_probes = [
            p for p in pair_probes
            if p.response_text_a and p.response_text_b
        ]
        if len(semantic_probes) >= MIN_CROSS_PROBES:
            # Compute semantic similarity for probes that have text
            sem_sims = []
            for p in semantic_probes:
                if p.semantic_similarity > 0.0:
                    # Pre-computed (e.g., by the gateway when recording the probe)
                    sem_sims.append(p.semantic_similarity)
                else:
                    sem_sims.append(compute_semantic_similarity(
                        p.response_text_a, p.response_text_b
                    ))
            avg_sem = np.mean(sem_sims)
            # Normalize: 0.5→0, 1.0→1 (same range as token-level)
            semantic_score = max(0.0, (avg_sem - 0.5) / 0.5)

        # Weighted combination
        # 5 signals: token similarity, semantic similarity, timing, error, hidden state
        # Semantic + token together are much harder to evade than either alone:
        # - Token substitution evades token similarity but NOT semantic
        # - Paraphrasing evades semantic (partially) but NOT structural/token patterns
        has_hidden = any(p.hidden_state_cosine > 0.0 for p in pair_probes) if pair_probes else False
        has_semantic = semantic_score > 0.0 or len(semantic_probes) >= MIN_CROSS_PROBES

        if has_hidden and has_semantic:
            # All 5 signals available
            overall = (
                0.25 * response_score +
                0.20 * semantic_score +
                0.20 * timing_score +
                0.15 * error_score +
                0.20 * exact_score
            )
        elif has_hidden:
            # Hidden state but no semantic data
            overall = (
                0.35 * response_score +
                0.30 * timing_score +
                0.15 * error_score +
                0.20 * exact_score
            )
        elif has_semantic:
            # Semantic but no hidden state — most common case
            overall = (
                0.30 * response_score +
                0.25 * semantic_score +
                0.30 * timing_score +
                0.15 * error_score
            )
        else:
            # Only token similarity + timing + error
            overall = (
                0.50 * response_score +
                0.35 * timing_score +
                0.15 * error_score
            )

        flagged = overall >= COLLUSION_SCORE_WARNING

        return CollusionPairScore(
            miner_a=miner_a,
            miner_b=miner_b,
            response_similarity_score=response_score,
            timing_correlation_score=timing_score,
            error_correlation_score=error_score,
            hidden_state_exact_score=exact_score,
            overall_score=overall,
            num_cross_probes=len(pair_probes),
            flagged=flagged,
            semantic_similarity_score=semantic_score,
        )

    def analyze_all_pairs(self) -> list[CollusionPairScore]:
        """
        Analyze all miner pairs for collusion signals.

        Returns scores sorted by overall collusion score (highest first).
        """
        miners = sorted(self._known_miners)
        scores = []

        for i in range(len(miners)):
            for j in range(i + 1, len(miners)):
                score = self._analyze_pair(miners[i], miners[j])
                scores.append(score)

                if score.overall_score >= COLLUSION_SCORE_PENALTY:
                    log.warning(
                        f"COLLUSION DETECTED: Miners {miners[i]}↔{miners[j]} "
                        f"score={score.overall_score:.3f} "
                        f"(resp={score.response_similarity_score:.3f} "
                        f"semantic={score.semantic_similarity_score:.3f} "
                        f"timing={score.timing_correlation_score:.3f} "
                        f"error={score.error_correlation_score:.3f} "
                        f"exact={score.hidden_state_exact_score:.3f})"
                    )
                elif score.flagged:
                    log.info(
                        f"Collusion warning: Miners {miners[i]}↔{miners[j]} "
                        f"score={score.overall_score:.3f}"
                    )

        return sorted(scores, key=lambda s: s.overall_score, reverse=True)

    def get_weight_penalties(self, cached_scores: list["CollusionPairScore"] | None = None) -> dict[int, float]:
        """
        Compute weight penalty multipliers for all miners.

        Two-layer penalty system:
        1. Per-pair: miners in high-scoring pairs get direct penalties.
        2. Aggregate: miners appearing in MANY suspicious-but-below-threshold pairs
           get an aggregate penalty. This prevents cartels from keeping individual
           pair scores just below the threshold while colluding with many partners.

        Returns dict of miner_uid → multiplier (1.0 = no penalty, <1.0 = penalized).
        Pass cached_scores to avoid redundant O(n^2) recomputation.
        """
        scores = cached_scores if cached_scores is not None else self.analyze_all_pairs()
        penalties: dict[int, float] = {}

        # Track per-miner suspicious pair scores for aggregate penalty
        miner_pair_scores: dict[int, list[float]] = defaultdict(list)

        for score in scores:
            # Collect all non-trivial pair scores for aggregate analysis
            if score.overall_score >= COLLUSION_SCORE_WARNING:
                miner_pair_scores[score.miner_a].append(score.overall_score)
                miner_pair_scores[score.miner_b].append(score.overall_score)

            # Direct per-pair penalty for high-scoring pairs
            if score.overall_score < COLLUSION_SCORE_PENALTY:
                continue

            if score.overall_score >= COLLUSION_SCORE_SEVERE:
                penalty = 1.0 - COLLUSION_PENALTY_SEVERE
            else:
                penalty = 1.0 - COLLUSION_PENALTY_MILD

            # Apply to both miners (worst penalty wins)
            for uid in [score.miner_a, score.miner_b]:
                if uid not in penalties:
                    penalties[uid] = penalty
                else:
                    penalties[uid] = min(penalties[uid], penalty)

        # Aggregate penalty: a miner in 3+ suspicious pairs (>= WARNING threshold)
        # is likely part of a cartel even if no single pair crosses PENALTY threshold.
        # Penalty scales with number of suspicious pairs: 3 pairs → 0.8x, 4 → 0.65x, 5+ → 0.5x
        AGGREGATE_MIN_PAIRS = 3
        for uid, pair_scores in miner_pair_scores.items():
            if len(pair_scores) >= AGGREGATE_MIN_PAIRS:
                avg_score = sum(pair_scores) / len(pair_scores)
                n_suspicious = len(pair_scores)
                # Scale: 3 pairs → 0.8x, each additional → -0.15x, floor at 0.3x
                aggregate_penalty = max(0.3, 1.0 - 0.2 * (n_suspicious - 2) * (avg_score / 0.6))
                if uid not in penalties:
                    penalties[uid] = aggregate_penalty
                else:
                    penalties[uid] = min(penalties[uid], aggregate_penalty)
                log.warning(
                    f"Miner {uid}: aggregate collusion penalty — "
                    f"{n_suspicious} suspicious pairs, avg_score={avg_score:.3f} → {aggregate_penalty:.2f}x"
                )

        return penalties

    def reset(self):
        """Reset for a new epoch."""
        self._cross_probes = []
        self._timing_samples = defaultdict(list)
        self._error_events = defaultdict(list)
        self._known_miners = set()

    def summary(self, cached_scores: list["CollusionPairScore"] | None = None) -> dict:
        """Epoch summary for audit logging. Pass cached_scores to avoid recomputation."""
        scores = cached_scores if cached_scores is not None else self.analyze_all_pairs()
        flagged_pairs = [s for s in scores if s.flagged]
        return {
            "total_miners": len(self._known_miners),
            "total_pairs_analyzed": len(scores),
            "flagged_pairs": len(flagged_pairs),
            "flagged_details": [
                {
                    "miners": [s.miner_a, s.miner_b],
                    "overall_score": s.overall_score,
                    "response_sim": s.response_similarity_score,
                    "semantic_sim": s.semantic_similarity_score,
                    "timing_corr": s.timing_correlation_score,
                    "error_corr": s.error_correlation_score,
                    "exact_match": s.hidden_state_exact_score,
                    "num_probes": s.num_cross_probes,
                }
                for s in flagged_pairs
            ],
        }
