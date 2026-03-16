#!/usr/bin/env python3
"""
KV Cache Probe System — Verify miners maintain real KV caches.

Attack this defends against:
  Miners claim KV cache hits without actually caching prefix computations.
  A faker would have identical TTFT for turn-1 and turn-2 of the same session
  because they recompute from scratch every time. A real cache produces a
  measurable TTFT speedup on the continuation turn.

Probe protocol:
  1. Send turn-1 (long shared prefix) to miner via session routing
  2. Wait a random delay (2-8s) to simulate real user behavior
  3. Send turn-2 (short continuation) on the same session
  4. Measure TTFT ratio (turn2_ttft / turn1_ttft)
  5. A real cache yields ratio < 0.7 (turn-2 is faster)
  6. Additionally challenge hidden states on turn-2 to verify the model
     actually ran (not just returning cached text)

Scoring integration:
  - cache_efficiency_score: 0-1 based on TTFT speedup ratio
  - Fakers score ~0 (no speedup), honest miners score ~0.8-1.0
  - Score feeds into HardenedScoringEngine as a multiplier on weight

Anti-gaming:
  - Prefix length varies randomly (100-500 tokens equivalent)
  - Wait delay varies with crypto randomness
  - Turn-2 prompt varies (can't precompute)
  - Hidden state challenge on turn-2 proves real model execution
  - Multiple probes per epoch, results aggregated via median (robust to outliers)
"""

import logging
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("kv_cache_prober")
log.propagate = False


# ── Configuration ────────────────────────────────────────────────────────────

# TTFT ratio thresholds for cache verification
# ratio = turn2_ttft / turn1_ttft
# Real cache: ratio < 0.7 (turn-2 is 30%+ faster due to prefix reuse)
# No cache:   ratio ≈ 1.0 (both turns take similar time)
CACHE_SPEEDUP_EXCELLENT = 0.3   # 70%+ speedup → clearly cached
CACHE_SPEEDUP_GOOD = 0.5       # 50%+ speedup → likely cached
CACHE_SPEEDUP_MARGINAL = 0.7   # 30%+ speedup → marginal
CACHE_SPEEDUP_NONE = 0.9       # < 10% speedup → probably fake

# Probe timing
PROBE_DELAY_MIN_S = 2.0        # Minimum wait between turns
PROBE_DELAY_MAX_S = 8.0        # Maximum wait between turns

# Minimum probes before scoring
MIN_PROBES_FOR_SCORE = 3

# Score multiplier for weight calculation
CACHE_WEIGHT_MULTIPLIER = 0.35  # 35% of total weight comes from cache score


# ── Probe Template Generation ──────────────────────────────────────────────
# Procedurally generated probes using multiple sentence structures.
# No single fixed prefix — miners cannot fingerprint on "I'm building..."
# or any other constant pattern. Each turn-1 is assembled from:
#   FRAME(domain) + scenario + detail + question
# Frames vary sentence structure so probes look like diverse organic chat.
#
# Combinatorial space: 12 frames × 20 domains × 15 scenarios × 16 details × 10 questions
#   = 576,000 turn-1 variants × 240 turn-2 variants = 138M unique probe pairs

_PROBE_FRAMES = [
    "I'm building {domain}.",
    "I'm working on {domain}.",
    "My team is working on {domain}.",
    "We're building {domain} and I'd appreciate some advice.",
    "I need some help with {domain}.",
    "I've been asked to work on {domain}.",
    "We just started on {domain}.",
    "We're investing in {domain}.",
    "I'm looking into {domain}.",
    "Hey, can you help me think through {domain}?",
    "So we have {domain} and I have some questions.",
    "Quick question about {domain}.",
]

_PROBE_DOMAINS = [
    "a distributed system for processing financial transactions",
    "a machine learning pipeline for satellite image analysis",
    "a microservices migration from a monolithic application",
    "a real-time collaborative editor using CRDTs",
    "a programming language for quantum-classical hybrid computing",
    "a high-frequency trading platform with sub-microsecond latency",
    "a content delivery network optimized for video streaming",
    "a privacy-preserving analytics system using differential privacy",
    "a decentralized identity management protocol",
    "a compiler for a new systems programming language",
    "a distributed database with strong consistency guarantees",
    "a robotics control system for autonomous warehouse operations",
    "a genome sequencing pipeline for clinical diagnostics",
    "a smart contract auditing platform with formal verification",
    "a recommendation engine serving 50 million daily active users",
    "an edge computing framework for IoT sensor networks",
    "a real-time fraud detection system for payment processing",
    "a multi-tenant SaaS platform with isolated data stores",
    "a natural language interface for SQL database queries",
    "a container orchestration system for GPU workloads",
]

_PROBE_SCENARIOS = [
    "The system needs to handle at least {n} requests per second with {latency} latency.",
    "We currently serve {n} active users and the load is growing {rate} month over month.",
    "The architecture includes {count} microservices communicating via {protocol}.",
    "Our team of {n} engineers needs to maintain this while shipping new features.",
    "The dataset is approximately {size} and growing by {growth} per day.",
    "We've been running in production for {time} and are hitting scaling limits.",
    "Reliability requirements are {nines} uptime with a {rpo} RPO.",
    "The budget constraint is {budget} per month for cloud infrastructure.",
    "We need to comply with {regulation} while maintaining performance.",
    "The system operates across {count} availability zones in {regions} regions.",
    "Current P99 latency is {latency} but we need to bring it below {target}.",
    "The team is split across {count} time zones and uses trunk-based development.",
    "We need to support both batch processing and real-time streaming of data.",
    "The existing codebase has {kloc}k lines of code with limited test coverage.",
    "We're migrating from {old_tech} to {new_tech} without any downtime allowed.",
]

_PROBE_DETAILS = [
    "We need all the nodes to stay in sync on the order of operations. ",
    "Each part of the system tracks its own state independently. ",
    "We're using a log-based approach so we don't lose data if something crashes. ",
    "The type checker catches a lot of bugs at compile time for us. ",
    "We scale out by distributing work across multiple nodes using hashing. ",
    "Data flows through a pipeline of transformations before storage. ",
    "All service-to-service communication is encrypted and authenticated. ",
    "We have logging, tracing, and metrics set up for visibility into the system. ",
    "Reads and writes go through separate paths to optimize for each. ",
    "When one service fails, it doesn't take down the others. ",
    "We can toggle new features on and off without redeploying. ",
    "Query planning uses cost estimates and runtime feedback to choose the best path. ",
    "The team has good test coverage on the critical paths. ",
    "We recently migrated the main database and the transition was mostly smooth. ",
    "Caching is handled at multiple layers to reduce load on the backend. ",
    "We have automated alerts for the key health metrics. ",
]

_PROBE_QUESTIONS = [
    "Can you analyze the potential bottlenecks in this architecture?",
    "What challenges should I expect and how should I prioritize them?",
    "How should I design the system to handle the expected growth?",
    "What are the most critical failure modes I need to plan for?",
    "How would you structure the testing strategy for this?",
    "What monitoring and alerting should I set up from day one?",
    "Where are the biggest opportunities for performance optimization?",
    "How should I design the data model to support these requirements?",
    "What trade-offs should I be aware of with this approach?",
    "How would you handle the migration while maintaining availability?",
]

_TURN2_TEMPLATES = [
    "Can you elaborate on {focus}?",
    "What would be the most important thing to {action}?",
    "How would you handle {concern} in this scenario?",
    "Can you provide a concrete example of {topic}?",
    "What are the main {risk_type} I should watch out for?",
    "How does this compare to {alternative}?",
    "What {metrics} should I track to measure success?",
    "Can you break down the {aspect} in more detail?",
]

_TURN2_FOCUSES = ["the first point you mentioned", "the scaling approach", "the data consistency model",
                   "the failure handling strategy", "the performance characteristics", "the security implications"]
_TURN2_ACTIONS = ["prioritize first", "focus on for the MVP", "address before going to production",
                  "validate with a proof of concept", "get right from the start"]
_TURN2_CONCERNS = ["failures", "data loss", "network partitions", "concurrent access", "backward compatibility", "security threats"]
_TURN2_TOPICS = ["how to implement that", "the deployment pipeline", "the testing approach", "the monitoring setup"]
_TURN2_RISK_TYPES = ["risks", "performance pitfalls", "security vulnerabilities", "operational challenges"]
_TURN2_ALTERNATIVES = ["alternative approaches", "using a different technology stack", "a simpler design"]
_TURN2_METRICS = ["metrics", "KPIs", "SLOs", "performance indicators"]
_TURN2_ASPECTS = ["cost implications", "timeline", "resource requirements", "technical debt impact"]


def _generate_nonce_phrase() -> str:
    """Generate a random contextual phrase that makes probes non-enumerable.

    The combinatorial template space (~576k) is finite and could be pre-generated
    by a determined miner. This nonce injects a random clause with a
    crypto-random identifier, making each probe effectively unique and
    impossible to pre-enumerate.
    """
    nonce = secrets.token_hex(16)  # 32-char hex = 2^128 possibilities (C3-5)
    # Random temporal/organizational context to break template memorization (C3-9)
    year = 2020 + secrets.randbelow(7)
    month = 1 + secrets.randbelow(12)
    day = 1 + secrets.randbelow(28)
    perturbations = [
        f" as of {year}-{month:02d}-{day:02d}",
        f" (updated Q{1 + secrets.randbelow(4)} {year})",
        f" per revision {secrets.token_hex(3)}",
    ]
    suffix = secrets.choice(perturbations)
    templates = [
        f"For context, our project ID is {nonce}.{suffix}",
        f"The internal tracking number is {nonce}.{suffix}",
        f"Our team uses reference code {nonce} for this.{suffix}",
        f"The sprint identifier is {nonce}.{suffix}",
        f"This relates to ticket {nonce}.{suffix}",
        f"We track this under code {nonce}.{suffix}",
    ]
    return secrets.choice(templates)


def _generate_probe_turn1() -> str:
    """Generate a unique long-form turn-1 probe prompt procedurally.

    Uses varied sentence-opening frames so no single prefix pattern can
    fingerprint probes.  Template combinatorial space: 12 frames × 20 domains ×
    15 scenarios × 16 details × 10 questions = 576k variants.  Additionally,
    a crypto-random nonce phrase is injected, making each probe effectively
    unique (2^48 nonce space × 576k templates = practically infinite).
    """
    frame = secrets.choice(_PROBE_FRAMES)
    domain = secrets.choice(_PROBE_DOMAINS)
    scenario = secrets.choice(_PROBE_SCENARIOS)
    detail = secrets.choice(_PROBE_DETAILS)
    question = secrets.choice(_PROBE_QUESTIONS)
    nonce = _generate_nonce_phrase()

    # Fill scenario placeholders with plausible random values
    fillers = {
        "n": secrets.choice(["1,000", "10,000", "50,000", "100,000", "500,000"]),
        "latency": secrets.choice(["sub-millisecond", "single-digit millisecond", "< 50ms", "< 100ms"]),
        "rate": secrets.choice(["15%", "20%", "30%", "50%"]),
        "count": secrets.choice(["3", "5", "8", "12", "20", "50"]),
        "protocol": secrets.choice(["gRPC", "REST", "Kafka events", "NATS messaging"]),
        "size": secrets.choice(["500GB", "2TB", "10TB", "50TB"]),
        "growth": secrets.choice(["1GB", "5GB", "20GB", "100GB"]),
        "time": secrets.choice(["6 months", "1 year", "2 years", "5 years"]),
        "nines": secrets.choice(["99.9%", "99.95%", "99.99%"]),
        "rpo": secrets.choice(["zero", "5 minute", "1 hour"]),
        "budget": secrets.choice(["$5,000", "$20,000", "$50,000", "$100,000"]),
        "regulation": secrets.choice(["GDPR", "HIPAA", "SOC 2", "PCI-DSS"]),
        "regions": secrets.choice(["2", "3", "5"]),
        "target": secrets.choice(["10ms", "50ms", "100ms"]),
        "kloc": secrets.choice(["50", "150", "300", "500"]),
        "old_tech": secrets.choice(["a monolith", "MySQL", "on-prem servers", "legacy Java"]),
        "new_tech": secrets.choice(["microservices", "PostgreSQL", "Kubernetes", "Go services"]),
    }

    filled_scenario = scenario
    for key, val in fillers.items():
        filled_scenario = filled_scenario.replace("{" + key + "}", val, 1)

    return f"{frame.format(domain=domain)} {filled_scenario} {nonce} {detail}{question}"


def _generate_probe_turn2() -> str:
    """Generate a unique turn-2 follow-up prompt procedurally."""
    template = secrets.choice(_TURN2_TEMPLATES)
    fillers = {
        "focus": secrets.choice(_TURN2_FOCUSES),
        "action": secrets.choice(_TURN2_ACTIONS),
        "concern": secrets.choice(_TURN2_CONCERNS),
        "topic": secrets.choice(_TURN2_TOPICS),
        "risk_type": secrets.choice(_TURN2_RISK_TYPES),
        "alternative": secrets.choice(_TURN2_ALTERNATIVES),
        "metrics": secrets.choice(_TURN2_METRICS),
        "aspect": secrets.choice(_TURN2_ASPECTS),
    }
    result = template
    for key, val in fillers.items():
        result = result.replace("{" + key + "}", val, 1)
    return result


@dataclass
class CacheProbeResult:
    """Result of a single KV cache probe."""
    miner_uid: int
    session_id: str
    turn1_ttft_ms: float
    turn2_ttft_ms: float
    ttft_ratio: float           # turn2 / turn1 (lower = better cache)
    cache_score: float          # 0-1 score
    challenge_passed: bool      # Hidden state verification on turn-2
    turn1_input_tokens: int
    turn2_input_tokens: int
    probe_delay_s: float        # Wait between turns
    timestamp: float = field(default_factory=time.time)

    @property
    def has_real_cache(self) -> bool:
        """Conservative assessment: is the cache likely real?"""
        return self.ttft_ratio < CACHE_SPEEDUP_MARGINAL and self.challenge_passed


@dataclass
class MinerCacheProfile:
    """Aggregated cache performance for a miner across an epoch."""
    miner_uid: int
    probe_results: list[CacheProbeResult] = field(default_factory=list)

    @property
    def num_probes(self) -> int:
        return len(self.probe_results)

    @property
    def median_ttft_ratio(self) -> float:
        """Median TTFT ratio (robust to outliers)."""
        if not self.probe_results:
            return 1.0
        ratios = [p.ttft_ratio for p in self.probe_results]
        return float(np.median(ratios))

    @property
    def median_cache_score(self) -> float:
        """Median cache score."""
        if not self.probe_results:
            return 0.0
        scores = [p.cache_score for p in self.probe_results]
        return float(np.median(scores))

    @property
    def challenge_pass_rate(self) -> float:
        """What fraction of turn-2 challenges passed."""
        if not self.probe_results:
            return 0.0
        passed = sum(1 for p in self.probe_results if p.challenge_passed)
        return passed / len(self.probe_results)

    @property
    def cache_efficiency_score(self) -> float:
        """
        Final cache efficiency score for weight calculation.
        Requires minimum probes. Returns a mild penalty (0.35) if insufficient data
        to prevent miners from evading KV cache verification by avoiding probes.
        Only returns neutral (0.5) for miners with zero interactions.
        """
        if self.num_probes == 0:
            return 0.5  # Truly no data — neutral
        if self.num_probes < MIN_PROBES_FOR_SCORE:
            # Miner was probed but not enough times. Mild penalty to discourage
            # evasion tactics (dropping sessions early, refusing multi-turn).
            return 0.35

        # Combine median cache score with challenge pass rate
        # Both must be good for high score
        return self.median_cache_score * self.challenge_pass_rate


def compute_cache_score(ttft_ratio: float) -> float:
    """
    Convert TTFT ratio to a 0-1 score.

    Score mapping:
      ratio <= 0.3  → 1.0 (excellent cache, 70%+ speedup)
      ratio  = 0.5  → 0.75
      ratio  = 0.7  → 0.5
      ratio  = 0.9  → 0.1
      ratio >= 1.0  → 0.0 (no speedup = no cache)
    """
    if ttft_ratio <= CACHE_SPEEDUP_EXCELLENT:
        return 1.0
    if ttft_ratio >= 1.0:
        return 0.0
    # Linear interpolation between excellent (1.0) and none (0.0)
    score = 1.0 - (ttft_ratio - CACHE_SPEEDUP_EXCELLENT) / (1.0 - CACHE_SPEEDUP_EXCELLENT)
    return max(0.0, min(1.0, score))


def generate_probe_pair() -> tuple[str, str, float]:
    """
    Generate a (turn1, turn2, delay) probe pair.

    Uses cryptographic randomness and procedural generation.
    Combinatorial space > 8M unique pairs — infeasible to fingerprint.
    """
    turn1 = _generate_probe_turn1()
    turn2 = _generate_probe_turn2()

    # Random delay between turns (crypto-random millisecond precision)
    delay_range_ms = int((PROBE_DELAY_MAX_S - PROBE_DELAY_MIN_S) * 1000)
    delay_s = PROBE_DELAY_MIN_S + secrets.randbelow(max(delay_range_ms, 1)) / 1000.0

    return turn1, turn2, delay_s


class KVCacheProber:
    """
    Manages KV cache probes across all miners in an epoch.

    Usage:
        prober = KVCacheProber()

        # After sending turn-1 and turn-2 through the gateway:
        prober.record_probe(result)

        # At epoch end:
        for uid in miner_uids:
            profile = prober.get_profile(uid)
            cache_weight_multiplier = 1.0 + profile.cache_efficiency_score * CACHE_WEIGHT_MULTIPLIER
    """

    def __init__(self):
        self._profiles: dict[int, MinerCacheProfile] = {}
        self.total_probes = 0
        self.total_cache_hits = 0  # probes where cache appears real

    def _get_profile(self, miner_uid: int) -> MinerCacheProfile:
        if miner_uid not in self._profiles:
            self._profiles[miner_uid] = MinerCacheProfile(miner_uid=miner_uid)
        return self._profiles[miner_uid]

    def record_probe(self, result: CacheProbeResult):
        """Record a completed cache probe."""
        profile = self._get_profile(result.miner_uid)
        profile.probe_results.append(result)
        self.total_probes += 1
        if result.has_real_cache:
            self.total_cache_hits += 1

        status = "CACHE_HIT" if result.has_real_cache else "CACHE_MISS"
        log.info(
            f"[KV Probe] Miner {result.miner_uid} {status} | "
            f"ratio={result.ttft_ratio:.3f} score={result.cache_score:.3f} "
            f"t1={result.turn1_ttft_ms:.1f}ms t2={result.turn2_ttft_ms:.1f}ms "
            f"challenge={'PASS' if result.challenge_passed else 'FAIL'}"
        )

    def get_profile(self, miner_uid: int) -> MinerCacheProfile:
        """Get cache profile for a miner."""
        return self._get_profile(miner_uid)

    def get_all_profiles(self) -> dict[int, MinerCacheProfile]:
        """Get all miner cache profiles."""
        return dict(self._profiles)

    def get_cache_weight_adjustments(self) -> dict[int, float]:
        """
        Get weight adjustment multipliers for all miners based on cache performance.

        Returns dict of miner_uid → multiplier where:
          1.0 = neutral (insufficient data or average cache)
          > 1.0 = bonus for good cache performance
          < 1.0 = penalty for poor/fake cache
        """
        adjustments = {}
        for uid, profile in self._profiles.items():
            eff = profile.cache_efficiency_score
            # Map efficiency to multiplier:
            # 0.0 → 0.65 (35% penalty for zero cache)
            # 0.5 → 1.0  (neutral)
            # 1.0 → 1.35 (35% bonus for perfect cache)
            adjustments[uid] = 1.0 + (eff - 0.5) * 2 * CACHE_WEIGHT_MULTIPLIER
        return adjustments

    def reset(self):
        """Reset for a new epoch."""
        self._profiles = {}
        self.total_probes = 0
        self.total_cache_hits = 0

    def summary(self) -> dict:
        """Epoch summary for audit logging."""
        miner_summaries = {}
        for uid, profile in self._profiles.items():
            miner_summaries[uid] = {
                "num_probes": profile.num_probes,
                "median_ttft_ratio": profile.median_ttft_ratio,
                "median_cache_score": profile.median_cache_score,
                "challenge_pass_rate": profile.challenge_pass_rate,
                "cache_efficiency_score": profile.cache_efficiency_score,
            }
        return {
            "total_probes": self.total_probes,
            "total_cache_hits": self.total_cache_hits,
            "miners": miner_summaries,
        }
