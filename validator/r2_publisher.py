#!/usr/bin/env python3
"""
R2 Audit Publisher — Publishes all requests/responses to Cloudflare R2 for transparency.

All inference interactions are logged:
- Request prompt + params
- Miner response + timings
- Hidden state challenge result
- Points awarded
- Whether request was organic or synthetic

Anyone can read the R2 bucket to audit the validator's scoring.

For the PoC, this supports both real R2 (via boto3/S3-compatible API) and
a local file-based mock for testing without R2 credentials.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("r2-publisher")
log.propagate = False


class AuditRecord:
    """A single auditable interaction record."""

    def __init__(
        self,
        request_id: str,
        miner_uid: int,
        miner_hotkey: str,
        is_synthetic: bool,
        prompt: str,
        response_text: str,
        ttft_ms: float,
        tokens_per_sec: float,
        input_tokens: int,
        output_tokens: int,
        challenge_layer: int | None = None,
        challenge_token_pos: int | None = None,
        cosine_sim: float | None = None,
        challenge_latency_ms: float | None = None,
        challenge_passed: bool | None = None,
        speed_score: float = 0.0,
        verification_score: float = 0.0,
        quality_score: float = 1.0,
        points_awarded: float = 0.0,
        messages: list | None = None,
        **kwargs,
    ):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.request_id = request_id
        self.miner_uid = miner_uid
        self.miner_hotkey = miner_hotkey
        self.is_synthetic = is_synthetic
        self.prompt = prompt
        self.response_text = response_text
        self.ttft_ms = ttft_ms
        self.tokens_per_sec = tokens_per_sec
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.challenge_layer = challenge_layer
        self.challenge_token_pos = challenge_token_pos
        self.cosine_sim = cosine_sim
        self.challenge_latency_ms = challenge_latency_ms
        self.challenge_passed = challenge_passed
        self.speed_score = speed_score
        self.verification_score = verification_score
        self.quality_score = quality_score
        self.points_awarded = points_awarded
        self.messages = messages

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "type": "synthetic" if self.is_synthetic else "organic",
            "miner_uid": self.miner_uid,
            "miner_hotkey": self.miner_hotkey,
            # C5 H5-13: Redact prompts/responses for user privacy — hash for linkability
            "prompt": hashlib.sha256((self.prompt or "").encode()).hexdigest()[:16],
            "response": f"[{len(self.response_text or '')} chars]",
            "ttft_ms": round(self.ttft_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "speed_score": round(self.speed_score, 4),
            "verification_score": round(self.verification_score, 4),
            "quality_score": round(self.quality_score, 4),
            "points_awarded": round(self.points_awarded, 4),
        }
        if self.messages:
            d["messages"] = "[redacted]"  # C5 H5-13: Don't publish user messages
        if self.challenge_passed is not None:
            # C5 H5-13: Only publish pass/fail + rounded cosine — omit layer/token_pos
            d["challenge"] = {
                "passed": self.challenge_passed,
                "cosine_sim": round(self.cosine_sim, 2) if self.cosine_sim is not None else None,
            }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class R2Publisher:
    """
    Publishes audit records to R2 (or local files for testing).

    In production, uses boto3 with S3-compatible API to write to Cloudflare R2.
    For PoC testing, writes JSONL files to a local directory.
    """

    def __init__(
        self,
        bucket_name: str = "inference-subnet-audit",
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        local_dir: str | None = None,
    ):
        self.bucket_name = bucket_name
        self.local_dir = local_dir
        self.records_published = 0
        self._client = None

        if endpoint_url and access_key and secret_key:
            # Real R2 mode
            try:
                import boto3
                self._client = boto3.client(
                    "s3",
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                )
                self._mode = "r2"
                log.info(f"R2Publisher: R2 mode → {bucket_name}")
            except ImportError:
                log.warning("boto3 not installed, falling back to local mode")
                self.local_dir = local_dir or "/tmp/r2-audit"
                Path(self.local_dir).mkdir(parents=True, exist_ok=True)
                self._mode = "local"
        elif local_dir:
            # Explicit local dir
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            self._mode = "local"
            log.info(f"R2Publisher: local mode → {local_dir}")
        else:
            # Default: local mode
            self.local_dir = "/tmp/r2-audit"
            Path(self.local_dir).mkdir(parents=True, exist_ok=True)
            self._mode = "local"
            log.info(f"R2Publisher: no R2 credentials, local mode → {self.local_dir}")

    def publish(self, record: AuditRecord):
        """Publish a single audit record."""
        if self._mode == "r2":
            self._publish_r2(record)
        else:
            self._publish_local(record)
        self.records_published += 1

    def _publish_r2(self, record: AuditRecord):
        """Upload audit record to R2 bucket."""
        now = datetime.now(timezone.utc)
        key = f"audit/{now.strftime('%Y/%m/%d/%H')}/{record.request_id}.json"
        try:
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=record.to_json().encode(),
                ContentType="application/json",
            )
        except Exception as e:
            log.error(f"R2 publish failed: {e}")
            # Fallback to local
            self._publish_local(record)

    def _publish_local(self, record: AuditRecord):
        """Write audit record to local JSONL file."""
        now = datetime.now(timezone.utc)
        day_dir = Path(self.local_dir) / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        filepath = day_dir / f"hour-{now.strftime('%H')}.jsonl"
        with open(filepath, "a") as f:
            f.write(record.to_json() + "\n")

    def publish_epoch_summary(self, summary: dict):
        """Publish epoch summary (weights, stats)."""
        key = f"epochs/{summary['epoch']:06d}.json"
        data = json.dumps(summary, indent=2)

        if self._mode == "r2":
            try:
                self._client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=data.encode(),
                    ContentType="application/json",
                )
            except Exception as e:
                log.error(f"R2 epoch publish failed: {e}")
        else:
            epoch_dir = Path(self.local_dir) / "epochs"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            with open(epoch_dir / f"{summary['epoch']:06d}.json", "w") as f:
                f.write(data)

        log.info(f"Published epoch {summary['epoch']} summary")
