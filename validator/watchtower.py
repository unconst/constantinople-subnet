#!/usr/bin/env python3
"""
Constantinople Watchtower Validator — Lightweight weight-follower for SN97.

No GPU required. Reads epoch summaries published by the primary validator
to an R2 bucket, extracts miner scores, and sets weights on chain.

This allows anyone to run a validator that follows the primary validator's
scoring without needing GPU resources for hidden-state verification.

Usage:
    python watchtower.py --wallet validator --hotkey default --netuid 97

Docker:
    docker compose up -d
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("watchtower")

# ── Config ────────────────────────────────────────────────────────────────────

# Public R2 endpoint for reading epoch summaries
DEFAULT_R2_ENDPOINT = os.environ.get("R2_ENDPOINT", "")
DEFAULT_R2_BUCKET = os.environ.get("R2_BUCKET", "affine")
DEFAULT_R2_ACCESS_KEY = os.environ.get("R2_ACCESS_KEY_ID", "")
DEFAULT_R2_SECRET_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")

EPOCH_CHECK_INTERVAL = 120  # Check for new epochs every 2 minutes
WEIGHT_SET_COOLDOWN = 1200  # Minimum 20 min between weight sets (chain rate limit)


# ── R2 Epoch Reader ──────────────────────────────────────────────────────────

class EpochReader:
    """Reads epoch summaries from R2 bucket."""

    def __init__(self, endpoint_url: str, bucket: str, access_key: str, secret_key: str):
        import boto3
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.bucket = bucket
        self.last_epoch = -1

    def list_recent_epochs(self, limit: int = 10) -> list[int]:
        """List the most recent epoch numbers."""
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket, Prefix="epochs/", MaxKeys=1000,
            )
            if "Contents" not in response:
                return []
            keys = [obj["Key"] for obj in response["Contents"]]
            epochs = []
            for k in keys:
                try:
                    # epochs/000042.json -> 42
                    num = int(k.split("/")[-1].replace(".json", ""))
                    epochs.append(num)
                except (ValueError, IndexError):
                    continue
            return sorted(epochs)[-limit:]
        except Exception as e:
            log.error(f"Failed to list epochs: {e}")
            return []

    def read_epoch(self, epoch_num: int) -> dict | None:
        """Read a specific epoch summary."""
        key = f"epochs/{epoch_num:06d}.json"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            data = json.loads(response["Body"].read().decode())
            return data
        except Exception as e:
            log.error(f"Failed to read epoch {epoch_num}: {e}")
            return None

    def get_latest_epoch(self) -> dict | None:
        """Get the most recent epoch summary."""
        epochs = self.list_recent_epochs(limit=1)
        if not epochs:
            return None
        return self.read_epoch(epochs[-1])


# ── Chain Weight Setter ──────────────────────────────────────────────────────

async def set_weights_on_chain(
    wallet_name: str,
    hotkey: str,
    netuid: int,
    network: str,
    wallet_path: str,
    weights: dict[int, float],
) -> bool:
    """Set weights on chain via subprocess (avoids blocking asyncio)."""
    if not weights:
        log.warning("No weights to set")
        return False

    uids = list(weights.keys())
    weight_values = [weights[uid] for uid in uids]

    script = f"""
import sys
import numpy as np
try:
    import bittensor as bt
    wallet = bt.Wallet(name="{wallet_name}", hotkey="{hotkey}", path="{wallet_path}")
    sub = bt.Subtensor(network="{network}")
    response = sub.set_weights(
        wallet=wallet,
        netuid={netuid},
        uids=np.array({uids}, dtype=np.int64),
        weights=np.array({weight_values}, dtype=np.float32),
        wait_for_finalization=True,
        period=None,
    )
    success = response[0] if isinstance(response, tuple) else bool(response)
    if success:
        print(f"OK:{{response}}")
    else:
        msg = response[1] if isinstance(response, tuple) and len(response) > 1 else str(response)
        print(f"FAIL:{{msg}}", file=sys.stderr)
        sys.exit(2)
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
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode == 0:
            log.info(f"Weights set successfully: {stdout.decode().strip()}")
            return True
        else:
            err = stderr.decode().strip()
            log.error(f"Weight set failed: {err}")
            return False
    except asyncio.TimeoutError:
        log.error("Weight set timed out (120s)")
        try:
            proc.kill()
        except Exception:
            pass
        return False
    except Exception as e:
        log.error(f"Weight set error: {e}")
        return False


# ── Main Loop ────────────────────────────────────────────────────────────────

async def run_watchtower(args):
    """Main watchtower loop: poll R2 for new epochs, set weights."""
    log.info(f"Starting watchtower for SN{args.netuid} on {args.network}")
    log.info(f"Wallet: {args.wallet}/{args.hotkey}")
    log.info(f"R2 bucket: {args.r2_bucket}")

    reader = EpochReader(
        endpoint_url=args.r2_endpoint,
        bucket=args.r2_bucket,
        access_key=args.r2_access_key,
        secret_key=args.r2_secret_key,
    )

    last_epoch_processed = -1
    last_weight_set_time = 0.0

    while True:
        try:
            # Check for new epochs
            epochs = reader.list_recent_epochs(limit=5)
            if not epochs:
                log.info("No epochs found in R2 yet, waiting...")
                await asyncio.sleep(EPOCH_CHECK_INTERVAL)
                continue

            latest = epochs[-1]
            if latest <= last_epoch_processed:
                log.debug(f"No new epochs (latest={latest}, processed={last_epoch_processed})")
                await asyncio.sleep(EPOCH_CHECK_INTERVAL)
                continue

            log.info(f"New epoch detected: {latest}")
            epoch_data = reader.read_epoch(latest)
            if not epoch_data:
                log.error(f"Could not read epoch {latest}")
                await asyncio.sleep(EPOCH_CHECK_INTERVAL)
                continue

            # Extract weights from epoch data
            weights_raw = epoch_data.get("weights", {})
            if not weights_raw:
                log.warning(f"Epoch {latest} has no weights")
                last_epoch_processed = latest
                await asyncio.sleep(EPOCH_CHECK_INTERVAL)
                continue

            # Convert to {uid: weight} dict
            weights = {}
            for uid_str, weight in weights_raw.items():
                try:
                    uid = int(uid_str)
                    weights[uid] = float(weight)
                except (ValueError, TypeError):
                    continue

            if not weights:
                log.warning(f"No valid weights in epoch {latest}")
                last_epoch_processed = latest
                await asyncio.sleep(EPOCH_CHECK_INTERVAL)
                continue

            log.info(f"Epoch {latest} weights: {weights}")

            # Check cooldown
            elapsed = time.time() - last_weight_set_time
            if elapsed < WEIGHT_SET_COOLDOWN:
                wait = WEIGHT_SET_COOLDOWN - elapsed
                log.info(f"Weight set cooldown: waiting {wait:.0f}s")
                await asyncio.sleep(wait)

            # Set weights on chain
            success = await set_weights_on_chain(
                wallet_name=args.wallet,
                hotkey=args.hotkey,
                netuid=args.netuid,
                network=args.network,
                wallet_path=args.wallet_path,
                weights=weights,
            )

            if success:
                last_weight_set_time = time.time()
                log.info(f"Successfully set weights for epoch {latest}")
            else:
                log.error(f"Failed to set weights for epoch {latest}")

            last_epoch_processed = latest

        except Exception as e:
            log.error(f"Watchtower loop error: {e}")

        await asyncio.sleep(EPOCH_CHECK_INTERVAL)


def main():
    parser = argparse.ArgumentParser(description="Constantinople Watchtower Validator")
    parser.add_argument("--wallet", default="validator", help="Bittensor wallet name")
    parser.add_argument("--hotkey", default="default", help="Hotkey name")
    parser.add_argument("--netuid", type=int, default=97, help="Subnet UID")
    parser.add_argument("--network", default="finney", help="Bittensor network")
    parser.add_argument("--wallet-path", default=os.path.expanduser("~/.bittensor/wallets"),
                        help="Path to wallet directory")
    parser.add_argument("--r2-endpoint", default=DEFAULT_R2_ENDPOINT,
                        help="R2 S3-compatible endpoint URL")
    parser.add_argument("--r2-bucket", default=DEFAULT_R2_BUCKET, help="R2 bucket name")
    parser.add_argument("--r2-access-key", default=DEFAULT_R2_ACCESS_KEY, help="R2 access key")
    parser.add_argument("--r2-secret-key", default=DEFAULT_R2_SECRET_KEY, help="R2 secret key")
    args = parser.parse_args()

    asyncio.run(run_watchtower(args))


if __name__ == "__main__":
    main()
