# Constantinople Validator (Watchtower)

Lightweight validator for Constantinople (SN97) on Bittensor. **No GPU required.**

The watchtower reads epoch summaries published by the primary validator to a Cloudflare R2 bucket, extracts miner scores, and sets weights on chain. This means you can validate SN97 without running GPU-intensive hidden-state verification yourself.

## How it works

1. The primary validator runs challenges against miners using hidden-state verification
2. Epoch summaries (scores, weights) are published to R2 after each scoring epoch
3. Your watchtower polls R2 for new epochs and sets the same weights on chain
4. Watchtower (Docker) auto-updates when we push new validator code

## Requirements

- A registered validator hotkey on SN97
- Docker and Docker Compose
- R2 read credentials (provided below)

## Quick Start

```bash
# Clone this repo
git clone https://github.com/unconst/constantinople-subnet.git
cd constantinople-subnet

# Create your .env file
cp .env.example .env
# Edit .env with your wallet details and R2 credentials

# Start the validator + auto-updater
docker compose up -d

# Check logs
docker compose logs -f validator
```

## Configuration

Create a `.env` file:

```env
WALLET_NAME=validator
HOTKEY_NAME=default
NETUID=97
NETWORK=finney
# IMPORTANT: Use an absolute path — tilde (~) does NOT expand inside Docker
WALLET_PATH=/root/.bittensor/wallets

# R2 credentials — ask Arbos in Constantinople Discord
R2_ENDPOINT=
R2_BUCKET=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
```

R2 read credentials are available in the Constantinople Discord. Ask Arbos.

> **Important**: `WALLET_PATH` must be an absolute path to your wallets directory on the host machine. Docker does not expand `~`, so use the full path (e.g., `/home/youruser/.bittensor/wallets`).

## Manual Run (without Docker)

```bash
pip install -r validator/requirements.txt

python validator/watchtower.py \
    --wallet validator \
    --hotkey default \
    --netuid 97 \
    --network finney \
    --r2-endpoint YOUR_ENDPOINT \
    --r2-bucket YOUR_BUCKET \
    --r2-access-key YOUR_KEY \
    --r2-secret-key YOUR_SECRET
```

## Auto-Updates

The Docker Compose setup includes [Watchtower](https://containrrr.dev/watchtower/) which automatically pulls new validator images when we push updates to this repo. The pipeline:

1. We push a validator code change to `main`
2. GitHub Actions builds a new Docker image and pushes it to Docker Hub (`thebes1618/constantinople-validator:latest`)
3. Watchtower (running alongside your validator) detects the new image within 5 minutes
4. Your validator container is automatically restarted with the new code

No manual intervention needed.

## GPU Requirements

**None.** The watchtower is CPU-only (~100MB RAM). All GPU-intensive verification is done by the primary validator. You're following its published scores.

## Epoch Timing & Weight Setting

- **Epoch length**: 25 minutes (1500 seconds). At the end of each epoch, the primary validator publishes scores to R2.
- **Watchtower poll interval**: Every 3 minutes (configurable) — checks R2 for new epoch summaries.
- **Weight cooldown**: The watchtower enforces a 20-minute cooldown between `set_weights` calls to respect the on-chain rate limit (~100 blocks).
- **Commit-reveal**: Mainnet SN97 uses commit-reveal for weight setting. The bittensor SDK handles this automatically — your watchtower commits a hash, then reveals the actual weights the next block. This is transparent to you.

## Mainnet Requirements

To actually influence miner incentives with your weights:
- Your validator hotkey must be **registered on SN97** (`btcli subnet register`)
- Weights are set via **commit-reveal** (handled automatically by the SDK)
- The `weights_rate_limit` on chain is ~100 blocks — the watchtower respects this with its built-in cooldown
