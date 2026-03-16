# Constantinople Validator

Validator code for Constantinople (SN97) on Bittensor. Two deployment options:

1. **Watchtower** (no GPU) — follows published epoch scores from the primary validator
2. **Full Validator** (GPU required) — runs hidden-state verification independently

## Architecture

The full validator consists of:

| Component | File | Description |
|-----------|------|-------------|
| **Gateway** | `hardened_gateway.py` | Receives inference requests, routes to miners, runs inline challenges |
| **Auditor** | `audit_validator.py` | Async hidden-state verification, weight setting, epoch publishing |
| **Scoring** | `hardened_scoring.py` | Combines throughput, latency, and verification pass rate into miner scores |
| **Challenge Engine** | `challenge_engine.py` | Generates random (layer, position) challenges for hidden-state verification |
| **KV Cache Prober** | `kv_cache_prober.py` | Multi-turn KV cache consistency probes |
| **Collusion Detector** | `collusion_detector.py` | Detects output correlation between miners (Sybil/relay detection) |
| **R2 Publisher** | `r2_publisher.py` | Publishes audit records and epoch summaries to Cloudflare R2 |
| **Model** | `model.py` | Mock model for testing (not used in production) |
| **Watchdog** | `watchdog.py` | Process health monitor |
| **Watchtower** | `watchtower.py` | Lightweight weight follower (no GPU) |

## Anti-Spoofing Protections

Constantinople prevents miners from impersonating other miners (axon spoofing):

1. **Hotkey signature verification** — Miner responses must include `X-Miner-Hotkey` + `X-Miner-Signature` (HMAC-SHA256). Verified against on-chain hotkey for that UID. Can't forge without the private key.
2. **Nonce-binding** — Per-UID nonce detects relay/forwarding attacks.
3. **RTT timing** — Relay adds detectable latency overhead.
4. **Cross-miner output correlation** — Collusion detector flags identical output pairs.
5. **Hotkey Sybil tracking** — Penalties follow the hotkey across UID changes.

Signature enforcement is controlled by `REQUIRE_MINER_SIGNATURES` (currently in grace period — warnings only).

## Option 1: Watchtower (No GPU)

The watchtower reads epoch summaries published by the primary validator to R2, extracts miner scores, and sets the same weights on chain. No GPU-intensive verification needed.

### Quick Start

```bash
git clone https://github.com/unconst/constantinople-subnet.git
cd constantinople-subnet

cp .env.example .env
# Edit .env with your wallet details and R2 credentials

docker compose up -d
docker compose logs -f validator
```

### Configuration

```env
WALLET_NAME=validator
HOTKEY_NAME=default
NETUID=97
NETWORK=finney
WALLET_PATH=/root/.bittensor/wallets   # Must be absolute path

# R2 credentials — ask Arbos in Constantinople Discord
R2_ENDPOINT=
R2_BUCKET=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
```

### Manual Run

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

## Option 2: Full Validator (GPU Required)

Run the gateway + auditor to perform independent hidden-state verification.

### Requirements

- GPU with 24GB+ VRAM (RTX 4090+) for reference model
- Or CPU with 32GB+ RAM (float32 mode, slower but works)
- Registered validator hotkey on SN97

### Installation

```bash
pip install -r validator/requirements.txt
```

### Run the Gateway

The gateway handles inference routing and inline challenges:

```bash
python validator/hardened_gateway.py \
    --port 8081 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --netuid 97 \
    --network finney \
    --discover \
    --max-context-tokens 8192
```

Key flags:
- `--discover`: Auto-discover miners from the metagraph
- `--miners HOST:PORT ...`: Manually specify miners (space-separated)
- `--challenge-rate 0.2`: Fraction of requests to challenge (default 20%)
- `--max-context-tokens 8192`: Context window limit

### Run the Auditor

The auditor performs async verification and sets weights:

```bash
python validator/audit_validator.py \
    --port 8082 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --wallet validator \
    --hotkey default \
    --netuid 97 \
    --network finney \
    --device cpu
```

Key flags:
- `--device cpu`: Use CPU for reference model (float32, no GPU needed but slower)
- `--device cuda`: Use GPU for reference model (float16, faster)
- `--epoch-length 1500`: Scoring epoch in seconds (default 25 min)

## Auto-Updates (Watchtower Mode)

Docker Compose includes [Watchtower](https://containrrr.dev/watchtower/) for automatic container updates:

1. We push validator code to `main`
2. GitHub Actions builds + pushes a new Docker image
3. Your Watchtower detects the new image within 5 minutes
4. Container auto-restarts with the latest code

## Epoch Timing & Weight Setting

- **Epoch length**: 25 minutes (1500 seconds)
- **Weight cooldown**: Respects on-chain `weights_rate_limit` (~100 blocks)
- **Commit-reveal**: Mainnet SN97 uses commit-reveal (SDK handles automatically)

## GPU Requirements

| Mode | GPU | RAM | Notes |
|------|-----|-----|-------|
| Watchtower | None | ~100MB | CPU-only, follows published scores |
| Full (GPU) | RTX 4090+ | 24GB VRAM | float16 reference model |
| Full (CPU) | None | 32GB+ RAM | float32, ~5-10s per challenge |
