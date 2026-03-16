# Constantinople (SN97)

Decentralized LLM inference on Bittensor. Miners serve language models, validators verify quality using hidden-state verification.

## Architecture

Constantinople uses a challenge-response protocol to verify that miners are running real models:

1. **Gateway** receives inference requests and routes them to miners
2. **Miners** serve Qwen2.5-7B-Instruct via vLLM with hidden state extraction
3. **Validator** challenges miners by requesting hidden states at random (layer, position) pairs and comparing against reference computations
4. **Scoring** combines throughput, latency, and verification pass rate
5. **Epoch summaries** are published to R2 for transparency — anyone can audit the scoring

## Quick Links

- **API**: `api.constantinople.cloud` (OpenAI-compatible)
- **Dataset**: `api.constantinople.cloud/v1/dataset/` (all inference data, downloadable)
- **Website**: `constantinople.cloud`

## Run a Validator

See [validator/README.md](validator/README.md) for full details. Two options:

**Watchtower (no GPU)** — follow published scores:
```bash
docker compose up -d
```

**Full Validator (GPU)** — run independent hidden-state verification:
```bash
pip install -r validator/requirements.txt
python validator/hardened_gateway.py --port 8081 --model Qwen/Qwen2.5-7B-Instruct --netuid 97 --discover
python validator/audit_validator.py --port 8082 --model Qwen/Qwen2.5-7B-Instruct --wallet validator --hotkey default --netuid 97
```

## Run a Miner

See [miners/README.md](miners/README.md) for the full setup guide (registration, PM2, Docker, networking). Requires a GPU (RTX 4090+ recommended).

```bash
pip install -r miners/requirements.txt
python miners/vllm_miner.py --model Qwen/Qwen2.5-7B-Instruct --port 8091 --gpu-memory-utilization 0.70 --hf-device cpu
```

## Incentive Mechanics

- **Throughput** (~40% weight): Tokens per second. Faster miners earn more.
- **Latency** (~20% weight): Time-to-first-token. Lower is better.
- **Verification** (~40% weight): Hidden-state challenge pass rate. Must be consistently honest.
- **Bayesian smoothing**: New miners get the benefit of the doubt (2 virtual passes). Score stabilizes after ~5 challenges.
- **Adaptive challenge rate**: 10% floor, minimum 5 challenges per 25-minute epoch.

## Auto-Updates

Validators running via Docker Compose get automatic updates through Watchtower. When we push a new validator image, Watchtower detects the change and restarts your container with the latest code.

## License

MIT
