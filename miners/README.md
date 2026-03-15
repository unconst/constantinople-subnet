# Constantinople Miner Guide

Mine on Constantinople (SN97) — a decentralized LLM inference subnet on Bittensor.

## How Mining Works

Miners serve an LLM (currently Qwen2.5-7B-Instruct) via vLLM and respond to inference requests from the validator gateway. The validator verifies response quality using **hidden-state verification**: it picks a random intermediate transformer layer and token position, computes a reference hidden state, and compares it to the miner's. Honest miners get cosine similarity > 0.99.

## Scoring

Miners are scored on three dimensions:
1. **Throughput** (tokens/sec) — faster is better
2. **Latency** (time-to-first-token) — lower is better
3. **Challenge pass rate** — hidden-state verification must pass consistently

Scores use Bayesian smoothing so a single bad sample won't tank your score. The adaptive challenge rate has a 10% floor with a minimum of 5 challenges per epoch.

## Hardware Requirements

| GPU | Expected TPS | Notes |
|-----|-------------|-------|
| RTX 4090 (24GB) | ~60 TPS | Good entry-level option |
| RTX 5090 (32GB) | ~160 TPS | Best value currently |
| A100 (80GB) | ~200+ TPS | Enterprise option |

Minimum: Any GPU with 24GB+ VRAM that can run Qwen2.5-7B-Instruct via vLLM.

## Quick Start

### 1. Register on SN97

We recommend [agcli](https://github.com/unconst/agcli) (Rust, fast) for wallet and registration. btcli works too.

```bash
# Install agcli (Rust — fast, no Python overhead)
cargo install --git https://github.com/unconst/agcli

# Create a wallet
agcli wallet create --name miner --password YOUR_PASSWORD --yes

# Register on subnet 97 (burn registration)
agcli subnet register-neuron 97 --wallet miner --hotkey default --password YOUR_PASSWORD --yes
```

<details>
<summary>Alternative: using btcli (Python)</summary>

```bash
pip install bittensor
btcli wallet create --wallet-name miner --hotkey default
btcli subnet register --wallet-name miner --hotkey default --netuid 97
```
</details>

### 2. Install Dependencies

```bash
pip install vllm fastapi uvicorn aiohttp numpy pydantic accelerate transformers torch

# RTX 4090 may also need:
pip install nvidia-cusparselt-cu12
```

### 3. Run the Miner

```bash
python vllm_miner.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8091 \
    --gpu-memory-utilization 0.70 \
    --hf-device cpu
```

Key flags:
- `--model`: The model to serve (must match what the validator expects)
- `--port`: Port to expose (must match your registered axon port)
- `--gpu-memory-utilization 0.70`: Leaves room for HF model used in hidden state extraction
- `--hf-device cpu`: Forces the HF reference model to CPU (recommended to avoid GPU OOM)
- `--tensor-parallel-size N`: For multi-GPU setups (e.g., 2 for dual-GPU)

### 4. Set Your Axon

Register your miner's IP and port on-chain so the validator can discover and send requests to you:

```bash
agcli serve axon --netuid 97 --ip YOUR_PUBLIC_IP --port 8091 --wallet miner --hotkey default --password YOUR_PASSWORD
```

Replace `YOUR_PUBLIC_IP` with your server's public IP and `8091` with your miner's port.

<details>
<summary>Alternative: using Python bittensor SDK</summary>

```python
# save as set_axon.py and run: python set_axon.py
import bittensor

wallet = bittensor.wallet(name="miner", hotkey="default")
subtensor = bittensor.subtensor(network="finney")
axon = bittensor.axon(wallet=wallet, ip="YOUR_PUBLIC_IP", port=8091)

success = subtensor.serve_axon(netuid=97, axon=axon)
print("Axon registered!" if success else "Failed — check wallet and registration")
```
</details>

## Architecture

The miner runs two models internally:

1. **vLLM AsyncLLMEngine** — handles all inference (generation) with continuous batching and PagedAttention for high throughput
2. **HuggingFace model** (CPU) — runs a single forward pass over completed sequences to extract hidden states for verification challenges

The HF forward pass is encoding-only (no generation), so it runs in ~1-5 seconds even for long sequences on CPU.

## Endpoints

Your miner exposes these endpoints (the gateway calls them automatically):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/inference` | POST | Generate text, return tokens + hidden state commitments |
| `/inference/stream` | POST | SSE streaming generation |
| `/hidden_state` | POST | Return cached hidden state at a specific (layer, position) |
| `/health` | GET | Health check with model info |

## Tips

- **Keep VRAM headroom**: Set `--gpu-memory-utilization` to 0.70-0.80 to leave room for the HF model
- **Use CPU for HF model**: Always use `--hf-device cpu` to avoid GPU memory conflicts
- **Monitor health**: Check `/health` endpoint to verify both vLLM engine and HF model are loaded
- **Uptime matters**: Consistent uptime improves your score via Bayesian smoothing
- **Model must match**: You **must** run `Qwen/Qwen2.5-7B-Instruct` — the validator computes reference hidden states from this exact model. Running a different model, a quantized variant, or a fine-tune will cause all hidden-state challenges to fail (cosine similarity will be near zero instead of > 0.99)

## Scoring Timeline

- **Epochs are 25 minutes** (1500 seconds). At the end of each epoch, miners are scored and weights are updated.
- **Challenge rate**: Adaptive, 10% floor. You'll see at least 5 challenges per epoch once the validator has enough organic traffic.
- **New miner grace period**: Bayesian smoothing starts you with 2 virtual passes, so you won't be penalized for low sample count early on. Scores stabilize after ~5 real challenges.

## Troubleshooting

**vLLM EngineDeadError**: The vLLM engine worker can crash while FastAPI stays up (returning 500s). Kill the process, verify VRAM is freed (`nvidia-smi`), and restart.

**Low cosine similarity**: If your challenges are failing, ensure you're using `--hf-device cpu` with float32 (default). GPU float16 vs CPU float32 divergence can cause mid-range cosines that get voided.

**OOM on HF model load**: If `device_map="auto"` splits the HF model across GPU+CPU, you'll get errors. Use `--hf-device cpu` to force it all to CPU.

## Support

Join the Constantinople Discord and ask Arbos (the bot) for help. The bot has full knowledge of the subnet architecture and can answer technical questions.

API: `api.constantinople.cloud`
Website: `constantinople.cloud`
