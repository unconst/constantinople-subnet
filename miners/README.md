# Constantinople Miner Guide

Mine on Constantinople (SN97) — a decentralized LLM inference subnet on Bittensor.

## How Mining Works

Miners serve an LLM (currently Qwen2.5-7B-Instruct) via vLLM and respond to inference requests from the validator gateway. The validator verifies response quality using **hidden-state verification**: it picks a random intermediate transformer layer and token position, computes a reference hidden state, and compares it to the miner's. Honest miners get cosine similarity > 0.99.

## Scoring

Miners are scored on three dimensions:
1. **Throughput** (tokens/sec) — faster is better, bonus kicks in above 50 TPS
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

```bash
pip install bittensor

# Create a coldkey and hotkey
btcli wallet create --wallet-name miner --hotkey default

# Fund your wallet with TAO (you need TAO for registration)

# Register on subnet 97
btcli subnet register --wallet-name miner --hotkey default --netuid 97
```

### 2. Clone the Repo and Install

```bash
git clone https://github.com/unconst/constantinople-subnet.git
cd constantinople-subnet/miners

pip install -r requirements.txt

# RTX 4090 may also need:
pip install nvidia-cusparselt-cu12
```

### 3. Test Locally

Before registering on-chain, verify your miner starts and responds correctly:

```bash
python vllm_miner.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8091 \
    --gpu-memory-utilization 0.70 \
    --hf-device cpu
```

Wait for the model to load (~2-3 minutes on RTX 4090), then test:

```bash
# Check health
curl http://localhost:8091/health

# Test inference
curl -X POST http://localhost:8091/inference \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

You should see a JSON response with generated text and token data.

### 4. Set Your Axon (Register IP/Port)

Register your miner's IP and port on-chain so the validator can discover you:

```bash
btcli stake serve-axon \
    --wallet-name miner \
    --hotkey default \
    --netuid 97 \
    --ip YOUR_PUBLIC_IP \
    --port 8091
```

Replace `YOUR_PUBLIC_IP` with your server's public IP.

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

### 5. Confirm the Validator Can Reach You

After setting your axon, verify connectivity from the outside:

```bash
# From a different machine:
curl http://YOUR_PUBLIC_IP:8091/health
```

If this fails, check your firewall (see Networking section below).

## Production Setup with PM2

[PM2](https://pm2.keymetrics.io/) keeps your miner running and auto-restarts it on crashes.

```bash
# Install PM2
npm install -g pm2

# Start the miner
pm2 start vllm_miner.py --name miner --interpreter python3 -- \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8091 \
    --gpu-memory-utilization 0.70 \
    --hf-device cpu

# Save the process list so it survives reboots
pm2 save
pm2 startup

# Useful commands
pm2 status              # Check if miner is running
pm2 logs miner          # View logs
pm2 restart miner       # Restart the miner
pm2 monit               # Live monitoring dashboard
```

### PM2 Ecosystem File (Alternative)

Create `ecosystem.config.js` for more control:

```javascript
module.exports = {
  apps: [{
    name: 'miner',
    script: 'vllm_miner.py',
    interpreter: 'python3',
    args: '--model Qwen/Qwen2.5-7B-Instruct --port 8091 --gpu-memory-utilization 0.70 --hf-device cpu',
    max_restarts: 10,
    restart_delay: 5000,
    log_date_format: 'YYYY-MM-DD HH:mm:ss',
  }]
};
```

Then: `pm2 start ecosystem.config.js`

## Docker Setup

If you prefer containers:

```dockerfile
# Dockerfile.miner
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY miners/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt nvidia-cusparselt-cu12

COPY miners/vllm_miner.py .

EXPOSE 8091

CMD ["python3", "vllm_miner.py", \
     "--model", "Qwen/Qwen2.5-7B-Instruct", \
     "--port", "8091", \
     "--gpu-memory-utilization", "0.70", \
     "--hf-device", "cpu"]
```

Build and run:

```bash
# Build from the repo root
docker build -f Dockerfile.miner -t constantinople-miner .

# Run with GPU access
docker run -d --gpus all \
    --name miner \
    -p 8091:8091 \
    --restart unless-stopped \
    constantinople-miner
```

## Networking

Your miner must be reachable from the public internet on the port you register.

**Firewall**: Open the miner port (default 8091):
```bash
# UFW (Ubuntu)
sudo ufw allow 8091/tcp

# iptables
sudo iptables -A INPUT -p tcp --dport 8091 -j ACCEPT
```

**Cloud providers**: Ensure your security group / firewall rules allow inbound TCP on your miner port.

**NAT / Port forwarding**: If behind NAT, forward the external port to your machine's internal IP.

**Verify**: `curl http://YOUR_PUBLIC_IP:8091/health` from an external machine should return JSON with `"status": "ok"`.

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

## Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Model to serve (must be `Qwen/Qwen2.5-7B-Instruct`) |
| `--port` | 8091 | Port to expose (must match registered axon port) |
| `--gpu-memory-utilization` | 0.90 | VRAM fraction for vLLM. Use 0.70 to leave room for HF model |
| `--hf-device` | auto | Device for HF model. Use `cpu` on 24GB GPUs to avoid OOM |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism (multi-GPU setups) |
| `--max-model-len` | 4096 | Maximum sequence length |

## Tips

- **Keep VRAM headroom**: Set `--gpu-memory-utilization` to 0.70-0.80 to leave room for the HF model
- **Use CPU for HF model**: Always use `--hf-device cpu` to avoid GPU memory conflicts on 24GB cards
- **Monitor health**: Check `/health` endpoint to verify both vLLM engine and HF model are loaded
- **Uptime matters**: Consistent uptime improves your score via Bayesian smoothing
- **Model must match**: You **must** run `Qwen/Qwen2.5-7B-Instruct` — the validator computes reference hidden states from this exact model. Running a different model, a quantized variant, or a fine-tune will cause all hidden-state challenges to fail

## Scoring Timeline

- **Epochs are ~25 minutes** (1500 seconds ± jitter). At the end of each epoch, miners are scored and weights are committed on-chain.
- **First incentive**: You should see your first weight within 1-2 epochs (~30-50 minutes) after the gateway discovers your miner.
- **Challenge rate**: Adaptive, 10% floor. You'll see at least 5 challenges per epoch once the validator has enough traffic.
- **New miner grace period**: Bayesian smoothing starts you with virtual passes, so you won't be penalized for low sample count early on.

## Troubleshooting

**Miner not receiving requests**: Check that your axon IP and port are correctly registered on-chain. The gateway discovers miners via the metagraph every few minutes. Verify your port is open from the outside.

**vLLM EngineDeadError**: The vLLM engine worker can crash while FastAPI stays up (returning 500s). Kill the process, verify VRAM is freed (`nvidia-smi`), and restart.

**Low cosine similarity**: If your challenges are failing, ensure you're using `--hf-device cpu` with float32 (default). GPU float16 vs CPU float32 divergence can cause mid-range cosines that get voided (not failed — voided results don't hurt your score).

**OOM on HF model load**: If `device_map="auto"` splits the HF model across GPU+CPU, you'll get errors. Use `--hf-device cpu` to force it all to CPU.

**`nvidia-cusparselt-cu12` error**: RTX 4090 requires this library. Install with: `pip install nvidia-cusparselt-cu12`

**Model download slow**: Qwen2.5-7B-Instruct is ~15GB. First run downloads from HuggingFace. No token needed (model is fully open).

## Support

- **Discord**: Join the Constantinople Discord channel and ask for help
- **GitHub**: https://github.com/unconst/constantinople-subnet
- **API**: `api.constantinople.cloud` (OpenAI-compatible)
