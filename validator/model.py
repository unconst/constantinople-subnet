"""
Mock model for inference subnet PoC.

Simulates a transformer model that produces deterministic hidden states
given the same weights + input. In production, this gets replaced by
vLLM with hidden state extraction hooks.

The mock uses a seeded PRNG to simulate model weights, then produces
hidden states by hashing (input_tokens, layer_index, token_position)
through those "weights". This gives us:
- Deterministic outputs for same input
- Different outputs for different inputs/layers/positions
- A realistic hidden state shape (hidden_dim=4096)
"""

import hashlib
import struct
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration matching common architectures."""
    name: str = "mock-7b"
    num_layers: int = 28
    hidden_dim: int = 3584
    vocab_size: int = 32000
    max_seq_len: int = 4096


class MockModel:
    """
    Deterministic mock transformer model.

    Produces hidden states that are:
    - Deterministic: same input → same output (across calls)
    - Input-dependent: different prompts → completely different states
    - Layer-dependent: different layers → different states
    - Position-dependent: different token positions → different states

    In production, replace with vLLM model + hidden state hooks.
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        # Simulated "model weights" — fixed seed for determinism
        rng = np.random.RandomState(42)
        self.weight_seed = rng.bytes(32)

    def tokenize(self, text: str) -> list[int]:
        """Simple tokenizer: hash each word to a token ID."""
        words = text.split()
        tokens = []
        for w in words:
            h = hashlib.sha256(w.encode()).digest()
            token_id = int.from_bytes(h[:4], 'little') % self.config.vocab_size
            tokens.append(token_id)
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        """Reverse tokenization (mock: just return token IDs as words)."""
        return " ".join(f"tok_{t}" for t in tokens)

    def _compute_hidden_state(self, input_tokens: list[int], layer: int, position: int) -> np.ndarray:
        """
        Compute the hidden state at a specific (layer, position).

        Uses a deterministic hash chain: hash(weights + tokens[:position+1] + layer)
        This simulates a real transformer where each position's hidden state
        depends on all prior tokens and the layer's weights.
        """
        # Build a seed from: model weights + input context up to position + layer
        ctx_tokens = input_tokens[:position + 1]
        ctx_bytes = struct.pack(f'{len(ctx_tokens)}i', *ctx_tokens)
        layer_bytes = struct.pack('i', layer)

        seed_material = self.weight_seed + ctx_bytes + layer_bytes
        seed_hash = hashlib.sha256(seed_material).digest()

        # Use the hash as a seed to generate a hidden_dim-sized vector
        rng = np.random.RandomState(int.from_bytes(seed_hash[:4], 'little'))
        hidden_state = rng.randn(self.config.hidden_dim).astype(np.float32)

        # Normalize to unit sphere (realistic for transformer hidden states)
        hidden_state /= np.linalg.norm(hidden_state)

        return hidden_state

    def generate(self, prompt: str, max_tokens: int = 64) -> dict:
        """
        Run "inference": tokenize, compute hidden states, generate tokens.

        Returns:
            dict with keys:
                - input_tokens: tokenized prompt
                - output_tokens: generated tokens
                - hidden_states: {layer: {position: np.ndarray}} for ALL layers/positions
                - text: generated text
        """
        input_tokens = self.tokenize(prompt)

        # Generate output tokens (mock: deterministic from input)
        gen_seed = hashlib.sha256(
            self.weight_seed + struct.pack(f'{len(input_tokens)}i', *input_tokens)
        ).digest()
        gen_rng = np.random.RandomState(int.from_bytes(gen_seed[:4], 'little'))
        output_tokens = [int(gen_rng.randint(0, self.config.vocab_size)) for _ in range(max_tokens)]

        all_tokens = input_tokens + output_tokens

        # Compute hidden states for all layers at all positions
        # In production, these come from vLLM's forward pass hooks
        hidden_states = {}
        for layer in range(self.config.num_layers):
            hidden_states[layer] = {}
            for pos in range(len(all_tokens)):
                hidden_states[layer][pos] = self._compute_hidden_state(all_tokens, layer, pos)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "all_tokens": all_tokens,
            "hidden_states": hidden_states,
            "text": self.detokenize(output_tokens),
        }

    def compute_hidden_state_at(self, tokens: list[int], layer: int, position: int) -> np.ndarray:
        """
        Compute a single hidden state for verification.

        This is what the validator calls — it only needs to compute
        the hidden state at ONE (layer, position), not the full forward pass.
        In production, this runs a partial forward pass.
        """
        return self._compute_hidden_state(tokens, layer, position)
