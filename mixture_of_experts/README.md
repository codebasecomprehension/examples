# PyTorch Mixture-of-Experts (MoE) Implementation

A simple, production-ready implementation of Sparse Mixture-of-Experts language models using PyTorch.

## Features

- Mixture-of-Experts architecture with load-balanced routing
- Rotary Position Embeddings (RoPE)
- Flash Attention via PyTorch's SDPA
- SwiGLU activation functions
- Simple training loop

## Requirements

```bash
pip install torch
from mixture_of_experts import MoEConfig, MoELanguageModel

config = MoEConfig(
    hidden_size=1024,
    num_experts=8,
    num_experts_per_tok=2,
)

model = MoELanguageModel(config)
