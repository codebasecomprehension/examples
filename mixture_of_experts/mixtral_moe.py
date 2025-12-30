"""
PyTorch Mixture-of-Experts (MoE) Implementation

A simple, working implementation of Sparse Mixture-of-Experts for language models.

This code demonstrates:
- Expert routing with load balancing
- MoE feed-forward layers
- Basic training loop
- No external dependencies (only torch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MoEConfig:
    """Configuration for MoE model."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_hidden_layers: int = 12
    num_experts: int = 32
    expert_capacity: int = 128
    num_experts_per_tok: int = 2
    learning_rate: float = 3e-4
    max_seq_length: int = 2048


# ============================================================================
# RMSNorm - Simple and Effective
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., hidden_size)
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).to(x.dtype)


# ============================================================================
# Rotary Position Embedding (RoPE)
# ============================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for attention."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for positions
        self._cos_cached: Optional[torch.Tensor] = None
        self._sin_cached: Optional[torch.Tensor] = None
    
    def forward(self, seq_len: int, device: torch.device) -> tuple:
        """Return cos and sin for given sequence length."""
        if self._cos_cached is None or self._cos_cached.size(0) < seq_len:
            positions = torch.arange(seq_len, device=device).float()
            freqs = positions[:, None] * self.inv_freq[None, :]  # (seq_len, dim//2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
            self._cos_cached = emb.cos()[None, None, :, :].to(torch.float32)  # (1,1,seq_len,dim)
            self._sin_cached = emb.sin()[None, None, :, :].to(torch.float32)
        
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """Apply rotary embedding to query and key."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_rope = q * cos + rotate_half(q) * sin
    k_rope = k * cos + rotate_half(k) * sin
    return q_rope, k_rope


# ============================================================================
# Multi-Head Attention
# ============================================================================

class Attention(nn.Module):
    """Multi-head attention with RoPE and Flash Attention support."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_length
        )
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.hidden_size, dim=-1)
        
        # Reshape for heads: (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        if position_ids is not None:
            cos, sin = self.rope(seq_len, x.device)
            q, k = apply_rope(q, k, cos, sin)
        
        # Scaled dot-product attention (Flash Attention)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=attention_mask is None
        )
        
        # Reshape and output projection
        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn)


# ============================================================================
# Expert Router
# ============================================================================

class ExpertRouter(nn.Module):
    """Router that assigns tokens to top-k experts."""
    
    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Route tokens to top-k experts.
        
        Args:
            x: (batch, seq_len, hidden_size)
        
        Returns:
            Dict with routing weights, indices, and auxiliary loss
        """
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.hidden_size)  # (batch*seq_len, hidden_size)
        
        # Get routing logits
        logits = self.gate(x_flat)  # (batch*seq_len, num_experts)
        
        # Get top-k experts
        routing_weights, routing_indices = torch.topk(
            F.softmax(logits, dim=-1),
            k=self.num_experts_per_tok,
            dim=-1
        )
        
        # Normalize weights
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # Auxiliary loss for load balancing
        # Encourages uniform distribution of tokens across experts
        expert_mask = F.one_hot(routing_indices, num_classes=self.num_experts).float()
        expert_mask = expert_mask.sum(dim=1)  # (batch*seq_len, num_experts)
        
        # Loss: variance in expert usage
        aux_loss = torch.var(expert_mask.mean(dim=0)) * self.num_experts
        
        return {
            "routing_weights": routing_weights,      # (batch*seq_len, k)
            "routing_indices": routing_indices,      # (batch*seq_len, k)
            "aux_loss": aux_loss,
        }


# ============================================================================
# Expert MLP
# ============================================================================

class ExpertMLP(nn.Module):
    """Individual expert network (SwiGLU architecture)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through expert.
        
        Args:
            x: (batch*seq_len, hidden_size)
        
        Returns:
            Expert output (batch*seq_len, hidden_size)
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ============================================================================
# MoE Feed-Forward Layer
# ============================================================================

class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward layer."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Router
        self.router = ExpertRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok
        )
        
        # Experts
        self.experts = nn.ModuleList([
            ExpertMLP(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Process input through MoE layer.
        
        Args:
            x: (batch, seq_len, hidden_size)
        
        Returns:
            Dict with output and auxiliary loss
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (batch*seq_len, hidden_size)
        
        # Get routing decisions
        route = self.router(x)
        routing_weights = route["routing_weights"]  # (batch*seq_len, k)
        routing_indices = route["routing_indices"]  # (batch*seq_len, k)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        expert_used = torch.zeros(self.num_experts, dtype=torch.bool, device=x.device)
        
        # Process each expert
        for expert_id in range(self.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (routing_indices == expert_id).any(dim=-1)
            expert_indices = torch.where(expert_mask)[0]
            
            if len(expert_indices) == 0:
                continue
            
            # Limit to expert capacity
            if len(expert_indices) > self.expert_capacity:
                perm = torch.randperm(len(expert_indices), device=x.device)
                expert_indices = expert_indices[perm[:self.expert_capacity]]
            
            # Get tokens for this expert
            expert_input = x_flat[expert_indices]
            
            # Process through expert
            expert_output = self.experts[expert_id](expert_input)
            
            # Get routing weights for these tokens
            weight_mask = (routing_indices[expert_indices] == expert_id)
            expert_weights = routing_weights[expert_indices][weight_mask]
            
            # Weight and accumulate output
            output[expert_indices] = output[expert_indices] + expert_output * expert_weights.unsqueeze(-1)
            
            expert_used[expert_id] = True
        
        # Reshape output
        output = output.view(batch_size, seq_len, hidden_size)
        
        return {
            "output": output,
            "aux_loss": route["aux_loss"],
            "experts_used": expert_used.sum().item(),
        }


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """Complete transformer block with attention and MoE FFN."""
    
    def __init__(self, config: MoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Attention
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = Attention(config)
        
        # FFN - MoE for even layers, regular MLP for odd
        use_moe = (layer_idx % 2 == 0) and (config.num_experts > 1)
        self.use_moe = use_moe
        
        if self.use_moe:
            self.ffn_norm = RMSNorm(config.hidden_size)
            self.ffn = MoEFeedForward(config)
        else:
            # Regular MLP
            self.ffn_norm = RMSNorm(config.hidden_size)
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # Pre-norm attention
        x_norm = self.attention_norm(x)
        attn_out = self.attention(x_norm, attention_mask, position_ids)
        x = x + attn_out
        
        # Pre-norm FFN
        x_norm = self.ffn_norm(x)
        
        if self.use_moe:
            ffn_out = self.ffn(x_norm)
            x = x + ffn_out["output"]
            aux_loss = ffn_out["aux_loss"]
        else:
            gate, up = self.gate_proj(x_norm).chunk(2, dim=-1)
            ffn_out = F.silu(gate) * up
            x = x + self.down_proj(ffn_out)
            aux_loss = torch.tensor(0.0, device=x.device)
        
        return {"hidden_states": x, "aux_loss": aux_loss}


# ============================================================================
# Complete Language Model
# ============================================================================

class MoELanguageModel(nn.Module):
    """Complete MoE language model."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Output
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embed_tokens.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        
        for layer in self.layers:
            nn.init.normal_(layer.attention.qkv_proj.weight, std=0.02)
            nn.init.normal_(layer.attention.o_proj.weight, std=0.02)
            
            if layer.use_moe:
                for expert in layer.ffn.experts:
                    nn.init.normal_(expert.gate_proj.weight, std=0.02)
                    nn.init.normal_(expert.up_proj.weight, std=0.02)
                    nn.init.normal_(expert.down_proj.weight, std=0.02)
            else:
                nn.init.normal_(layer.gate_proj.weight, std=0.02)
                nn.init.normal_(layer.down_proj.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position IDs
        if attention_mask is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Process through layers
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        
        for layer in self.layers:
            output = layer(hidden_states, attention_mask, position_ids)
            hidden_states = output["hidden_states"]
            total_aux_loss = total_aux_loss + output["aux_loss"]
        
        # Final norm and logits
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Total loss includes auxiliary loss
            loss = lm_loss + 0.01 * total_aux_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self(input_ids)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k sampling
                top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                probs = F.softmax(top_k_values, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_k_indices, 1, next_token)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================================================
# Training Loop (Simplified)
# ============================================================================

class Trainer:
    """Simple training loop for MoE model."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 3e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def train_step(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "aux_loss": outputs["aux_loss"].item(),
        }
    
    def evaluate(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
        
        return {
            "loss": outputs["loss"].item(),
            "aux_loss": outputs["aux_loss"].item(),
        }


# ============================================================================
# Dummy Dataset for Testing
# ============================================================================

class DummyDataset:
    """Random token sequences for testing."""
    
    def __init__(self, num_samples: int = 100, seq_length: int = 128, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random tokens (avoid padding tokens 0-3)
        input_ids = torch.randint(4, self.vocab_size - 1, (self.seq_length,))
        
        # Labels are shifted input_ids
        labels = input_ids.clone()
        labels[0] = -100  # No loss for first token
        
        return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# Main
# ============================================================================

def main():
    """Simple main function to test the model."""
    print("Initializing MoE model...")
    
    # Config
    config = MoEConfig(
        vocab_size=10000,  # Smaller for testing
        hidden_size=1024,
        intermediate_size=4096,
        num_attention_heads=8,
        num_hidden_layers=6,
        num_experts=8,
        expert_capacity=64,
        num_experts_per_tok=2,
    )
    
    # Model
    model = MoELanguageModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dataset
    train_dataset = DummyDataset(num_samples=50, seq_length=64, vocab_size=config.vocab_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
    
    # Trainer
    trainer = Trainer(model, learning_rate=1e-4)
    
    # Training loop
    print("Starting training...")
    for epoch in range(2):
        total_loss = 0
        total_aux = 0
        
        for batch in train_loader:
            metrics = trainer.train_step(
                batch["input_ids"],
                batch["labels"]
            )
            total_loss += metrics["loss"]
            total_aux += metrics["aux_loss"]
        
        avg_loss = total_loss / len(train_loader)
        avg_aux = total_aux / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Aux Loss={avg_aux:.4f}")
    
    print("Training complete!")
    
    # Generation test
    print("\nTesting generation...")
    test_input = torch.randint(4, 100, (1, 10))
    generated = model.generate(test_input, max_new_tokens=20)
    print(f"Input: {test_input[0][:10].tolist()}")
    print(f"Generated: {generated[0].tolist()}")
    
    return model


if __name__ == "__main__":
    main()
