"""Chess transformer adapted from luxia-base.

Key differences from the language model:
- Bidirectional attention (no causal mask) — all squares see all squares
- Learned positional embeddings (no RoPE) — fixed board geometry
- Chess-specific input encoding: piece type + global state tokens
- Dual heads: policy (8x8x73 = 4672 move logits) + value (scalar [-1, 1])
- z-loss on policy logits prevents explosion during RL training

Preserved from luxia-base:
- QK-norm for training stability
- Block Attention Residuals (AttnRes) for adaptive compute depth
- SwiGLU FFN, GQA, RMSNorm
- FA2 backend support
- Weight initialization with residual scaling
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

logger = logging.getLogger(__name__)

# ── Optional Flash Attention 2 ─────────────────────────────────────────────
_FA2_AVAILABLE = False
try:
    from flash_attn import flash_attn_func

    _FA2_AVAILABLE = True
except ImportError:
    flash_attn_func = None  # type: ignore[assignment,misc]


# ── Configuration ──────────────────────────────────────────────────────────


@dataclass
class ChessModelConfig:
    """Chess transformer configuration."""

    hidden_size: int = 512
    num_layers: int = 12
    num_attention_heads: int = 8
    num_kv_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 1408
    norm_eps: float = 1e-5
    qk_norm: bool = True
    z_loss_weight: float = 1e-5
    activation_checkpointing: bool = False

    # Attention backend: "auto" (FA2 if available, else SDPA), "fa2", "sdpa"
    attn_impl: str = "auto"

    # Block Attention Residuals
    attn_res: bool = False
    attn_res_n_blocks: int = 3  # 12 layers / 4 per block
    attn_res_boundaries: Optional[list[int]] = None

    # Chess-specific dimensions
    num_piece_types: int = 13     # empty + 6 white + 6 black
    num_squares: int = 64
    num_global_tokens: int = 3    # castling, en_passant, side_to_move
    num_castling_states: int = 16 # 4-bit packed
    num_ep_states: int = 9        # 0-7 files + none
    num_sides: int = 2
    policy_moves_per_square: int = 73  # AlphaZero encoding
    value_hidden_size: int = 256       # Value head hidden layer

    @property
    def seq_len(self) -> int:
        return self.num_squares + self.num_global_tokens  # 67

    @property
    def policy_size(self) -> int:
        return self.num_squares * self.policy_moves_per_square  # 4672

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads

    def param_count(self) -> int:
        """Estimate total parameter count."""
        embed = (
            self.num_piece_types * self.hidden_size
            + self.seq_len * self.hidden_size
            + self.num_castling_states * self.hidden_size
            + self.num_ep_states * self.hidden_size
            + self.num_sides * self.hidden_size
        )
        q = self.hidden_size * self.num_attention_heads * self.head_dim
        k = self.hidden_size * self.num_kv_heads * self.head_dim
        v = self.hidden_size * self.num_kv_heads * self.head_dim
        o = self.num_attention_heads * self.head_dim * self.hidden_size
        attn = q + k + v + o
        mlp = 3 * self.hidden_size * self.intermediate_size
        norms = 2 * self.hidden_size
        qk_norms = 2 * self.head_dim if self.qk_norm else 0
        per_layer = attn + mlp + norms + qk_norms
        policy = self.hidden_size * self.policy_moves_per_square + self.policy_moves_per_square
        value = (
            self.hidden_size * self.value_hidden_size + self.value_hidden_size
            + self.value_hidden_size + 1
        )
        return embed + self.num_layers * per_layer + policy + value + self.hidden_size


# ── Model size presets ─────────────────────────────────────────────────────

MODEL_CONFIGS: dict[str, dict] = {
    "smoke": dict(
        hidden_size=128, num_layers=4, num_attention_heads=4,
        num_kv_heads=2, head_dim=32, intermediate_size=384,
        value_hidden_size=64,
    ),
    "small": dict(
        hidden_size=256, num_layers=8, num_attention_heads=4,
        num_kv_heads=2, head_dim=64, intermediate_size=768,
        value_hidden_size=128,
    ),
    "medium": dict(
        hidden_size=512, num_layers=12, num_attention_heads=8,
        num_kv_heads=4, head_dim=64, intermediate_size=1408,
        value_hidden_size=256,
    ),
    "large": dict(
        hidden_size=512, num_layers=16, num_attention_heads=8,
        num_kv_heads=4, head_dim=64, intermediate_size=1408,
        value_hidden_size=256,
    ),
}


# ── Core modules ───────────────────────────────────────────────────────────


def _resolve_attn_impl(config: ChessModelConfig) -> str:
    """Resolve attention backend: "auto" picks FA2 if available, else SDPA."""
    impl = config.attn_impl
    if impl == "auto":
        return "fa2" if _FA2_AVAILABLE else "sdpa"
    if impl == "fa2" and not _FA2_AVAILABLE:
        raise ImportError(
            "attn_impl='fa2' requested but flash-attn is not installed."
        )
    return impl


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class GQAttention(nn.Module):
    """Grouped-Query Attention with optional QK-norm.

    Bidirectional (no causal mask). No RoPE — positional info comes from
    learned embeddings added before the transformer.

    Supports SDPA and FA2 backends.
    """

    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self._attn_impl = _resolve_attn_impl(config)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-norm: RMSNorm on Q and K after projection, before attention
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

    def _forward_sdpa(self, x: torch.Tensor) -> torch.Tensor:
        """SDPA path: (B, nheads, S, D) layout, bidirectional."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_fa2(self, x: torch.Tensor) -> torch.Tensor:
        """FA2 path: (B, S, nheads, D) layout, bidirectional."""
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # FA2 doesn't participate in autocast — ensure bf16
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        attn_output = flash_attn_func(q, k, v, causal=False)

        attn_output = attn_output.contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._attn_impl == "fa2":
            return self._forward_fa2(x)
        return self._forward_sdpa(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with optional AttnRes parameters."""

    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = GQAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

        # AttnRes per-layer pseudo-queries and norms
        if config.attn_res:
            self.attn_res_query = nn.Parameter(torch.zeros(config.hidden_size))
            self.attn_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp_res_query = nn.Parameter(torch.zeros(config.hidden_size))
            self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────


class ChessTransformer(nn.Module):
    """Chess transformer with dual policy/value heads.

    Input: board token tensor from BoardEncoder (shape B x 67).
    Output: policy logits (B x 4672), value (B x 1).
    """

    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Chess-specific embeddings ──────────────────────────────────
        self.piece_embed = nn.Embedding(config.num_piece_types, config.hidden_size)
        self.pos_embed = nn.Embedding(config.seq_len, config.hidden_size)
        self.castling_embed = nn.Embedding(config.num_castling_states, config.hidden_size)
        self.ep_embed = nn.Embedding(config.num_ep_states, config.hidden_size)
        self.side_embed = nn.Embedding(config.num_sides, config.hidden_size)

        # ── Transformer backbone ───────────────────────────────────────
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # ── Policy head: per-square projection to 73 move types ────────
        self.policy_head = nn.Linear(config.hidden_size, config.policy_moves_per_square)

        # ── Value head: pool → hidden → tanh ───────────────────────────
        self.value_fc1 = nn.Linear(config.hidden_size, config.value_hidden_size)
        self.value_fc2 = nn.Linear(config.value_hidden_size, 1)

        # ── Block Attention Residuals ──────────────────────────────────
        if config.attn_res:
            self.final_res_query = nn.Parameter(torch.zeros(config.hidden_size))
            self.final_res_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

            if config.attn_res_boundaries is not None:
                self._attn_res_boundary_set = frozenset(config.attn_res_boundaries)
            else:
                block_size = math.ceil(config.num_layers / config.attn_res_n_blocks)
                self._attn_res_boundary_set = frozenset(
                    range(0, config.num_layers, block_size)
                )

            self._attn_res_max_sources = len(self._attn_res_boundary_set) + 1

            # Precompute validity masks: 2 per layer + 1 final
            masks = torch.zeros(
                2 * config.num_layers + 1, self._attn_res_max_sources, dtype=torch.bool,
            )
            n_committed = 0
            for i in range(config.num_layers):
                masks[2 * i, : n_committed + 1] = True
                if i in self._attn_res_boundary_set:
                    n_committed += 1
                masks[2 * i + 1, : n_committed + 1] = True
            masks[2 * config.num_layers, : n_committed + 1] = True
            self.register_buffer("_attn_res_masks", masks, persistent=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Weight initialization matching luxia-base conventions."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Scale residual projections by 1/sqrt(2 * num_layers)
        residual_scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.o_proj.weight, mean=0.0, std=std * residual_scale)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std * residual_scale)

        # Policy head: small init → near-uniform initial policy
        nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.policy_head.bias)

        # Value head: zero-init final layer → predict draws initially
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    # ── AttnRes routing (from luxia-base) ──────────────────────────────

    @staticmethod
    def _route_static(
        buf: torch.Tensor,
        query: torch.Tensor,
        norm: nn.Module,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Block Attention Residual routing with fixed-shape masked softmax.

        Args:
            buf: (max_S, B, T, D) padded source buffer
            query: (D,) learned pseudo-query
            norm: RMSNorm for keys
            active_mask: (max_S,) bool
        Returns:
            (B, T, D) attended mixture of active sources
        """
        qw = query * norm.weight  # (D,)
        eps = norm.eps

        rsqrt = torch.rsqrt(buf.pow(2).mean(-1) + eps)  # (max_S, B, T)
        logits = (buf * qw).sum(-1) * rsqrt  # (max_S, B, T)
        logits = logits.masked_fill(~active_mask.view(-1, 1, 1), float("-inf"))
        weights = F.softmax(logits, dim=0)  # (max_S, B, T)

        return (weights.unsqueeze(-1) * buf).sum(0)  # (B, T, D)

    def _forward_attn_res(self, embed: torch.Tensor) -> torch.Tensor:
        """Forward with Block Attention Residuals (compile-friendly)."""
        committed: list[torch.Tensor] = []
        partial = embed
        boundary_set = self._attn_res_boundary_set
        max_s = self._attn_res_max_sources
        masks = self._attn_res_masks
        zero = torch.zeros_like(embed)

        def _pad_and_stack(
            committed: list[torch.Tensor], partial: torch.Tensor,
        ) -> torch.Tensor:
            sources = committed + [partial]
            while len(sources) < max_s:
                sources.append(zero)
            return torch.stack(sources, dim=0)

        for i, layer in enumerate(self.layers):
            # Pre-attention routing
            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.attn_res_query, layer.attn_res_norm, masks[2 * i])

            if i in boundary_set:
                committed.append(partial.clone())
                partial = zero.clone()

            attn_out = layer.attn(layer.attn_norm(h))
            partial = partial + attn_out

            # Pre-MLP routing
            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.mlp_res_query, layer.mlp_res_norm, masks[2 * i + 1])

            mlp_out = layer.ffn(layer.ffn_norm(h))
            partial = partial + mlp_out

        # Final aggregation
        buf = _pad_and_stack(committed, partial)
        x = self._route_static(
            buf, self.final_res_query, self.final_res_norm,
            masks[2 * self.config.num_layers],
        )
        return self.norm(x)

    # ── Forward pass ───────────────────────────────────────────────────

    def forward(
        self,
        board_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            board_tokens: (B, 67) long tensor from BoardEncoder.encode_board().

        Returns:
            policy_logits: (B, 4672) raw logits over all possible moves.
            value: (B, 1) tanh-squashed value in [-1, 1].
        """
        B = board_tokens.shape[0]

        # ── Embed input tokens ─────────────────────────────────────────
        # Global tokens get dedicated embeddings + positional
        castling = self.castling_embed(board_tokens[:, 0])  # (B, D)
        ep = self.ep_embed(board_tokens[:, 1])              # (B, D)
        side = self.side_embed(board_tokens[:, 2])           # (B, D)

        # Square tokens: piece embedding
        pieces = self.piece_embed(board_tokens[:, 3:])       # (B, 64, D)

        # Positional embeddings for all 67 positions
        pos_ids = torch.arange(self.config.seq_len, device=board_tokens.device)
        pos = self.pos_embed(pos_ids)  # (67, D)

        # Combine: global tokens + square tokens, each with positional encoding
        x = torch.cat([
            (castling + pos[0]).unsqueeze(1),
            (ep + pos[1]).unsqueeze(1),
            (side + pos[2]).unsqueeze(1),
            pieces + pos[3:].unsqueeze(0),
        ], dim=1)  # (B, 67, D)

        # ── Transformer backbone ──────────────────────────────────────
        if self.config.attn_res:
            x = self._forward_attn_res(x)
        else:
            for layer in self.layers:
                if self.config.activation_checkpointing and self.training:
                    x = torch_checkpoint(
                        layer, x,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    x = layer(x)
            x = self.norm(x)

        # ── Policy head ───────────────────────────────────────────────
        # Only square tokens (positions 3:67) contribute to policy
        square_features = x[:, self.config.num_global_tokens :, :]  # (B, 64, D)
        policy_logits = self.policy_head(square_features)           # (B, 64, 73)
        policy_logits = policy_logits.reshape(B, -1)                # (B, 4672)

        # ── Value head ────────────────────────────────────────────────
        # Mean pool ALL tokens (global + squares)
        pooled = x.mean(dim=1)                                      # (B, D)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(pooled))))  # (B, 1)

        return policy_logits, value

    # ── NCA transition helpers ─────────────────────────────────────────

    def reinit_embeddings_for_chess(self) -> None:
        """Reinitialize embeddings for chess after NCA pre-training.

        Keeps attention and FFN weights (which learned grid dynamics from NCA).
        Only resets the input embeddings and output heads.
        """
        std = 0.02
        for embed in [
            self.piece_embed, self.pos_embed,
            self.castling_embed, self.ep_embed, self.side_embed,
        ]:
            nn.init.normal_(embed.weight, mean=0.0, std=std)

        nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

    def reinit_mlps(self) -> None:
        """Reinitialize all MLP weights (optional, after NCA pre-training)."""
        std = 0.02
        residual_scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std * residual_scale)
