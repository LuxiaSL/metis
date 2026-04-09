"""Chess transformer adapted from luxia-base.

Key differences from the language model:
- Bidirectional attention (no causal mask) — all squares see all squares
- Shaw-style relative position bias — chess-aware topology encoding
- Chess-specific input encoding: piece type + global state tokens
- WDL value head (win/draw/loss 3-class) + material & activity aux heads
- z-loss on policy logits prevents explosion during RL training

Preserved from luxia-base:
- QK-norm for training stability
- Block Attention Residuals (AttnRes) — learned selective routing that equalizes
  gradient flow across depth and prevents late-layer collapse
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

    # Attention backend: "sdpa" (default, confirmed fastest), "fa2", "auto"
    attn_impl: str = "sdpa"

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


class RelativePositionBias(nn.Module):
    """Shaw-style relative position bias with chess-aware topology features.

    For each pair of positions (i, j) in the 67-token sequence, computes a
    scalar bias added to attention logits before softmax. Encodes chess-relevant
    spatial relationships between squares.

    Features for square pairs (positions 3-66 → squares a1-h8):
    - Rank difference (0-7): captures vertical distance
    - File difference (0-7): captures horizontal distance
    - Same diagonal: captures bishop-like relationships
    - Same anti-diagonal: captures the other diagonal
    - Knight-reachable: captures knight move topology

    Global tokens (0-2: castling, en passant, side) get separate learned biases.
    """

    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        seq_len = config.seq_len  # 67
        n_global = config.num_global_tokens  # 3
        num_heads = config.num_attention_heads  # per-head biases for specialization
        self.num_heads = num_heads

        # Per-head embeddings for distance features
        self.rank_diff_embed = nn.Embedding(8, num_heads)  # |rank_i - rank_j| ∈ [0, 7]
        self.file_diff_embed = nn.Embedding(8, num_heads)  # |file_i - file_j| ∈ [0, 7]

        # Per-head learned biases for boolean features
        self.diag_bias = nn.Parameter(torch.zeros(num_heads))
        self.antidiag_bias = nn.Parameter(torch.zeros(num_heads))
        self.knight_bias = nn.Parameter(torch.zeros(num_heads))

        # Per-head global token biases
        self.global_bias = nn.Parameter(torch.zeros(num_heads, n_global, seq_len))

        # Precompute topology feature indices for the 64x64 square block
        rank_diff = torch.zeros(64, 64, dtype=torch.long)
        file_diff = torch.zeros(64, 64, dtype=torch.long)
        same_diag = torch.zeros(64, 64, dtype=torch.bool)
        same_antidiag = torch.zeros(64, 64, dtype=torch.bool)
        knight_reach = torch.zeros(64, 64, dtype=torch.bool)

        for sq_i in range(64):
            ri, fi = sq_i // 8, sq_i % 8
            for sq_j in range(64):
                rj, fj = sq_j // 8, sq_j % 8
                dr, df = abs(ri - rj), abs(fi - fj)
                rank_diff[sq_i, sq_j] = dr
                file_diff[sq_i, sq_j] = df
                same_diag[sq_i, sq_j] = (ri - fi) == (rj - fj)
                same_antidiag[sq_i, sq_j] = (ri + fi) == (rj + fj)
                knight_reach[sq_i, sq_j] = (dr == 2 and df == 1) or (dr == 1 and df == 2)

        self.register_buffer("_rank_diff", rank_diff)
        self.register_buffer("_file_diff", file_diff)
        self.register_buffer("_same_diag", same_diag.float())
        self.register_buffer("_same_antidiag", same_antidiag.float())
        self.register_buffer("_knight_reach", knight_reach.float())
        self._seq_len = seq_len
        self._n_global = n_global

    def forward(self) -> torch.Tensor:
        """Compute the (1, H, 67, 67) per-head relative position bias matrix."""
        H = self.num_heads
        bias = torch.zeros(
            H, self._seq_len, self._seq_len,
            device=self._rank_diff.device, dtype=self.rank_diff_embed.weight.dtype,
        )

        # Square-to-square biases (positions 3-66)
        # Embedding lookups: (64, 64) indices → (64, 64, H)
        # Boolean features: (64, 64, 1) * (H,) → (64, 64, H) via broadcast
        g = self._n_global
        sq_bias = (
            self.rank_diff_embed(self._rank_diff)
            + self.file_diff_embed(self._file_diff)
            + self._same_diag.unsqueeze(-1) * self.diag_bias
            + self._same_antidiag.unsqueeze(-1) * self.antidiag_bias
            + self._knight_reach.unsqueeze(-1) * self.knight_bias
        )  # (64, 64, H)
        bias[:, g:, g:] = sq_bias.permute(2, 0, 1)  # (H, 64, 64)

        # Global token biases: per-head (H, n_global, seq_len)
        # Row: global→all positions; Column: square→global (avoids double-counting g×g)
        bias[:, :g, :] = self.global_bias                            # (H, 3, 67)
        bias[:, g:, :g] = self.global_bias[:, :, g:].transpose(1, 2)  # (H, 64, 3)

        return bias.unsqueeze(0)  # (1, H, 67, 67)


class GQAttention(nn.Module):
    """Grouped-Query Attention with optional QK-norm.

    Bidirectional (no causal mask). Positional info comes from relative
    position bias added to attention scores.

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

    def _forward_sdpa(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            attn_mask=attn_bias,  # additive bias (1, 1, S, S) or None
            is_causal=False,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _forward_fa2(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """FA2 path: (B, S, nheads, D) layout, bidirectional.

        FA2 doesn't support arbitrary attention biases. If attn_bias is provided,
        falls back to SDPA.
        """
        if attn_bias is not None:
            return self._forward_sdpa(x, attn_bias=attn_bias)

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

    def forward(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._attn_impl == "fa2":
            return self._forward_fa2(x, attn_bias=attn_bias)
        return self._forward_sdpa(x, attn_bias=attn_bias)


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

    def forward(
        self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attn_bias=attn_bias)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────


class ChessTransformer(nn.Module):
    """Chess transformer with policy, WDL value, and auxiliary heads.

    Input: board token tensor from BoardEncoder (shape B x 67).
    Output: (policy_logits, wdl_logits, material_pred, activity_pred).
    """

    def __init__(self, config: ChessModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Chess-specific embeddings ──────────────────────────────────
        self.piece_embed = nn.Embedding(config.num_piece_types, config.hidden_size)
        self.castling_embed = nn.Embedding(config.num_castling_states, config.hidden_size)
        self.ep_embed = nn.Embedding(config.num_ep_states, config.hidden_size)
        self.side_embed = nn.Embedding(config.num_sides, config.hidden_size)

        # ── Relative position bias (replaces learned pos_embed) ────────
        self.rel_pos_bias = RelativePositionBias(config)

        # ── Transformer backbone ───────────────────────────────────────
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        # ── Policy head: per-square projection to 73 move types ────────
        self.policy_head = nn.Linear(config.hidden_size, config.policy_moves_per_square)

        # ── WDL value head: pool → hidden → 3-class (loss/draw/win) ────
        self.value_fc1 = nn.Linear(config.hidden_size, config.value_hidden_size)
        self.value_fc2 = nn.Linear(config.value_hidden_size, 3)

        # ── Auxiliary heads ────────────────────────────────────────────
        aux_hidden = 128
        self.material_fc1 = nn.Linear(config.hidden_size, aux_hidden)
        self.material_fc2 = nn.Linear(aux_hidden, 1)
        self.activity_fc1 = nn.Linear(config.hidden_size, aux_hidden)
        self.activity_fc2 = nn.Linear(aux_hidden, 1)

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

        # WDL value head: zero-init final layer → uniform [1/3, 1/3, 1/3]
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)

        # Auxiliary heads: zero-init final layers → predict 0 initially
        nn.init.zeros_(self.material_fc2.weight)
        nn.init.zeros_(self.material_fc2.bias)
        nn.init.zeros_(self.activity_fc2.weight)
        nn.init.zeros_(self.activity_fc2.bias)

    # ── AttnRes helpers ─────────────────────────────────────────────────

    @staticmethod
    def _ckpt_sublayer(
        norm: nn.Module, fn: nn.Module, x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Checkpoint-friendly helper: norm → sublayer (attention or FFN)."""
        normed = norm(x)
        if attn_bias is not None:
            return fn(normed, attn_bias=attn_bias)
        return fn(normed)

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

    def _forward_attn_res(
        self, embed: torch.Tensor, attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward with Block Attention Residuals.

        Supports activation checkpointing: wraps each layer's attention and FFN
        in torch.utils.checkpoint to trade compute for memory. Routing is kept
        outside the checkpoint boundary since its intermediates are smaller.
        """
        committed: list[torch.Tensor] = []
        partial = embed
        boundary_set = self._attn_res_boundary_set
        max_s = self._attn_res_max_sources
        masks = self._attn_res_masks
        zero = torch.zeros_like(embed)
        do_ckpt = self.config.activation_checkpointing and self.training

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

            if do_ckpt:
                attn_out = torch_checkpoint(
                    self._ckpt_sublayer, layer.attn_norm, layer.attn, h, attn_bias,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                attn_out = layer.attn(layer.attn_norm(h), attn_bias=attn_bias)
            partial = partial + attn_out

            # Pre-MLP routing
            buf = _pad_and_stack(committed, partial)
            h = self._route_static(buf, layer.mlp_res_query, layer.mlp_res_norm, masks[2 * i + 1])

            if do_ckpt:
                mlp_out = torch_checkpoint(
                    self._ckpt_sublayer, layer.ffn_norm, layer.ffn, h,
                    use_reentrant=False, preserve_rng_state=False,
                )
            else:
                mlp_out = layer.ffn(layer.ffn_norm(h))
            partial = partial + mlp_out

        # Final aggregation
        buf = _pad_and_stack(committed, partial)
        x = self._route_static(
            buf, self.final_res_query, self.final_res_norm,
            masks[2 * self.config.num_layers],
        )
        return self.norm(x)

    # ── Forward passes ──────────────────────────────────────────────────

    def _embed(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """Embed board tokens into hidden representations. (B, 67) → (B, 67, D).

        Positional information is provided by relative position bias in attention,
        not additive positional embeddings.
        """
        castling = self.castling_embed(board_tokens[:, 0])  # (B, D)
        ep = self.ep_embed(board_tokens[:, 1])              # (B, D)
        side = self.side_embed(board_tokens[:, 2])          # (B, D)
        pieces = self.piece_embed(board_tokens[:, 3:])      # (B, 64, D)

        return torch.cat([
            castling.unsqueeze(1),
            ep.unsqueeze(1),
            side.unsqueeze(1),
            pieces,
        ], dim=1)

    def backbone_forward(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """Run embedding + transformer layers + norm, without heads.

        Used by NCA bootstrap training (which attaches its own prediction head).

        Args:
            board_tokens: (B, 67) long tensor.

        Returns:
            hidden_states: (B, 67, D) normalized hidden representations.
        """
        x = self._embed(board_tokens)
        attn_bias = self.rel_pos_bias()

        if self.config.attn_res:
            x = self._forward_attn_res(x, attn_bias=attn_bias)
        else:
            for layer in self.layers:
                if self.config.activation_checkpointing and self.training:
                    x = torch_checkpoint(
                        layer, x, attn_bias,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    x = layer(x, attn_bias=attn_bias)
            x = self.norm(x)

        return x

    def forward(
        self,
        board_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            board_tokens: (B, 67) long tensor from BoardEncoder.encode_board().

        Returns:
            policy_logits: (B, 4672) raw logits over all possible moves.
            wdl_logits: (B, 3) win/draw/loss logits.
            material_pred: (B, 1) predicted material balance (normalized).
            activity_pred: (B, 1) predicted legal move count (normalized).
        """
        B = board_tokens.shape[0]
        x = self.backbone_forward(board_tokens)

        # Policy head: square tokens only
        square_features = x[:, self.config.num_global_tokens :, :]  # (B, 64, D)
        policy_logits = self.policy_head(square_features)           # (B, 64, 73)
        policy_logits = policy_logits.reshape(B, -1)                # (B, 4672)

        # Pool all tokens for value + aux heads
        pooled = x.mean(dim=1)                                      # (B, D)

        # WDL value head
        wdl_logits = self.value_fc2(F.relu(self.value_fc1(pooled))) # (B, 3)

        # Auxiliary heads
        material_pred = self.material_fc2(F.relu(self.material_fc1(pooled)))  # (B, 1)
        activity_pred = self.activity_fc2(F.relu(self.activity_fc1(pooled)))  # (B, 1)

        return policy_logits, wdl_logits, material_pred, activity_pred

    # ── NCA transition helpers ─────────────────────────────────────────

    def reinit_embeddings_for_chess(self) -> None:
        """Reinitialize embeddings and heads for chess after NCA pre-training.

        Keeps attention and FFN weights (which learned grid dynamics from NCA).
        Resets input embeddings, output heads, and auxiliary heads.
        """
        std = 0.02
        for embed in [
            self.piece_embed,
            self.castling_embed, self.ep_embed, self.side_embed,
        ]:
            nn.init.normal_(embed.weight, mean=0.0, std=std)

        # Reinit relative position bias (small init for gradual integration)
        for param in self.rel_pos_bias.parameters():
            if param.dim() >= 2:
                nn.init.normal_(param, mean=0.0, std=0.01)
            else:
                nn.init.zeros_(param)

        nn.init.normal_(self.policy_head.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_fc2.weight)
        nn.init.zeros_(self.value_fc2.bias)
        nn.init.zeros_(self.material_fc2.weight)
        nn.init.zeros_(self.material_fc2.bias)
        nn.init.zeros_(self.activity_fc2.weight)
        nn.init.zeros_(self.activity_fc2.bias)

    def reinit_mlps(self) -> None:
        """Reinitialize all MLP weights (optional, after NCA pre-training)."""
        std = 0.02
        residual_scale = 1.0 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.ffn.gate_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.up_proj.weight, mean=0.0, std=std)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=std * residual_scale)
