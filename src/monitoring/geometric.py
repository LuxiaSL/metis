"""Geometric health monitoring for chess transformer training.

Adapted from luxia-base Anamnesis monitoring. Tracks representational
health during self-play RL training, where representation collapse is
a known risk.

Tier 1 (every N steps, <1% overhead):
  - RankMe (effective rank) on probe batch hidden states
  - Stable rank per layer (weight-space)
  - Anisotropy (pairwise cosine similarity)
  - Dead unit fraction
  - Attention entropy per head
  - Policy entropy (chess-specific)

Tier 2 (less frequently, minutes):
  - WeightWatcher alpha proxy per layer
  - TwoNN intrinsic dimensionality at sampled layers

All metric computation functions are from luxia-base (stateless, pure).
The probe forward pass is adapted for bidirectional attention + chess embeddings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for geometric monitoring."""

    tier1_every: int = 500
    tier1_probe_size: int = 256    # Smaller than language (positions are information-dense)
    tier1_sample_layers: list[int] = field(default_factory=lambda: [])

    tier2_every: int = 5000
    tier2_twonn_samples: int = 2000
    tier2_twonn_layers: list[int] = field(default_factory=lambda: [])


class GeometricMonitor:
    """Geometric health monitor for chess transformer.

    Computes metrics at configurable intervals. Holds a fixed probe batch
    of board positions for longitudinal comparability.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MonitorConfig] = None,
    ) -> None:
        self.config = config or MonitorConfig()

        # Unwrap DDP / compile wrappers
        m = model.module if hasattr(model, "module") else model
        if hasattr(m, "_orig_mod"):
            m = m._orig_mod
        self.model = m

        num_layers = len(self.model.layers)
        if not self.config.tier1_sample_layers:
            step = max(1, num_layers // 4)
            self.config.tier1_sample_layers = sorted(
                set([0, step, 2 * step, 3 * step, num_layers - 1])
            )
        if not self.config.tier2_twonn_layers:
            self.config.tier2_twonn_layers = [
                0, num_layers // 4, num_layers // 2,
                3 * num_layers // 4, num_layers - 1,
            ]

        self._probe_batch: Optional[torch.Tensor] = None
        self._attn_res_diagnostics: dict[str, float] = {}

        logger.info(
            "GeometricMonitor: %d layers, tier1=%s, tier2_ID=%s",
            num_layers, self.config.tier1_sample_layers, self.config.tier2_twonn_layers,
        )

    def set_probe_batch(self, board_tokens: torch.Tensor) -> None:
        """Set fixed probe batch of board positions (shape: B x 67)."""
        self._probe_batch = board_tokens.clone()
        logger.info("Probe batch set: shape %s", tuple(board_tokens.shape))

    # ── Tier 1: lightweight streaming metrics ──────────────────────────

    @torch.no_grad()
    def tier1(
        self,
        step: int,
        probe_batch: Optional[torch.Tensor] = None,
        policy_logits: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Compute Tier 1 geometric health metrics."""
        t0 = time.time()
        batch = probe_batch if probe_batch is not None else self._probe_batch
        if batch is None:
            logger.warning("No probe batch — skipping Tier 1")
            return {}

        device = next(self.model.parameters()).device
        batch = batch.to(device)
        metrics: dict[str, float] = {}

        hidden_states, attn_weights = self._probe_forward(batch)

        # RankMe on last-layer hidden states
        last_idx = len(self.model.layers) - 1
        if last_idx in hidden_states:
            last_h = hidden_states[last_idx]
            H = last_h.reshape(-1, last_h.shape[-1]).float()
            metrics["geo/rankme_last"] = _rankme(H)

        # Per-layer metrics
        for layer_idx in self.config.tier1_sample_layers:
            prefix = f"geo/layer_{layer_idx}"
            layer = self.model.layers[layer_idx]

            # Stable rank of weight matrices
            for name, param in [
                ("q_proj", layer.attn.q_proj.weight),
                ("k_proj", layer.attn.k_proj.weight),
                ("o_proj", layer.attn.o_proj.weight),
                ("gate_proj", layer.ffn.gate_proj.weight),
                ("down_proj", layer.ffn.down_proj.weight),
            ]:
                metrics[f"{prefix}/stable_rank_{name}"] = _stable_rank(param)

            if layer_idx in hidden_states:
                h = hidden_states[layer_idx]
                metrics[f"{prefix}/dead_units"] = _dead_unit_fraction(h)
                h_flat = h.reshape(-1, h.shape[-1])
                metrics[f"{prefix}/anisotropy"] = _anisotropy(h_flat, max_samples=512)

            if layer_idx in attn_weights:
                ent_mean, ent_std = _attention_entropy_stats(attn_weights[layer_idx])
                metrics[f"{prefix}/attn_entropy_mean"] = ent_mean
                metrics[f"{prefix}/attn_entropy_std"] = ent_std

        # Chess-specific: policy entropy (how confident is the model?)
        if policy_logits is not None:
            metrics["geo/policy_entropy"] = _policy_entropy(policy_logits)

        if self._attn_res_diagnostics:
            metrics.update(self._attn_res_diagnostics)

        elapsed = time.time() - t0
        metrics["geo/tier1_time_s"] = elapsed
        metrics["geo/step"] = float(step)

        logger.info(
            "Tier 1 [step %d]: RankMe=%.1f, time=%.2fs",
            step, metrics.get("geo/rankme_last", 0), elapsed,
        )
        return metrics

    # ── Tier 2: checkpoint-level metrics ───────────────────────────────

    @torch.no_grad()
    def tier2(
        self, step: int, probe_batch: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Compute Tier 2 metrics (WeightWatcher alpha, TwoNN ID)."""
        t0 = time.time()
        metrics: dict[str, float] = {}

        # WeightWatcher alpha (proxy — log-log regression on eigenspectrum)
        all_alphas: list[float] = []
        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            for name, param in [
                ("q_proj", layer.attn.q_proj.weight),
                ("o_proj", layer.attn.o_proj.weight),
                ("gate_proj", layer.ffn.gate_proj.weight),
                ("down_proj", layer.ffn.down_proj.weight),
            ]:
                alpha = _weightwatcher_alpha(param)
                if alpha is not None:
                    metrics[f"geo/ww_alpha/layer_{layer_idx}/{name}"] = alpha
                    all_alphas.append(alpha)

        if all_alphas:
            metrics["geo/ww_alpha_mean"] = sum(all_alphas) / len(all_alphas)
            healthy = sum(1 for a in all_alphas if 2.0 < a < 4.0)
            metrics["geo/ww_alpha_healthy_frac"] = healthy / len(all_alphas)

        # TwoNN intrinsic dimensionality
        batch = probe_batch if probe_batch is not None else self._probe_batch
        if batch is not None:
            device = next(self.model.parameters()).device
            batch = batch.to(device)
            hidden_states, _ = self._probe_forward(batch)

            for layer_idx in self.config.tier2_twonn_layers:
                if layer_idx in hidden_states:
                    h = hidden_states[layer_idx].reshape(-1, hidden_states[layer_idx].shape[-1])
                    n = min(self.config.tier2_twonn_samples, h.shape[0])
                    idx = torch.randperm(h.shape[0])[:n]
                    id_est = _twonn_id(h[idx].float())
                    if id_est is not None:
                        metrics[f"geo/twonn_id/layer_{layer_idx}"] = id_est

        elapsed = time.time() - t0
        metrics["geo/tier2_time_s"] = elapsed
        logger.info(
            "Tier 2 [step %d]: WW_alpha=%.2f, healthy=%.0f%%, time=%.1fs",
            step,
            metrics.get("geo/ww_alpha_mean", 0),
            metrics.get("geo/ww_alpha_healthy_frac", 0) * 100,
            elapsed,
        )
        return metrics

    # ── Forward pass for probing ───────────────────────────────────────

    def _probe_forward(
        self, board_tokens: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """Run forward pass capturing hidden states and attention weights.

        Handles both standard and AttnRes forward paths.
        """
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}

        needed = (
            set(self.config.tier1_sample_layers)
            | set(self.config.tier2_twonn_layers)
            | {len(self.model.layers) - 1}
        )

        was_training = self.model.training
        self.model.eval()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            use_attn_res = getattr(self.model.config, "attn_res", False)

            if use_attn_res:
                hidden_states, attn_weights = self._probe_forward_attn_res(
                    board_tokens, needed,
                )
            else:
                # Replicate chess embedding from ChessTransformer.forward()
                x = self._embed_board(board_tokens)

                for i, layer in enumerate(self.model.layers):
                    x = layer(x)
                    if i in needed:
                        hidden_states[i] = x.detach()
                        if i in self.config.tier1_sample_layers:
                            attn_w = self._get_attention_weights(layer, layer.attn_norm(x))
                            if attn_w is not None:
                                attn_weights[i] = attn_w

        if was_training:
            self.model.train()

        return hidden_states, attn_weights

    def _probe_forward_attn_res(
        self,
        board_tokens: torch.Tensor,
        needed: set[int],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        """AttnRes-aware probe forward pass."""
        hidden_states: dict[int, torch.Tensor] = {}
        attn_weights: dict[int, torch.Tensor] = {}
        model = self.model

        embed = self._embed_board(board_tokens)
        committed: list[torch.Tensor] = []
        partial = embed
        boundary_set = model._attn_res_boundary_set
        max_s = model._attn_res_max_sources
        masks = model._attn_res_masks
        zero = torch.zeros_like(embed)

        def _pad_and_stack(committed: list[torch.Tensor], partial: torch.Tensor) -> torch.Tensor:
            sources = committed + [partial]
            while len(sources) < max_s:
                sources.append(zero)
            return torch.stack(sources, dim=0)

        for i, layer in enumerate(model.layers):
            buf = _pad_and_stack(committed, partial)
            h = model._route_static(buf, layer.attn_res_query, layer.attn_res_norm, masks[2 * i])

            if i in boundary_set:
                committed.append(partial.clone())
                partial = zero.clone()

            if i in needed:
                hidden_states[i] = h.detach()
                if i in self.config.tier1_sample_layers:
                    attn_w = self._get_attention_weights(layer, layer.attn_norm(h))
                    if attn_w is not None:
                        attn_weights[i] = attn_w

            attn_out = layer.attn(layer.attn_norm(h))
            partial = partial + attn_out

            buf = _pad_and_stack(committed, partial)
            h = model._route_static(buf, layer.mlp_res_query, layer.mlp_res_norm, masks[2 * i + 1])
            mlp_out = layer.ffn(layer.ffn_norm(h))
            partial = partial + mlp_out

        # Final aggregation
        buf = _pad_and_stack(committed, partial)
        last_idx = len(model.layers) - 1
        final_h = model._route_static(
            buf, model.final_res_query, model.final_res_norm, masks[2 * len(model.layers)],
        )
        hidden_states[last_idx] = final_h.detach()

        # AttnRes diagnostics
        self._attn_res_diagnostics.clear()
        try:
            qw = model.final_res_query * model.final_res_norm.weight
            eps = model.final_res_norm.eps
            rsqrt_val = torch.rsqrt(buf.pow(2).mean(-1) + eps)
            logits = (buf * qw).sum(-1) * rsqrt_val
            final_mask = masks[2 * len(model.layers)]
            logits = logits.masked_fill(~final_mask.view(-1, 1, 1), float("-inf"))
            alpha_weights = F.softmax(logits, dim=0)
            avg_alpha = alpha_weights.mean(dim=(1, 2)).detach()
            n_active = final_mask.sum().item()
            for block_idx in range(int(n_active)):
                self._attn_res_diagnostics[f"attnres/final_alpha/block_{block_idx}"] = avg_alpha[block_idx].item()
                norm_val = buf[block_idx].detach().float().norm(dim=-1).mean().item()
                self._attn_res_diagnostics[f"attnres/block_norm/{block_idx}"] = norm_val
        except Exception as e:
            logger.debug("AttnRes diagnostics failed: %s", e)

        return hidden_states, attn_weights

    def _embed_board(self, board_tokens: torch.Tensor) -> torch.Tensor:
        """Replicate chess-specific embedding from ChessTransformer.forward()."""
        model = self.model
        castling = model.castling_embed(board_tokens[:, 0])
        ep = model.ep_embed(board_tokens[:, 1])
        side = model.side_embed(board_tokens[:, 2])
        pieces = model.piece_embed(board_tokens[:, 3:])

        pos_ids = torch.arange(model.config.seq_len, device=board_tokens.device)
        pos = model.pos_embed(pos_ids)

        return torch.cat([
            (castling + pos[0]).unsqueeze(1),
            (ep + pos[1]).unsqueeze(1),
            (side + pos[2]).unsqueeze(1),
            pieces + pos[3:].unsqueeze(0),
        ], dim=1)

    def _get_attention_weights(
        self, layer: nn.Module, x_normed: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute attention weights for a layer (bidirectional, no RoPE)."""
        try:
            attn = layer.attn
            x_sub = x_normed[:4]  # Limit batch for memory
            bsz, seq_len, _ = x_sub.shape

            q = attn.q_proj(x_sub).view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = attn.k_proj(x_sub).view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

            if attn.qk_norm:
                q = attn.q_norm(q)
                k = attn.k_norm(k)

            # Expand KV for GQA
            num_kv_groups = attn.num_heads // attn.num_kv_heads
            if num_kv_groups > 1:
                k = k.repeat_interleave(num_kv_groups, dim=1)

            scale = attn.head_dim ** -0.5
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            # No causal mask — bidirectional attention
            weights = F.softmax(scores, dim=-1)
            return weights.detach()
        except Exception as e:
            logger.debug("Failed to compute attention weights: %s", e)
            return None


# ── Metric functions (from luxia-base, stateless/pure) ─────────────────


def _rankme(H: torch.Tensor, eps: float = 1e-7) -> float:
    """RankMe: exp(entropy of normalized singular values)."""
    try:
        S = torch.linalg.svdvals(H)
        S = S / (S.sum() + eps)
        S = S[S > eps]
        entropy = -(S * torch.log(S)).sum()
        return torch.exp(entropy).item()
    except Exception:
        return 0.0


def _stable_rank(W: torch.Tensor) -> float:
    """Stable rank: ||W||_F^2 / ||W||_2^2."""
    try:
        W_f = W.float()
        frob_sq = W_f.pow(2).sum()
        spectral_sq = torch.linalg.svdvals(W_f)[0].pow(2)
        return (frob_sq / (spectral_sq + 1e-10)).item()
    except Exception:
        return 0.0


def _dead_unit_fraction(hidden: torch.Tensor, threshold: float = 1e-6) -> float:
    """Fraction of neurons with near-zero mean absolute activation."""
    mean_abs = hidden.float().abs().mean(dim=(0, 1))
    return (mean_abs < threshold).float().mean().item()


def _anisotropy(H: torch.Tensor, max_samples: int = 512) -> float:
    """Average pairwise cosine similarity."""
    if H.shape[0] > max_samples:
        H = H[torch.randperm(H.shape[0])[:max_samples]]
    H = F.normalize(H.float(), dim=-1)
    sim = H @ H.T
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return sim[mask].mean().item()


def _attention_entropy_stats(attn: torch.Tensor, eps: float = 1e-10) -> tuple[float, float]:
    """Mean and std of per-head attention entropy."""
    attn_c = attn.float().clamp(min=eps)
    entropy = -(attn_c * attn_c.log()).sum(dim=-1)
    per_head = entropy.mean(dim=(0, 2))
    return per_head.mean().item(), per_head.std().item()


def _policy_entropy(policy_logits: torch.Tensor) -> float:
    """Average entropy of the policy distribution (chess-specific)."""
    probs = F.softmax(policy_logits.float(), dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
    return entropy.mean().item()


def _weightwatcher_alpha(W: torch.Tensor, min_sv: int = 10) -> Optional[float]:
    """Proxy WeightWatcher alpha via log-log regression on eigenspectrum."""
    try:
        S = torch.linalg.svdvals(W.float().detach())
        eigs = S.pow(2)
        if len(eigs) < min_sv:
            return None
        eigs_sorted = eigs.sort(descending=True).values
        threshold = eigs_sorted[0] * 1e-10
        eigs_valid = eigs_sorted[eigs_sorted > threshold]
        if len(eigs_valid) < min_sv:
            return None
        n = len(eigs_valid)
        log_rank = torch.log(torch.arange(1, n + 1, dtype=torch.float32, device=W.device))
        log_eig = torch.log(eigs_valid)
        x_mean, y_mean = log_rank.mean(), log_eig.mean()
        slope = ((log_rank - x_mean) * (log_eig - y_mean)).sum() / ((log_rank - x_mean).pow(2).sum() + 1e-10)
        alpha = -slope.item()
        if alpha < 0.1 or alpha > 20:
            return None
        return alpha
    except Exception:
        return None


def _twonn_id(X: torch.Tensor) -> Optional[float]:
    """TwoNN intrinsic dimensionality estimation (Facco et al. 2017)."""
    try:
        n = X.shape[0]
        if n < 10:
            return None
        if n > 5000:
            X = X[:5000]
            n = 5000
        dists = torch.cdist(X, X)
        dists.fill_diagonal_(float("inf"))
        topk = dists.topk(2, dim=1, largest=False)
        r1, r2 = topk.values[:, 0], topk.values[:, 1]
        valid = r1 > 1e-10
        if valid.sum() < 10:
            return None
        mu = r2[valid] / r1[valid]
        mu_sorted = mu.sort().values
        n_valid = len(mu_sorted)
        i = torch.arange(1, n_valid + 1, dtype=torch.float32, device=X.device)
        log_survival = torch.log(1.0 - i / (n_valid + 1))
        log_mu = torch.log(mu_sorted)
        x_mean, y_mean = log_mu.mean(), log_survival.mean()
        slope = ((log_mu - x_mean) * (log_survival - y_mean)).sum() / ((log_mu - x_mean).pow(2).sum() + 1e-10)
        d = -slope.item()
        if d < 0.5 or d > 10000:
            return None
        return d
    except Exception:
        return None
