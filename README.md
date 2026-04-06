# Metis

AlphaZero-style chess engine built on the [luxia-base](https://github.com/LuxiaSL/luxia-base) transformer architecture. Research project exploring Muon optimizer, NCA pre-training, and geometric monitoring in self-play RL.

## Architecture

**Model**: Bidirectional transformer encoder (~36M params default) with:
- QK-norm, SwiGLU FFN, Grouped-Query Attention
- Optional Block Attention Residuals (AttnRes) for gradient flow equalization and late-layer health
- Dual heads: policy (AlphaZero 8x8x73 = 4672 moves) + value (scalar)
- Learned positional embeddings (no RoPE — fixed board geometry)

**Training**: AlphaZero self-play loop:
1. NCA bootstrap: pre-train attention on 8x8 cellular automata grid dynamics
2. Self-play: generate games via MCTS, train on (position, policy, value) tuples
3. Evaluate against Stockfish at various depths

**Key components from luxia-base**:
- Muon optimizer (Newton-Schulz orthogonalized momentum) resists representation collapse during RL
- NCA pre-pre-training bootstraps attention circuits before task-specific training
- Anamnesis geometric monitoring tracks representational health (RankMe, stable rank, anisotropy, WW alpha, TwoNN ID)

## Project Structure

```
src/
  model/transformer.py    — Chess transformer (adapted from luxia-base llama.py)
  chess/board.py          — Board + AlphaZero move encoding
  chess/mcts.py           — MCTS with virtual loss, tree reuse, batched eval
  chess/self_play.py      — Multiprocessing self-play (N CPU workers + GPU evaluator)
  chess/evaluation.py     — Stockfish gauntlet
  training/muon.py        — Muon optimizer (copied from luxia-base)
  training/replay_buffer.py — Circular experience buffer
  training/train.py       — Main training loop + NCA bootstrap
  nca/generator.py        — 8x8 NCA trajectory generator
  monitoring/geometric.py — Anamnesis metrics + AttnRes boundary discovery
configs/                  — Model size presets + training config reference
scripts/
  train.sh                — Launch script (single-GPU, DDP, smoke test)
  analyze_nca.py          — NCA checkpoint knee analysis
  nca_dashboard.py        — Live NCA training dashboard
```

## Quick Start

```bash
# Local smoke test (CPU, ~2 min)
bash scripts/train.sh --smoke

# GPU training with NCA bootstrap
python -m src.training.train \
    --model_size medium \
    --activation_checkpointing \
    --num_workers 64 \
    --nca_bootstrap \
    --monitor \
    --wandb

# With explicit AttnRes boundaries
python -m src.training.train \
    --model_size medium \
    --attn_res_boundaries 0,1,3,9 \
    --nca_bootstrap \
    --wandb

# Generate NCA dataset standalone (parallel)
python -m src.nca.generator \
    --output data/nca_seed17.pt \
    --seed 17 --num_rules 8000 --num_workers 64
```

## Model Sizes

| Config | Params | Layers | Hidden | Heads | Use |
|--------|--------|--------|--------|-------|-----|
| smoke | 819K | 4 | 128 | 4 | Unit tests |
| small | 6.5M | 8 | 256 | 4 | Fast iteration |
| medium | 36M | 12 | 512 | 8 | Main target |
| large | 47M | 16 | 512 | 8 | Scaling experiments |

## Key References

- [luxia-base](https://github.com/LuxiaSL/luxia-base) — Parent pretraining codebase
- AlphaZero (Silver et al., 2018) — Self-play + MCTS architecture
- Muon (MoonshotAI) — Spectral norm descent optimizer
