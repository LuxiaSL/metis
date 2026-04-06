#!/usr/bin/env bash
# Self-play training launch script for node2
# Usage: bash scripts/run_selfplay.sh [--validate]

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=7

PYTHON=~/luxi-files/.venv-shared/bin/python

if [[ "${1:-}" == "--validate" ]]; then
    echo "Running validation (3 iters, 10 sims, batch 32)..."
    exec $PYTHON -m src.training.train \
        --model_size medium \
        --attn_res_boundaries 0,1,3,9 \
        --activation_checkpointing \
        --nca_bootstrap \
        --batch_size 32 \
        --num_workers 8 \
        --num_parallel_games 8 \
        --mcts_simulations 10 \
        --games_per_iter 8 \
        --train_steps_per_iter 10 \
        --num_iterations 3 \
        --eval_every 0 \
        --save_every 3 \
        --monitor \
        --monitor_tier1_every 10 \
        --checkpoint_dir checkpoints/selfplay \
        --stockfish_path ~/luxi-files/bin/stockfish
else
    echo "Running full self-play training..."
    exec $PYTHON -m src.training.train \
        --model_size medium \
        --attn_res_boundaries 0,1,3,9 \
        --activation_checkpointing \
        --nca_bootstrap \
        --batch_size 32 \
        --num_workers 64 \
        --num_parallel_games 64 \
        --mcts_simulations 800 \
        --games_per_iter 256 \
        --train_steps_per_iter 1000 \
        --num_iterations 200 \
        --eval_every 10 \
        --save_every 10 \
        --monitor \
        --monitor_tier1_every 500 \
        --resume \
        --checkpoint_dir checkpoints/selfplay \
        --wandb \
        --wandb_api_key "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" \
        --wandb_name selfplay-attnres-0139 \
        --stockfish_path ~/luxi-files/bin/stockfish
fi
