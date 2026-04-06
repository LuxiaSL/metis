#!/usr/bin/env bash
# Chess engine training launch script
#
# Single GPU:
#   bash scripts/train.sh
#
# Multi-GPU (DDP):
#   bash scripts/train.sh --ddp 2
#
# Quick smoke test:
#   bash scripts/train.sh --smoke

set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ddp)
            NUM_GPUS="${2:-1}"
            shift 2
            ;;
        --smoke)
            EXTRA_ARGS+=(
                --model_size smoke
                --mcts_simulations 10
                --games_per_iter 4
                --train_steps_per_iter 10
                --num_parallel_games 2
                --num_iterations 3
                --eval_every 0
                --save_every 0
            )
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching DDP training with $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29500 \
        -m src.training.train \
        "${EXTRA_ARGS[@]}"
else
    echo "Launching single-GPU training..."
    python -m src.training.train "${EXTRA_ARGS[@]}"
fi
