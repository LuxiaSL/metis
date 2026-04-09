#!/usr/bin/env bash
# Launch a training job from a YAML config file.
# Usage: bash scripts/launch.sh configs/selfplay_v8.yaml [--dry-run]
#
# Reads the YAML, builds CLI args, submits via Heimdall.
# Requires: yq (or python yaml parsing as fallback)

set -euo pipefail

CONFIG="${1:?Usage: bash scripts/launch.sh <config.yaml> [--dry-run]}"
DRY_RUN="${2:-}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

# Parse YAML with Python (no yq dependency)
read_yaml() {
    python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

def get(path, default=''):
    parts = path.split('.')
    v = cfg
    for p in parts:
        if isinstance(v, dict) and p in v:
            v = v[p]
        else:
            return default
    return v

# Model
model_size = get('model.size', 'medium')
boundaries = get('model.attn_res_boundaries', '')
ckpt = get('model.activation_checkpointing', False)
compile_flag = get('model.compile', False)

# Self-play
workers = get('self_play.num_workers', 96)
parallel = get('self_play.num_parallel_games', 96)
sims = get('self_play.mcts_simulations', 400)
games = get('self_play.games_per_iter', 128)
max_moves = get('self_play.max_moves', 200)
nvl = get('self_play.num_virtual_leaves', 8)
temp_thresh = get('self_play.temperature_threshold', 200)
dir_eps = get('self_play.dirichlet_epsilon', 0.4)

# MCTS algorithm
mcts_algo = get('self_play.mcts_algorithm', 'alphazero')
gumbel_k = get('self_play.gumbel_K', 16)
gumbel_c = get('self_play.gumbel_c_visit', 50.0)

# Training
batch = get('training.batch_size', 32)
steps = get('training.train_steps_per_iter', 1000)
iters = get('training.num_iterations', 200)
buffer = get('training.buffer_size', 1000000)
decisive = get('training.decisive_boost', 1.0)
q_blend = get('training.q_blend', 0.0)
sf_anchor = get('training.sf_anchor_positions', 0)
sf_anchor_depth = get('training.sf_anchor_depth', 8)

# Self-play (continued)
pcr_frac = get('self_play.playout_cap_fraction', 1.0)
fast_move_sims = get('self_play.fast_move_sims', 0)
mat_adj_thresh = get('self_play.material_adjudication_threshold', 9.0)

# Eval
eval_every = get('eval.eval_every', 5)
eval_gpd = get('eval.eval_games_per_depth', 10)
eval_sims = get('eval.eval_mcts_sims', 200)
eval_depths = get('eval.eval_depths', '1,3,5')

# Monitoring
monitor = get('monitoring.monitor', True)
t1 = get('monitoring.monitor_tier1_every', 500)
t2 = get('monitoring.monitor_tier2_every', 5000)

# Checkpoint
save_every = get('checkpoint.save_every', 5)
ckpt_dir = get('checkpoint.checkpoint_dir', 'checkpoints/selfplay')
resume = get('checkpoint.resume', True)

# NCA
nca = get('nca.nca_bootstrap', True)

# Infra
node = get('infra.node', 'node1')
gpu = get('infra.gpu', 7)
sf = get('infra.stockfish_path', '~/luxi-files/bin/stockfish')
wandb_flag = get('infra.wandb', True)
wandb_name = get('infra.wandb_name', '')

# Build args
args = []
args.append(f'--model_size {model_size}')
if boundaries:
    args.append(f'--attn_res_boundaries {boundaries}')
if ckpt:
    args.append('--activation_checkpointing')
if compile_flag:
    args.append('--compile')
args.append(f'--num_workers {workers}')
args.append(f'--num_parallel_games {parallel}')
args.append(f'--mcts_simulations {sims}')
args.append(f'--games_per_iter {games}')
args.append(f'--train_steps_per_iter {steps}')
args.append(f'--num_iterations {iters}')
args.append(f'--batch_size {batch}')
args.append(f'--buffer_size {buffer}')
args.append(f'--temperature_threshold {temp_thresh}')
args.append(f'--dirichlet_epsilon {dir_eps}')
args.append(f'--num_virtual_leaves {nvl}')
args.append(f'--mcts_algorithm {mcts_algo}')
if mcts_algo == 'gumbel':
    args.append(f'--gumbel_K {gumbel_k}')
    args.append(f'--gumbel_c_visit {gumbel_c}')
if float(decisive) > 1.0:
    args.append(f'--decisive_boost {decisive}')
if float(q_blend) > 0.0:
    args.append(f'--q_blend {q_blend}')
if float(pcr_frac) < 1.0:
    args.append(f'--playout_cap_fraction {pcr_frac}')
if int(fast_move_sims) > 0:
    args.append(f'--fast_move_sims {fast_move_sims}')
args.append(f'--material_adjudication_threshold {mat_adj_thresh}')
if int(sf_anchor) > 0:
    args.append(f'--sf_anchor_positions {sf_anchor}')
    args.append(f'--sf_anchor_depth {sf_anchor_depth}')
args.append(f'--eval_every {eval_every}')
args.append(f'--eval_games_per_depth {eval_gpd}')
args.append(f'--eval_mcts_sims {eval_sims}')
args.append('--eval_depths ' + str(eval_depths).replace(',', ' '))
if monitor:
    args.append('--monitor')
    args.append(f'--monitor_tier1_every {t1}')
    args.append(f'--monitor_tier2_every {t2}')
args.append(f'--save_every {save_every}')
args.append(f'--checkpoint_dir {ckpt_dir}')
if resume:
    args.append('--resume')
if nca:
    args.append('--nca_bootstrap')
if wandb_flag:
    args.append('--wandb')
args.append(f'--stockfish_path {sf}')

print(f'NODE=\"{node}\"')
print(f'GPU=\"{gpu}\"')
print(f'WANDB_NAME=\"{wandb_name}\"')
print('ARGS=\"' + ' '.join(args) + '\"')
"
}

TMPENV=$(mktemp)
read_yaml > "$TMPENV"
source "$TMPENV"
rm -f "$TMPENV"

# Load credentials
WANDB_KEY=$(python3 -c "
with open('$HOME/.claude/projects/-home-luxia-projects-metis/memory/credentials.md') as f:
    for line in f:
        if 'wandb_v1_' in line:
            print(line.split('wandb_v1_')[0] + 'wandb_v1_' + line.split('wandb_v1_')[1].strip())
            break
" 2>/dev/null | grep -oP 'wandb_v1_\S+' || echo "")

if [[ -n "$WANDB_KEY" ]]; then
    ARGS="$ARGS --wandb_api_key $WANDB_KEY"
fi

# Generate wandb run name from config filename if not set
if [[ -z "$WANDB_NAME" ]]; then
    WANDB_NAME="$(basename "$CONFIG" .yaml)-$(date +%m%d-%H%M)"
fi
ARGS="$ARGS --wandb_name $WANDB_NAME"

# Build the full command
# Python path uses ~ which expands on the node (athuser), not locally
CMD="cd ~/luxi-files/metis && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 ~/luxi-files/.venv-shared/bin/python -m src.training.train $ARGS"
JOB_NAME="metis-$(basename "$CONFIG" .yaml)"

echo "=== Launch Config ==="
echo "Node:    $NODE"
echo "GPU:     $GPU"
echo "Job:     $JOB_NAME"
echo "Wandb:   $WANDB_NAME"
echo "Config:  $CONFIG"
echo ""
echo "Command:"
echo "  $CMD"
echo ""

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo "[dry-run] Would submit to Heimdall. Exiting."
    exit 0
fi

# Submit
JOB_ID=$(heimdall submit "$CMD" \
    --name "$JOB_NAME" \
    --gpus 0 \
    --node "$NODE" \
    --estimated 6000 \
    --max-retries 0 2>&1 | grep -oP '[a-f0-9]{12}')

echo "Submitted: $JOB_ID"
echo "Logs: ssh $NODE \"tail -50 /tmp/heimdall_${JOB_ID}.log\""
echo "Watch: heimdall watch $JOB_ID"
