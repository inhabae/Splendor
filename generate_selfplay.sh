#!/bin/bash
#SBATCH -p 256x44
#SBATCH -J splendor-generate-selfplay
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH -o /mnt/beegfs/home/ibae/scratch2/splendor/logs/%x-%j.out
#SBATCH -e /mnt/beegfs/home/ibae/scratch2/splendor/logs/%x-%j.err
#SBATCH --exclusive

# ─── generate_selfplay.sh ────────────────────────────────────────────────────
#
# Script 1 of 3: Teacher self-play data generation
#
# Runs self-play using a fixed teacher checkpoint in an infinite loop,
# writing one .npz session file per iteration into SELFPLAY_OUT_DIR.
# The learner (train_from_selfplay.sh) reads from the same directory.
#
# Key env overrides:
#   TEACHER_CKPT     - path to the teacher checkpoint to use (required)
#   SELFPLAY_OUT_DIR - where to write session .npz files
#   GAMES_PER_ITER   - games per iteration (default 500)
#   COLLECTOR_WORKERS, MCTS_SIMS, MAX_TURNS, SEED
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${REPO:-/mnt/beegfs/home/ibae/scratch2/splendor}"
cd "${REPO}"

module load ohpc
module load miniconda3/3.13
eval "$(conda shell.bash hook)"
conda activate splendor

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Run identity ──────────────────────────────────────────────────────────────
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_TAG="${RUN_TAG}_a${SLURM_ARRAY_TASK_ID}"
fi

BASE_SEED="${BASE_SEED:-123}"
SEED="${SEED:-${BASE_SEED}}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SELFPLAY_OUT_DIR="${SELFPLAY_OUT_DIR:-${REPO}/nn_artifacts/teacher_learner/selfplay_data}"
LOG_DIR="${LOG_DIR:-${REPO}/nn_artifacts/teacher_learner/logs/generate}"
STATE_DIR="${STATE_DIR:-${REPO}/nn_artifacts/teacher_learner/state}"

mkdir -p "${SELFPLAY_OUT_DIR}" "${LOG_DIR}" "${STATE_DIR}"

# ── Hyperparameters ───────────────────────────────────────────────────────────
TEACHER_CKPT="${TEACHER_CKPT:-}"
GAMES_PER_ITER="${GAMES_PER_ITER:-5000}"
COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-40}"
MCTS_TREE_WORKERS="${MCTS_TREE_WORKERS:-1}"
MCTS_SIMS="${MCTS_SIMS:-800}"
MAX_TURNS="${MAX_TURNS:-80}"
# Playout cap randomization
FULL_SEARCH_SIMS="${FULL_SEARCH_SIMS:-800}"
FAST_SEARCH_SIMS="${FAST_SEARCH_SIMS:-200}"
FULL_SEARCH_PROB="${FULL_SEARCH_PROB:-0.25}"

if [[ -z "${TEACHER_CKPT}" ]]; then
  echo "generate_selfplay_error=TEACHER_CKPT is not set" >&2
  exit 1
fi
if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "generate_selfplay_error=TEACHER_CKPT not found: ${TEACHER_CKPT}" >&2
  exit 1
fi

# ── State (track iteration and seed across restarts) ──────────────────────────
STATE_FILE="${STATE_DIR}/generate_state.env"
ITERATION=1
NEXT_SEED="${SEED}"

if [[ -f "${STATE_FILE}" && "${IGNORE_SAVED_STATE:-0}" != "1" ]]; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
fi

echo "generate_selfplay_run_tag=${RUN_TAG}"
echo "generate_selfplay_teacher_ckpt=${TEACHER_CKPT}"
echo "generate_selfplay_out_dir=${SELFPLAY_OUT_DIR}"
echo "generate_selfplay_games_per_iter=${GAMES_PER_ITER}"
echo "generate_selfplay_workers=${COLLECTOR_WORKERS}"
echo "generate_selfplay_mcts_tree_workers=${MCTS_TREE_WORKERS}"
echo "generate_selfplay_mcts_sims=${MCTS_SIMS}"
echo "generate_selfplay_initial_iteration=${ITERATION}"
echo "generate_selfplay_initial_seed=${NEXT_SEED}"

# ── Main loop ─────────────────────────────────────────────────────────────────
while true; do
  LOG_PATH="${LOG_DIR}/iter_$(printf '%06d' "${ITERATION}").log"
  ITER_SEED="${NEXT_SEED}"

  CMD=(
    python -m nn.scripts.generate_selfplay
    --checkpoint        "${TEACHER_CKPT}"
    --out-dir           "${SELFPLAY_OUT_DIR}"
    --games             "${GAMES_PER_ITER}"
    --workers           "${COLLECTOR_WORKERS}"
    --mcts-tree-workers "${MCTS_TREE_WORKERS}"
    --mcts-sims         "${MCTS_SIMS}"
    --max-turns         "${MAX_TURNS}"
    --seed              "${ITER_SEED}"
    --full-search-sims  "${FULL_SEARCH_SIMS}"
    --fast-search-sims  "${FAST_SEARCH_SIMS}"
    --full-search-prob  "${FULL_SEARCH_PROB}"
  )

  echo "generate_selfplay_iter=${ITERATION} seed=${ITER_SEED} log=${LOG_PATH}"
  "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"

  # Verify a session file was actually written
  NEW_SESSION="$(awk '/^generate_selfplay_done /{match($0, /path=([^ ]+)/, m); print m[1]; exit}' "${LOG_PATH}")"
  if [[ -z "${NEW_SESSION}" || ! -f "${NEW_SESSION}" ]]; then
    echo "generate_selfplay_error=failed to find new session file in log ${LOG_PATH}" >&2
    exit 1
  fi

  NEXT_SEED=$(( ITER_SEED + GAMES_PER_ITER ))

  # Persist state atomically
  cat > "${STATE_FILE}.tmp.$$" <<EOF
ITERATION=$(( ITERATION + 1 ))
NEXT_SEED=${NEXT_SEED}
SEED=${SEED}
RUN_TAG=${RUN_TAG}
EOF
  mv "${STATE_FILE}.tmp.$$" "${STATE_FILE}"

  echo "generate_selfplay_iter_done=${ITERATION} session=${NEW_SESSION} next_seed=${NEXT_SEED}"
  ITERATION=$(( ITERATION + 1 ))
done