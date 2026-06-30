#!/usr/bin/env bash
# Z01 AMP staged training recipe.
#
# Phase 1: flat warmup — learn basic standing/walking with conservative settings.
# Phase 2: flat full   — resume warmup checkpoint on flat with full velocity range.
# Phase 3: rough       — resume flat checkpoint on rough terrain + recovery delay.
#
# Usage:
#   bash scripts/train_z01_staged.sh 1                   # phase 1 only
#   bash scripts/train_z01_staged.sh 2 <warmup_run_dir>  # phase 2
#   bash scripts/train_z01_staged.sh 3 <flat_run_dir>    # phase 3
#
# Optional env vars:
#   NUM_ENVS=4096          parallel env count
#   VIDEO=1                record training videos every 2000 steps
#
# Monitor these each phase:
#   skipped_non_finite_batches == 0
#   Mean episode length trending up (warmup target: >100 steps)
#   bad_base_height termination rate dropping below ~50%
#   Mean reward is finite (typically -1 ~ -50)

set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
VIDEO="${VIDEO:-0}"
PHASE="${1:-1}"
CHECKPOINT_RUN="${2:-}"

VIDEO_ARGS=()
if [[ "${VIDEO}" == "1" ]]; then
  VIDEO_ARGS=(--video=true --video-interval=2000 --video-length=200)
fi

case "${PHASE}" in
  1)
    python scripts/train.py Z01-AMP-Flat-Warmup \
      --env.scene.num-envs="${NUM_ENVS}" \
      --agent.run-name=warmup \
      "${VIDEO_ARGS[@]}"
    ;;
  2)
    if [[ -z "${CHECKPOINT_RUN}" ]]; then
      echo "Phase 2 requires warmup run dir, e.g.:"
      echo "  bash scripts/train_z01_staged.sh 2 logs/rsl_rl/z01_amp_warmup/<timestamp>_warmup"
      exit 1
    fi
    python scripts/train.py Z01-AMP-Flat \
      --env.scene.num-envs="${NUM_ENVS}" \
      --agent.run-name=flat \
      --agent.resume=true \
      --agent.load-run="${CHECKPOINT_RUN}" \
      --agent.load-checkpoint=model_10000.pt \
      "${VIDEO_ARGS[@]}"
    ;;
  3)
    if [[ -z "${CHECKPOINT_RUN}" ]]; then
      echo "Phase 3 requires flat run dir, e.g.:"
      echo "  bash scripts/train_z01_staged.sh 3 logs/rsl_rl/z01_amp_locomotion/<timestamp>_flat"
      exit 1
    fi
    python scripts/train.py Z01-AMP-Rough \
      --env.scene.num-envs="${NUM_ENVS}" \
      --agent.run-name=rough \
      --agent.resume=true \
      --agent.load-run="${CHECKPOINT_RUN}" \
      "${VIDEO_ARGS[@]}"
    ;;
  *)
    echo "Unknown phase: ${PHASE} (use 1, 2, or 3)"
    exit 1
    ;;
esac
