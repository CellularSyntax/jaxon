#!/bin/bash
# Submit the three manuscript-sweep phases as separate SLURM jobs,
# chained so each starts only after the previous finishes (regardless
# of success/failure — `afterany`).  That way:
#
#   - Each phase has its own walltime budget (no oversized monolithic
#     job that gets killed before Phase 3 can finish)
#   - Each phase has its own log + output directory
#   - You can `scancel <jobid>` for any one phase without touching the
#     others
#   - Concurrent compile contention is avoided — Phase N's JIT only
#     starts after Phase N-1 has released the node
#
# Usage (from the project root, after `git pull`):
#
#   bash slurm/submit_selectivity_manuscript.sh
#
# Override the seed range for Phase 3:
#
#   PHASE3_SEED_START=20 PHASE3_SEED_END=40 \
#     bash slurm/submit_selectivity_manuscript.sh
#
# Skip a phase (e.g. you already have the Phase 1 result):
#
#   SKIP_PHASE1=1 bash slurm/submit_selectivity_manuscript.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PHASE3_SEED_START="${PHASE3_SEED_START:-0}"
PHASE3_SEED_END="${PHASE3_SEED_END:-100}"

LAST_JOB=""

submit_with_dep() {
  local script="$1"
  shift
  local extra_args=("$@")
  if [ -n "${LAST_JOB}" ]; then
    local jid
    jid=$(sbatch --parsable --dependency=afterany:"${LAST_JOB}" \
          "${extra_args[@]}" "${script}")
  else
    local jid
    jid=$(sbatch --parsable "${extra_args[@]}" "${script}")
  fi
  echo "${jid}"
}

if [ "${SKIP_PHASE1:-0}" != "1" ]; then
  JID1=$(submit_with_dep slurm/run_phase1_mixed_diam.sbatch)
  echo "Phase 1 (mixed-diameter rect+wave) submitted as job ${JID1}"
  LAST_JOB="${JID1}"
else
  echo "Phase 1 skipped (SKIP_PHASE1=1)"
fi

if [ "${SKIP_PHASE2:-0}" != "1" ]; then
  JID2=$(submit_with_dep slurm/run_phase2_lbfgs.sbatch)
  echo "Phase 2 (LBFGS rect at N=200) submitted as job ${JID2}"
  LAST_JOB="${JID2}"
else
  echo "Phase 2 skipped (SKIP_PHASE2=1)"
fi

if [ "${SKIP_PHASE3:-0}" != "1" ]; then
  JID3=$(submit_with_dep slurm/run_phase3_manuscript_sweep.sbatch \
          --export=ALL,SEED_START="${PHASE3_SEED_START}",SEED_END="${PHASE3_SEED_END}")
  echo "Phase 3 (manuscript sweep, seeds ${PHASE3_SEED_START}..$((PHASE3_SEED_END-1))) submitted as job ${JID3}"
  LAST_JOB="${JID3}"
else
  echo "Phase 3 skipped (SKIP_PHASE3=1)"
fi

echo
echo "All requested phases queued.  Check progress with:"
echo "  squeue -u \${USER}"
echo "  tail -f logs/jaxfibers-phase*-*.out"
