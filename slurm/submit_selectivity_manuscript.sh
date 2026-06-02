#!/bin/bash
# Submit the three manuscript-sweep phases as separate SLURM jobs.
#
# All three phases are logically independent — no shared inputs,
# outputs, or filesystem locks — so they run in parallel by default
# and finish in max(Phase 1, Phase 2, Phase 3) = ~24 h, instead of
# the ~26 h sequential sum.
#
# If you want strict serialisation (e.g. you've already seen earlier
# in this project that concurrent JAX compiles on the same A16 node
# fight for CPU/RAM during the remat planner pass), set CHAIN=1 — then
# each phase starts only after the previous finishes (afterany).
#
# Usage (from the project root, after `git pull`):
#
#   bash slurm/submit_selectivity_manuscript.sh
#
# Variants:
#
#   CHAIN=1 bash slurm/submit_selectivity_manuscript.sh
#     → Phases run sequentially via SLURM --dependency=afterany.
#
#   PHASE3_SEED_START=20 PHASE3_SEED_END=40 \
#     bash slurm/submit_selectivity_manuscript.sh
#     → Custom seed range for Phase 3.
#
#   SKIP_PHASE1=1 bash slurm/submit_selectivity_manuscript.sh
#     → Skip a phase you already have results for.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PHASE3_SEED_START="${PHASE3_SEED_START:-0}"
PHASE3_SEED_END="${PHASE3_SEED_END:-100}"
CHAIN="${CHAIN:-0}"

LAST_JOB=""

submit_one() {
  local script="$1"
  shift
  local extra_args=("$@")
  local dep_arg=()
  if [ "${CHAIN}" = "1" ] && [ -n "${LAST_JOB}" ]; then
    dep_arg=(--dependency=afterany:"${LAST_JOB}")
  fi
  sbatch --parsable "${dep_arg[@]}" "${extra_args[@]}" "${script}"
}

if [ "${SKIP_PHASE1:-0}" != "1" ]; then
  JID1=$(submit_one slurm/run_phase1_mixed_diam.sbatch)
  echo "Phase 1 (mixed-diameter rect+wave) submitted as job ${JID1}"
  LAST_JOB="${JID1}"
else
  echo "Phase 1 skipped (SKIP_PHASE1=1)"
fi

if [ "${SKIP_PHASE2:-0}" != "1" ]; then
  JID2=$(submit_one slurm/run_phase2_lbfgs.sbatch)
  echo "Phase 2 (LBFGS rect at N=200) submitted as job ${JID2}"
  LAST_JOB="${JID2}"
else
  echo "Phase 2 skipped (SKIP_PHASE2=1)"
fi

if [ "${SKIP_PHASE3:-0}" != "1" ]; then
  JID3=$(submit_one slurm/run_phase3_manuscript_sweep.sbatch \
          --export=ALL,SEED_START="${PHASE3_SEED_START}",SEED_END="${PHASE3_SEED_END}")
  echo "Phase 3 (manuscript sweep, seeds ${PHASE3_SEED_START}..$((PHASE3_SEED_END-1))) submitted as job ${JID3}"
  LAST_JOB="${JID3}"
else
  echo "Phase 3 skipped (SKIP_PHASE3=1)"
fi

if [ "${CHAIN}" = "1" ]; then
  echo
  echo "CHAIN=1 — phases will run sequentially (afterany dependency)."
else
  echo
  echo "Phases will run in parallel.  If they all land on the same A16"
  echo "node their JAX compiles may contend for CPU/RAM (~10 min extra)."
  echo "Re-submit with CHAIN=1 if you want strict sequential execution."
fi

echo
echo "Check progress:"
echo "  squeue -u \${USER}"
echo "  tail -f logs/jaxfibers-phase*-*.out"
