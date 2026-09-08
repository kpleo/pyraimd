#!/usr/bin/env bash
#SBATCH --job-name=w_spike_prod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Flagship-B (W) production closed loop: the thermal-spike surrogate of the
# cascade core, 686-atom bcc W box, NVE VelocityVerlet off the
# two-temperature initial condition (xyz carries momenta; the Runner
# thermalizes only when momenta are absent). Production spike is seed 2 —
# the bootstrap labeled seed-1 frames, so this trajectory is genuinely new.
# One 14 h wall segment per submission; RESUBMIT THE SAME SCRIPT to chain
# (loop.db presence -> --resume). DO NOT SUBMIT before the merge gate
# (w_bootstrap_merge.sbatch, ensemble held-out fMax <= ~0.35 eV/A) passes.
#
# Pre-registered protocol (research log 2026-08-30, blind):
# - rho_W is expected event-driven (0.1-0.3/step >> rho_Li=0.0222): first
#   acceptance streaks run STREAK_RHO=0.0 to measure the drift rate on the
#   broken streak (the flagship-A before/after protocol), then switch to
#   the measured value.
# - eps operating point: initial 0.2 eV/A; re-selected by the
#   refusal-curve logic once the W spread floor is measured (qhat*s_floor
#   margin), exactly as flagship A's 0.25 -> 1.0 selection.
# - Coverage will be reported on the 4000-10000 K subset (design-m3 red
#   flag), and fallback fraction vs spike temperature will be known before
#   writing.
# W QE protocol (ecut campaign validated): mv 0.01, beta 0.3, local-TF,
# david 8 + full_acc, nbnd headroom +50 (14 valence e-/W -> nbnd 4852),
# 70/700 Ry, conv 1e-6, startpot chaining.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
WORKDIR="$PROJECT/calculations/w_spike_prod"
RUN_ID=flagship-b-w-prod
EPS_ACC="${EPS_ACC:-0.2}"
STREAK_RHO="${STREAK_RHO:-0.0}"
EXPLORE_FRAC="${EXPLORE_FRAC:-0.0}"

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x"
COMMON_ARGS=(
  --atomsXYZ "$PROJECT/inputs/generated/w_spike_686_s2.xyz"
  --model "$PROJECT/software/mace-mp-0b3-medium.model"
          "$PROJECT/software/mace-mpa-0-medium.model"
  --pseudo-dir "$PROJECT/inputs"
  --pw-cmd "$PW"
  --run-id "$RUN_ID"
  --workdir "$WORKDIR"
  --n-steps 4000 --temperature 300 --timestep-fs 0.5
  --eps-acc "$EPS_ACC" --alpha 0.05 --window 64 --w-min 16
  --streak-rho "$STREAK_RHO"
  --explore-frac "$EXPLORE_FRAC"
  --n-label 8 --epochs 30 --train-window 16
  --trainable-filters readout products
  --torch-threads 16
  --metallic --smearing mv --degauss 0.01 --mixing-beta 0.3
  --mixing-ndim 12 --diago-david-ndim 8 --diago-full-acc --nbnd-headroom 50
  --electron-maxstep 200 --ecutwfc 70 --ecutrho 700 --conv-thr 1e-6
  --startpot-chain
)

install -d -m 700 "$WORKDIR"
if [ -f "$WORKDIR/loop.db" ]; then
  MODE="--resume"
else
  MODE="--init-committee $PROJECT/sync/committee_w_bootstrap.pt"
fi
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID mode=$MODE eps_acc=$EPS_ACC streak_rho=$STREAK_RHO explore_frac=$EXPLORE_FRAC"

cd "$PROJECT"
# shellcheck disable=SC2086
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  "${COMMON_ARGS[@]}" $MODE

echo "done"
