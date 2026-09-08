#!/usr/bin/env bash
#SBATCH --job-name=w_spike_prod432
#SBATCH --partition=9242
#SBATCH --nodes=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=360G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Flagship-B (W) production closed loop on the 432-atom fallback cell
# (design-m3 v2 fallback clause; 686 label RAM wall + Act-2 label
# economics — see research log 2026-08-31). The thermal-spike surrogate
# of the cascade core, NVE VelocityVerlet off the two-temperature initial
# condition (xyz carries momenta; the Runner thermalizes only when they
# are absent). Production spike is seed 2 (bootstrap labeled seed 1).
# One 48 h wall segment per submission; RESUBMIT THE SAME SCRIPT to chain
# (loop.db presence -> --resume). DO NOT SUBMIT before the merge gate
# (w_bootstrap_merge_432.sbatch, ensemble held-out fMax <= ~0.35 eV/A).
#
# Runs on 9242 because a 432-atom label needs ~250 GB: QE labels fire on
# 192 ranks across both nodes (-ndiag 24); between labels the loop's
# committee inference/fine-tuning uses the head node's cores. Node-hour
# price of idle-between-labels noted; the label is the scarce resource.
#
# Pre-registered protocol (research log 2026-08-30, blind):
# - rho_W expected event-driven (0.1-0.3/step >> rho_Li=0.0222): first
#   acceptance streaks run STREAK_RHO=0.0 to measure the drift rate on
#   the broken streak (flagship-A before/after protocol), then switch.
# - eps operating point: initial 0.2 eV/A; re-selected by the
#   refusal-curve logic once the W spread floor is measured.
# - Coverage reported on the 4000-10000 K subset (design-m3 red flag);
#   fallback fraction vs spike temperature measured before writing.
# W QE protocol (ecut campaign + probe 7622601): mv 0.01, beta 0.3,
# local-TF, david 8 + full_acc, nbnd headroom +50 (nbnd 3074), 70/700 Ry,
# conv 1e-6, startpot chaining.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
WORKDIR="$PROJECT/calculations/w_spike_prod_432"
RUN_ID=flagship-b-w-prod-432
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

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 192 pw.x -ndiag 24"
COMMON_ARGS=(
  --atomsXYZ "$PROJECT/inputs/generated/w_spike_432_s2.xyz"
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
  MODE="--init-committee $PROJECT/sync/committee_w432_bootstrap.pt"
fi
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID mode=$MODE eps_acc=$EPS_ACC streak_rho=$STREAK_RHO explore_frac=$EXPLORE_FRAC"

cd "$PROJECT"
# shellcheck disable=SC2086
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  "${COMMON_ARGS[@]}" $MODE

echo "done"
