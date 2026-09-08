#!/usr/bin/env bash
#SBATCH --job-name=li_slab_control
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Bare-Li-slab control (from the physics review of flagship A): the EXACT
# 36-atom Li(100) 4-layer slab of the production initial frame (step -1,
# electrolyte removed, same 10.53x10.53x52.15 A cell), fresh 300 K MB
# velocities, propagated with pure DFT (eps_acc=0 -> every step refuses ->
# engine forces every step = plain AIMD with our store + decision record).
# Question: does the (100) layering persist at 300 K WITHOUT the
# electrolyte? Yes -> the production layering loss is adsorption-driven
# (physical); no -> the slab setup itself is unstable (flagship claim
# retreat). n-label huge -> no fine-tunes; the committee rides along as a
# free diagnostic (its forces are never used for propagation).
# Resumable via loop.db presence (same chaining convention as production).
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
WORKDIR="$PROJECT/calculations/li_slab_control"
RUN_ID=li-slab-control

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x"
COMMON_ARGS=(
  --atomsXYZ "$PROJECT/inputs/generated/li_slab_control.xyz"
  --model "$PROJECT/software/mace-mp-0b3-medium.model"
          "$PROJECT/software/mace-mpa-0-medium.model"
  --pseudo-dir "$PROJECT/inputs"
  --pw-cmd "$PW"
  --run-id "$RUN_ID"
  --workdir "$WORKDIR"
  --n-steps 500 --temperature 300 --timestep-fs 0.5
  --eps-acc 0.0 --alpha 0.05 --window 64 --w-min 16
  --streak-rho 0.0 --explore-frac 0.0
  --n-label 100000 --epochs 30 --train-window 16
  --trainable-filters readout products
  --torch-threads 16
  --metallic --smearing fd --degauss 0.02 --mixing-beta 0.20
  --mixing-ndim 12 --diago-david-ndim 8 --nbnd-headroom 50
  --electron-maxstep 200 --ecutwfc 60 --ecutrho 600 --conv-thr 1e-6
  --startpot-chain
)

install -d -m 700 "$WORKDIR"
if [ -f "$WORKDIR/loop.db" ]; then
  MODE="--resume"
else
  MODE="--init-committee $PROJECT/sync/committee_bootstrap.pt"
fi
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID mode=$MODE eps_acc=0 (pure-AIMD control)"

cd "$PROJECT"
# shellcheck disable=SC2086
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  "${COMMON_ARGS[@]}" $MODE

echo "done"
