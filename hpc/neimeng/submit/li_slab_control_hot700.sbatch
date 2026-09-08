#!/usr/bin/env bash
#SBATCH --job-name=li_slab_control_hot700
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# HOT bare-Li-slab control (pre-registered 2026-09-04, docs/research-log.md).
# Closes the temperature confound of the 300 K control: the loop is NVE
# (thermalize at init only), so the 300 K control equilibrates to ~138 K
# while the production slab is heated by electrolyte reactions to <350 K.
# Identical 36-atom Li(100) slab, identical pure-DFT protocol (eps_acc=0
# sentinel), identical 500 steps — the ONLY difference is the 700 K MB
# initialization: NVE equipartition settles near ~350 K and the early
# transient passes through >350 K, i.e. harsher than production. Verdict
# tree: keeps 4 layers -> temperature confound closed, interfacial-chemistry
# reconstruction claim stands; loses layers -> structural claim retreats to
# thermal+interfacial wording. Neimeng A "normal" contract: 28 CPU/node,
# --mem=120G, no GPU.
# INPUT NOTE: the xyz carries fresh 700 K MB momenta baked in by
# make_li_slab_control.py --temperature 700 (seed 20260902); the runner
# honors input momenta over --temperature (resume semantics), so the
# momenta-bearing input is the load-bearing piece. The first submission of
# this job (7623144) accidentally reused the 300 K input and produced a
# replica of the 300 K control; preserved at calculations/
# li_slab_control_hot700_momenta_replica for the record.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
WORKDIR="$PROJECT/calculations/li_slab_control_hot700"
RUN_ID=li-slab-control-hot700

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x"
COMMON_ARGS=(
  --atomsXYZ "$PROJECT/inputs/generated/li_slab_control_hot700.xyz"
  --model "$PROJECT/software/mace-mp-0b3-medium.model"
          "$PROJECT/software/mace-mpa-0-medium.model"
  --pseudo-dir "$PROJECT/inputs"
  --pw-cmd "$PW"
  --run-id "$RUN_ID"
  --workdir "$WORKDIR"
  --n-steps 500 --temperature 700 --timestep-fs 0.5
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
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID mode=$MODE eps_acc=0 (pure-AIMD hot control, 700 K init)"

cd "$PROJECT"
# shellcheck disable=SC2086
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  "${COMMON_ARGS[@]}" $MODE

echo "done"
