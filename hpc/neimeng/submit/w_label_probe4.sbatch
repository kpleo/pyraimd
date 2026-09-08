#!/usr/bin/env bash
#SBATCH --job-name=w_label_probe4
#SBATCH --partition=9242
#SBATCH --nodes=4
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=4
#SBATCH --mem=360G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# W label probe round 4 (7620792 postmortem): pure-MPI layouts cannot fit
# the 686-atom label (footprint 782->1469 GB grows WITH rank count). Move
# to the hybrid shape: 96 ranks x 4 OpenMP threads over 4 nodes (24 ranks
# per node), pw.x -ndiag 24 — the per-rank replicated workspace is cut 4x
# by the rank reduction while threads cover the cores; model footprint
# ~250 GB/node. Success = SCF converges (rc=0) + wall time recorded for
# the label-array sizing. diago_full_acc/ndim=8 kept (protocol must match
# the ecut campaign). 9242 contract: 96 CPU/node, --mem=360G, 4-node cap.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

PW="env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 mpirun -np 96 pw.x -ndiag 24"

RUN_DIR="$PROJECT/calculations/w_label_probe4_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS cpt=$SLURM_CPUS_PER_TASK pw=$PW"
t0=$(date +%s)
RC=0
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
  --atomsXYZ "$PROJECT/calculations/w_bootstrap_frames/all.extxyz" \
  --frame-index 0 \
  --pseudo-dir "$PROJECT/inputs" \
  --pw-cmd "$PW" \
  --metallic --smearing mv --degauss 0.01 --mixing-beta 0.3 \
  --nbnd-headroom 50 --electron-maxstep 200 \
  --ecutwfc 70 --ecutrho 700 --conv-thr 1e-6 \
  --out-jsonl "$RUN_DIR/labels.jsonl" \
  --workdir "$RUN_DIR/chain" || RC=$?
t1=$(date +%s)

echo "rc=$RC wall_s=$((t1 - t0))"
[ -f "$RUN_DIR/labels.jsonl" ] && tail -1 "$RUN_DIR/labels.jsonl" | cut -c1-200
echo "done"
