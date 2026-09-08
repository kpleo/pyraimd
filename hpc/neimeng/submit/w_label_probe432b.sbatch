#!/usr/bin/env bash
#SBATCH --job-name=w_label_probe432b
#SBATCH --partition=9242
#SBATCH --nodes=2
#SBATCH --ntasks=192
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# W 432-cell label probe: one frame (index 0, bulk 300 K) on TWO 9242
# node, 192 ranks pure MPI (2 nodes), -ndiag 24. The 432-atom label (nbnd 3074,
# 70/700 Ry) should fit in ~250 GB and run ~minutes; this probe verifies
# fit + measures the single-label wall time for the label-array sizing.
# Success = rc=0 + "convergence has been achieved" + wall_s. diago
# full_acc/ndim=8 kept (ecut-campaign protocol).
# 9242 contract: 96 CPU/node, --mem=360G, 4-node cap (docs/hpc.md).

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 192 pw.x -ndiag 24"

RUN_DIR="$PROJECT/calculations/w_label_probe432b_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID pw=$PW"
t0=$(date +%s)
RC=0
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/scf_check.py" \
  --atomsXYZ "$PROJECT/calculations/w_bootstrap_frames_432/all.extxyz" \
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
