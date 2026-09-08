#!/usr/bin/env bash
#SBATCH --job-name=w_label_probe2
#SBATCH --partition=9242
#SBATCH --nodes=3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# W label probe round 2 (7620784 postmortem): the 686-atom label's TOTAL
# footprint is ~815 GB regardless of rank count -> single 375G node cannot
# hold it; the footprint only divides across NODES. Three 9242 nodes
# (3x96=288 ranks, -ndiag 24 for the dense-diag groups) puts the estimate
# at ~272 GB/node. Frame 0; success = SCF converges (rc=0) with wall time
# recorded for the label-array sizing. diago_full_acc/ndim=8 stay (label
# protocol must match the ecut campaign's eigensolver settings).
# 9242 contract: 96 CPU/node, --mem=360G, 4-node cap (docs/hpc.md).

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 288 --oversubscribe pw.x -ndiag 24"

RUN_DIR="$PROJECT/calculations/w_label_probe2_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES pw=$PW"
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
