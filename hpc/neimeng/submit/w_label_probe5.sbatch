#!/usr/bin/env bash
#SBATCH --job-name=w_label_probe5
#SBATCH --partition=9242
#SBATCH --nodes=4
#SBATCH --ntasks=288
#SBATCH --cpus-per-task=1
#SBATCH --mem=360G
#SBATCH --time=01:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# W label probe round 5 — FOR THE RECORD: the 686-atom label at the fastest
# known layout (288 pure-MPI ranks) spread over the full 4-node 9242
# allocation (72 ranks/node; QE estimate 1469 GB total -> 367 GB/node,
# actual usage historically ~15-50% of the estimate). Round 4 (96r x 4t
# hybrid) fit but ran 30 min/iteration = nonviable. This round settles
# whether 686 is usable at all; the campaign decision (432 fallback) is
# already taken on Act-2 label economics regardless — this measurement is
# the on-record justification. Success = survives past iteration ~6 with a
# measured iteration rate.
# 9242 contract: 96 CPU/node, --mem=360G, 4-node cap (docs/hpc.md).

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 288 pw.x -ndiag 24"

RUN_DIR="$PROJECT/calculations/w_label_probe5_${SLURM_JOB_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES ntasks=$SLURM_NTASKS pw=$PW"
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
