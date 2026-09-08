#!/usr/bin/env bash
#SBATCH --job-name=w_label_probe
#SBATCH --partition=9242
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=360G
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A_%a.err

# W label RAM probe (7620780 OOM postmortem): the 686-atom W label
# (nbnd 4852, 70/700 Ry) needs 782 GB total at the 28-rank normal-node
# layout -> SIGKILL. Move to 9242 (96 CPU/375G) and find the layout that
# fits, measuring MaxRSS per task:
#   task 0: 96 ranks, no band-group tuning (G-vector distribution only)
#   task 1: 96 ranks, pw.x -ndiag 24 (dense-diag workspace distributed
#           over 24-rank linear-algebra groups)
# Frame 0 of the bootstrap set (bulk 300 K — the cheapest case; hot liquid
# frames only get harder). Success = SCF converges + MaxRSS < ~340 GB.
# 9242 contract: 96 CPU/node, --mem=360G, 4-node cap (docs/hpc.md).

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"

case "$SLURM_ARRAY_TASK_ID" in
  0) PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 96 --oversubscribe pw.x" ;;
  1) PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 96 --oversubscribe pw.x -ndiag 24" ;;
esac

RUN_DIR="$PROJECT/calculations/w_label_probe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
install -d -m 700 "$RUN_DIR"

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID pw=$PW"
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
