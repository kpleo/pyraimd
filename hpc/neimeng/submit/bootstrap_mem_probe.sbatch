#!/usr/bin/env bash
#SBATCH --job-name=boot_mem_probe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Stage-3 sizing probe after the 7616839 OOM (e3nn TP einsum workspace is
# per-torch-thread: 28 x ~1.03 GB ~= 29 GB killed the 120 G node). Measures
# per-forward time, fine-tune epoch time, and VmHWM at several thread
# counts on the real 474-atom box — sizes the resubmitted stage-3 job.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
# Threads are set in-process via torch.set_num_threads; keep BLAS pools
# from independently spawning 28-wide workspaces.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID"

cd "$PROJECT"
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/probe_bootstrap_mem.py" \
  --labels "$PROJECT/sync/bootstrap_labels_7616227_*.jsonl" \
  --frames "$PROJECT/calculations/bootstrap_frames/all.extxyz" \
  --model "$PROJECT/software/mace-mp-0b3-medium.model" \
          "$PROJECT/software/mace-mpa-0-medium.model" \
  --thread-counts 4 8 16 \
  --n-train-probe 4

echo "done"
