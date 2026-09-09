#!/usr/bin/env bash
# Generic Slurm template for Pyramid recipe runs.
# Copy this file, fill every <PLACEHOLDER>, and keep site-specific values
# (accounts, home paths, credentials) OUT of version control.
#
# Fill these before submitting:
#   <JOB_NAME>        short job name, e.g. pyramid_si
#   <PARTITION>       site partition/queue name
#   <NTASKS>          MPI ranks for pw.x (match the node shape)
#   <MEM>             memory directive valid for the site (or delete the line)
#   <TIME>            wall limit, e.g. 04:00:00
#   <LOG_DIR>         absolute directory for slurm_%x_%j logs
#   <ENV_SETUP>       commands that put pw.x and the pyraimd2 python on PATH
#   <WORK_DIR>        absolute scratch/run directory for this job
#SBATCH --job-name=<JOB_NAME>
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=<NTASKS>
#SBATCH --cpus-per-task=1
#SBATCH --mem=<MEM>
#SBATCH --time=<TIME>
#SBATCH --output=<LOG_DIR>/slurm_%x_%j.out
#SBATCH --error=<LOG_DIR>/slurm_%x_%j.err

# One node, one serial workflow: the MD advances inside this allocation;
# do NOT queue a separate job per MD step. Threads stay at 1 per rank so
# the node is not oversubscribed.
set -euo pipefail
<ENV_SETUP>
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

install -d -m 700 "<WORK_DIR>"
cd "<WORK_DIR>"
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID ntasks=$SLURM_NTASKS"

# Recipe steps go here, e.g.:
#   python make_structure.py
#   pyramid run run-singlepoint.toml
#   pyramid run run-md-reference.toml
#   pyramid run run-md-adaptive.toml
#   pyramid resume runs/<run-id> --steps 2
#   pyramid export runs/<run-id> --force-source driving
