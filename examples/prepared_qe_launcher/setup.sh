#!/bin/bash
# Prepared-launcher setup template — YOUR trusted environment preparation,
# sourced ONCE per Slurm allocation by `launcher.py prepare` (in a separate
# bash subprocess; the parent process environment is never modified).
#
# Replace the examples below with your site's real setup.  Everything you
# export here is filtered through the launcher's runtime allowlist (PATH,
# LD_LIBRARY_PATH, LIBRARY_PATH, XML_CATALOG_FILES, OMP_/OPENBLAS_/MKL_*
# thread variables and CONDA_*/MINIFORGE* by default; extend with
# `launcher.py prepare --allow-var NAME`).  Credential-class variables
# (KEY/TOKEN/SECRET/PASS/CRED/AUTH/CERT in the name) abort the prepare.

# --- examples (EDIT ME) ---
# module load qe/7.3
# source /opt/conda/etc/profile.d/conda.sh && conda activate qe-env
# export PATH="/opt/your-qe/bin:$PATH"
# export OMP_NUM_THREADS=1
