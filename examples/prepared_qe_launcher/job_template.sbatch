#!/bin/bash
#SBATCH --job-name=prepared-qe-demo   # adjust to your site
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
# Prepared-launcher invocation template: prepare -> configure -> run in
# ONE allocation, with no manual hash pasting after the job starts.
#
# RESOURCE LOCATION: this script is copied to Slurm's spool before it
# runs, so it NEVER locates resources relative to itself.  Submit from
# the example's directory (then SLURM_SUBMIT_DIR points at it), or set
# PYRAMID_EXAMPLE_DIR explicitly:
#   cd /path/to/prepared_qe_launcher && sbatch job_template.sbatch
#   # or: PYRAMID_EXAMPLE_DIR=/path/to/prepared_qe_launcher sbatch job_template.sbatch
set -euo pipefail

EXAMPLE_DIR="${PYRAMID_EXAMPLE_DIR:-${SLURM_SUBMIT_DIR:?submit from the example directory or set PYRAMID_EXAMPLE_DIR}}"
LAUNCHER="$EXAMPLE_DIR/launcher.py"
CONFIG_TEMPLATE="${RUN_CONFIG_TEMPLATE:-$EXAMPLE_DIR/run.toml.template}"

# A unique job-private parent for the state; the generated job config
# lives NEXT TO the user's template so its relative paths (structure,
# pseudos, run.directory) resolve exactly like the template's.
JOB_WORK="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/prepared-qe-${SLURM_JOB_ID}.XXXXXX")"
STATE="$JOB_WORK/state"
RUN_CONFIG="$EXAMPLE_DIR/run.job-${SLURM_JOB_ID}.toml"

cleanup() {
    # ONLY what this job created: the job-private state parent and the
    # generated config.  Run results, pseudos, caches and every
    # pre-existing user file/directory stay untouched.
    rm -rf "$JOB_WORK"
    if [ -f "$RUN_CONFIG" ]; then rm -f "$RUN_CONFIG"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 1. prepare ONCE per allocation (exclusive state creation)
python3 "$LAUNCHER" prepare \
    --setup "$EXAMPLE_DIR/setup.sh" --state "$STATE" \
    --mpi mpirun --pw pw.x --ranks "$SLURM_NTASKS"

# 2. render the job-specific config with THIS job's state argv/hash
#    (the user's template keeps its single @PREPARED_PW_CMD@ placeholder
#    and is never modified)
python3 "$LAUNCHER" render-config \
    --template "$CONFIG_TEMPLATE" --state "$STATE" --output "$RUN_CONFIG"

# 3. validate with the project's own loader, then run (resume keeps
#    working inside THIS allocation; a resubmission is a new allocation
#    and must re-prepare — the changed command identity is then subject
#    to Pyramid's existing checkpoint/cache compatibility checks)
pyramid validate "$RUN_CONFIG"
pyramid run "$RUN_CONFIG"
