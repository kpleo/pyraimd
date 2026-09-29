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

# The generated config follows the ACTUAL template's directory — with
# RUN_CONFIG_TEMPLATE pointing at another project, the job config lands
# next to THAT template, so its relative paths (structure.file,
# pseudo_dir, run.directory) resolve exactly like the template's.  Never
# the spool dir, never assumed next to the launcher.
TEMPLATE_DIR="$(cd "$(dirname "$CONFIG_TEMPLATE")" && pwd)"
CONFIG_TEMPLATE="$TEMPLATE_DIR/$(basename "$CONFIG_TEMPLATE")"

# A unique job-private parent for the state; the run config name is
# unique PER EXECUTION (job id AND a random component — never a
# predictable name another file could collide with).
JOB_WORK="$(mktemp -d "${SLURM_TMPDIR:-/tmp}/prepared-qe-${SLURM_JOB_ID}.XXXXXX")"
STATE="$JOB_WORK/state"
RUN_CONFIG="$TEMPLATE_DIR/run.job-${SLURM_JOB_ID}-${RANDOM}${RANDOM}.toml"

# Cleanup ownership starts EMPTY and is registered ONLY after this run's
# render succeeded and the new file's creation is confirmed — a failed
# setup/render deletes nothing, and a pre-existing user file is never
# touched.  The original exit code is preserved (compute failures are
# never masked), and only registered self-created paths are removed:
# run results, failure records, pseudos and user configs stay.
OWNED_RUN_CONFIG=""
cleanup() {
    rc=$?
    rm -rf "$JOB_WORK"
    if [ -n "$OWNED_RUN_CONFIG" ] && [ -e "$OWNED_RUN_CONFIG" ]; then
        rm -f -- "$OWNED_RUN_CONFIG"
    fi
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 1. prepare ONCE per allocation (exclusive state creation)
python3 "$LAUNCHER" prepare \
    --setup "$EXAMPLE_DIR/setup.sh" --state "$STATE" \
    --mpi mpirun --pw pw.x --ranks "$SLURM_NTASKS"

# 2. render the job-specific config with THIS job's state argv/hash —
#    the render's atomic no-clobber publish refuses to touch any
#    existing file; the user's template is never modified
python3 "$LAUNCHER" render-config \
    --template "$CONFIG_TEMPLATE" --state "$STATE" --output "$RUN_CONFIG"
# the render succeeded: THIS run created that file, so this run owns it
if [ -f "$RUN_CONFIG" ]; then
    OWNED_RUN_CONFIG="$RUN_CONFIG"
fi

# 3. validate with the project's own loader, then run (resume keeps
#    working inside THIS allocation; a resubmission is a new allocation
#    and must re-prepare — the changed command identity is then subject
#    to Pyramid's existing checkpoint/cache compatibility checks)
pyramid validate "$RUN_CONFIG"
pyramid run "$RUN_CONFIG"
