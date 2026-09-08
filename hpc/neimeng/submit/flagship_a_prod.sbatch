#!/usr/bin/env bash
#SBATCH --job-name=flagship_a_prod
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=14:00:00
#SBATCH --output=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.out
#SBATCH --error=/data/home/df103967/df103967/cloud_projects/pyraimd2/logs/slurm_%x_%A.err

# Flagship A production closed loop, warm-started from the bootstrap
# committee (stage-3: ensemble held-out fMAE 0.155/fMax 0.98). One 14 h
# wall segment per submission; RESUBMIT THE SAME SCRIPT to chain — the
# loop.db presence test switches segment 2+ to --resume automatically.
# Operating point: EPS_ACC env (default 0.25). Segments 1-3 ran 0.25
# (0% acceptance — persistently hard system, e waves 0.5-1.05); segment 4+
# runs 1.0 — the refusal-curve-selected point (B(s) ~ 0.84 at s~0.28,
# qhat~3.0), guarantee semantics: <=1.0 eV/A max-atom force error at 95%
# marginal coverage. eps enters only the accept rule, so a between-
# segment switch is clean: window/qhat/store carry over untouched.
# Streak inflation: STREAK_RHO env (default 0.0). Audit 7617790 found the
# plain bound breaks on acceptance streaks (4/6 violations; qhat freezes
# while r drifts ~0.05/step). Segments 4-5 run 0.0 (the "before" curve);
# segment 6+ runs 0.05 (measured drift rate) — the "after" curve.
# Exploration labels: EXPLORE_FRAC env (default 0.0 = OFF; referee §2.4
# pre-registered forward-looking protocol). On a fraction p of ACCEPTED
# steps the loop computes the DFT label anyway (shadow): it enters the
# conformal window/updater as a pure drift audit, the step still
# propagates with surrogate forces (certified trajectory bit-identical
# to p=0), the row keeps route=ml with "explore-label" in its reason,
# and the decision-driven streak is untouched. Enable only when
# acceptance resumes; draws are a dedicated default_rng(--seed) stream.
# Committee: K=2 mixed [0b3, MPA-0] (order must match the merged
# checkpoint), readout+products, 30 epochs, train-window 16.
# SCF: frozen metallic-interface protocol (fd 0.02, beta 0.20, nbnd
# nelec/2+150=942, 60/600 Ry, conv_thr 1e-6) + startpot density chaining.
# Neimeng A "normal" contract: 28 CPU/node, --mem=120G, no GPU.

set -euo pipefail
PROJECT=/data/home/df103967/df103967/cloud_projects/pyraimd2
WORKDIR="$PROJECT/calculations/flagship_a_prod"
RUN_ID=flagship-a-prod
EPS_ACC="${EPS_ACC:-0.25}"
STREAK_RHO="${STREAK_RHO:-0.0}"
EXPLORE_FRAC="${EXPLORE_FRAC:-0.0}"

source /data/hzwtech/profile/Miniforge3.env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/py312"
export PATH="$PROJECT/envs/qe/bin:$PATH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

PW="env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 mpirun -np 28 --oversubscribe pw.x"
COMMON_ARGS=(
  --atomsXYZ "$PROJECT/inputs/generated/interface_li100_lp30_474.xyz"
  --model "$PROJECT/software/mace-mp-0b3-medium.model"
          "$PROJECT/software/mace-mpa-0-medium.model"
  --pseudo-dir "$PROJECT/inputs"
  --pw-cmd "$PW"
  --run-id "$RUN_ID"
  --workdir "$WORKDIR"
  --n-steps 200 --temperature 300 --timestep-fs 0.5
  --eps-acc "$EPS_ACC" --alpha 0.05 --window 64 --w-min 16
  --streak-rho "$STREAK_RHO"
  --explore-frac "$EXPLORE_FRAC"
  --n-label 8 --epochs 30 --train-window 16
  --trainable-filters readout products
  --torch-threads 16
  --metallic --smearing fd --degauss 0.02 --mixing-beta 0.20
  --mixing-ndim 12 --diago-david-ndim 8 --nbnd-headroom 150
  --electron-maxstep 400 --ecutwfc 60 --ecutrho 600 --conv-thr 1e-6
  --startpot-chain
)

install -d -m 700 "$WORKDIR"
if [ -f "$WORKDIR/loop.db" ]; then
  MODE="--resume"
else
  MODE="--init-committee $PROJECT/sync/committee_bootstrap.pt"
fi
echo "host=$(hostname) date=$(date '+%F %T %Z') job=$SLURM_JOB_ID mode=$MODE eps_acc=$EPS_ACC streak_rho=$STREAK_RHO explore_frac=$EXPLORE_FRAC"

cd "$PROJECT"
# shellcheck disable=SC2086
"$PROJECT/envs/py312/bin/python" "$PROJECT/software/pyraimd2/experiments/adaptive_loop.py" \
  "${COMMON_ARGS[@]}" $MODE

echo "done"
