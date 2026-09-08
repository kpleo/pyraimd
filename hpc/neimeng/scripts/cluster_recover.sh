#!/usr/bin/env bash
# cluster_recover.sh — patient recovery loop for the maintenance wave
# (2026-09-03: nodes drained under our jobs; launch-failed requeues).
# Every 15 min, on the login node: release held jobs; resubmit vanished
# ones (max one resubmit per line per 30 min); exit when the flagship
# segment, the Li-slab control, and at least one label-array task are all
# RUNNING. Log: $P/logs/cluster_recover.log

P=/data/home/df103967/df103967/cloud_projects/pyraimd2
cd "$P"

declare -A LASTSUB
log() { echo "$(date '+%F %T') $*"; }

state_of() {  # $1 = job-name pattern -> R / PD / NONE
  local st
  st=$(squeue -u df103967 -h -o "%T" -n "$1" 2>/dev/null | head -1)
  if [ -z "$st" ]; then echo NONE; else echo "$st"; fi
}

release_held() {  # release any of our held jobs matching a name
  for jid in $(squeue -u df103967 -h -o "%i %T %R" -n "$1" 2>/dev/null | grep -i "held" | awk '{print $1}'); do
    scontrol release "$jid" 2>/dev/null && log "released $jid ($1)"
  done
}

maybe_submit() {  # $1 = name, $2 = submit command
  local now=$(date +%s)
  local last=${LASTSUB[$1]:-0}
  if [ $((now - last)) -ge 1800 ]; then
    log "submitting $1"
    eval "$2" >> /dev/null 2>&1 && LASTSUB[$1]=$now
  fi
}

for i in $(seq 1 96); do
  st_flag=$(state_of flagship_a_prod)
  st_ctrl=$(state_of li_slab_control)
  n_label_run=$(squeue -u df103967 -h -o "%T" -n w_boot_label432 2>/dev/null | grep -c RUNNING)
  n_label_any=$(squeue -u df103967 -h -n w_boot_label432 2>/dev/null | wc -l)
  s_idle=$(sinfo -p 9242 -h -o "%t" 2>/dev/null | grep -c "^idle" || true)

  log "flagship=$st_flag control=$st_ctrl label(run=$n_label_run any=$n_label_any) 9242_idle_rows=$s_idle"

  [ "$st_flag" = "NONE" ] && maybe_submit flagship_a_prod "EPS_ACC=1.0 STREAK_RHO=0.05 sbatch $P/submit/flagship_a_prod.sbatch"
  [ "$st_ctrl" = "NONE" ] && maybe_submit li_slab_control "sbatch $P/software/pyraimd2/hpc/neimeng/submit/li_slab_control.sbatch"
  [ "$n_label_any" = "0" ] && maybe_submit w_boot_label432 "sbatch $P/software/pyraimd2/hpc/neimeng/submit/w_bootstrap_label_432.sbatch"

  release_held flagship_a_prod
  release_held li_slab_control
  release_held w_boot_label432

  if [ "$st_flag" = "RUNNING" ] && [ "$st_ctrl" = "RUNNING" ] && [ "$n_label_run" -ge 1 ]; then
    log "ALL RUNNING — recovery complete"
    exit 0
  fi
  sleep 900
done
log "24h budget exhausted — still waiting"
