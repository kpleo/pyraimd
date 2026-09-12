# Calibration pacing demo (analytic, offline)

A tiny fixed-cell NVE run of H2 on two analytically defined potentials —
reference `U_r = k s / 2` (s = Σᵢ |rᵢ − r0|²) and a surrogate with a smooth
Gaussian error bump `U_s = U_r + A·exp(−s / 2ℓ²)` — demonstrating the
opt-in calibration-pacing controller: after a configurable streak of
calibrations that never produced an accept, recalibration probes are
deferred and the steps drive reference-direct (the existing refusal path);
the wait is retried on a bounded backoff and cancelled early if the
retained correction's measured error exceeds the run's own force budget.

This toy is not a material and its wall clock is Python-dominated — the
point is the accounting and the state machine, not a speedup.

## Run it

From the repository root with the package installed (core only — no QE,
no MACE, no network):

```sh
python examples/calibration_pacing/run_demo.py --output results/off
python examples/calibration_pacing/run_demo.py --output results/on --pacing
python examples/calibration_pacing/run_demo.py --output results/on --pacing --resume --extra-steps 20
```

The third command continues the pacing run in place: frozen pacing
decisions replay verbatim, persisted probes are reused, and the rule
state is restored from the committed records (delete `results/` to start
over; `--force-unlock` reclaims a crashed run's writer lock
deliberately).

## What to look at

Each run prints:

- reference evaluations split by purpose (anchors, probes, refusals,
  checks) — compare the off/on groups;
- the pacing decision log (`calibrate` / `defer` with reasons such as
  `sterile_streak`, `forced_retry`, `safety_exit`);
- every accepted step's true driving-force error against the analytic
  reference (offline verification — never used by the online gate), with
  the over-budget count against the run's force budget;
- the reference-Hamiltonian drift (E_ref + K) over the run as an NVE
  conservation diagnostic.

With this construction the on group defers a handful of sterile
recalibrations early (fewer probes than off) and recovers into sustained
accepts; the accepted-step errors stay under the budget in both groups.
Exact counts depend on the fixed seeds in `run_demo.py` — change the
potentials or seeds and you get a different toy, not a better one.

## Configuration form

The same feature through a config file (adaptive MD runs):

```toml
[policy.calibration_pacing]
enabled = true
failure_streak_limit = 3   # consecutive sterile calibrations before deferring
wait_initial = 1           # skipped opportunities before the first forced retry
wait_max = 8               # backoff cap
```

See `docs/configuration.md` for the field semantics and the
non-composition rules (no online model updates, no explicit direction
callbacks).  Default off; an absent section is exactly the pre-0.6
behavior.

Scope note: while a wait is active the current step's reference driving
force is the ordinary refusal path, but future anchors and routes may
differ from a non-pacing run — pacing can change the trajectory within
the same error governance, it does not promise an identical one.
