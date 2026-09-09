# Independent review evidence — 2026-09-09

Reviewed source: `5a1d1054d7cdd9444bdd85a62afeba6a5fc607d5` on `dev/0.4.0` (0.4.0rc1).

The overall assessment and correction order are in [INDEPENDENT_REVIEW_20260909.md](../../docs/development_reports/INDEPENDENT_REVIEW_20260909.md). Detailed findings are in [core_review.md](core_review.md) and [backend_review.md](backend_review.md).

These scripts reproduce the reviewed version's **incorrect behavior**. Exit code zero means a reproduction completed, not that the software passed the proposed acceptance criteria. Some scripts intentionally assert the presence of a defect and should fail or be rewritten once that defect is corrected. They are review probes, not a new production test suite.

## Environment and cost

- Python 3.12.13, ASE 3.29.0, NumPy 2.5.2; existing project environment.
- No model downloads, dependency installation, cloud jobs, or real electronic-structure calculation.
- Tiny analytic systems and fake executables reading `tests/data/qe_si_scf.out` only.
- Scripts create fresh temporary run directories. A few inject failures or change files **inside their own temporary directories** to expose recovery defects.
- The main core suite was independently run once: 389 passed, 1 deselected, 102562 ASE/NumPy deprecation warnings in 15.79 s. That test result does not cover the additional defects below.

## Reproductions

Run from the repository root with its existing environment:

```sh
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/reproduce_findings.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_core_review.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_core_more.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_core_guard.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_core_process.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_review_backends_5a1d105.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_review_workflows_5a1d105.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/pyramid_review_qe_retry_5a1d105.py
```

Evidence mapping:

- `reproductions.json`: force_metric forwarding, half-step momentum export, incomplete-step counting/export, and hidden QE retry cost; produced by `reproduce_findings.py`.
- `pyramid_core_review.log`: pending RNG, consumed-cache replay, database/event commit window, publication/validation rollback, and periodic training structure.
- `pyramid_core_more.log`: model artifact corruption, cached-label retraining after restart, torn event tail, and supplementary recalibration-tail observations.
- `pyramid_core_guard.log`: an internally inconsistent dimer force passes the rigid-translation guard.
- `pyramid_core_process.log`: positive continuous-versus-independent-process complete-boundary comparison, and negative pending-proposal RNG comparison.
- `pyramid_review_backends_5a1d105.log`: ASE identities, QE status/input mapping, directory collision, warm start, and malformed outputs.
- `pyramid_review_workflows_5a1d105.log`: constrained relaxation reporting, ASE convergence metric, stationary ordinary MD resume, and changed-surrogate resume.
- `pyramid_review_qe_retry_5a1d105.log`: nonconvergence classification, physical attempt versus logical task cost, and relative pseudopotential directory resolution.
- `fixed_dof_temperature.json`: a separate two-atom FixAtoms observation. Construct H2, fix atom 0, set momenta to `[[0,0,0],[0.1,0,0]]`, and compare `runtime.inspect._temperature_K(atoms)` with `atoms.get_temperature()`. The former divides by 3N and gives half the constraint-aware result.

The seven archived component scripts were re-executed after making their fixture/import paths portable; their corresponding logs are from that execution. The detailed reports retain some original temporary-directory names for provenance. No large trajectory or electronic-structure files are included. The retained workflow stderr records the failed final-output refresh warnings caused by empty stored labels in the stationary-resume cases; it is part of the observation, not a hidden test pass.

Static-only observations, including tensor serialization in the real Committee/MACE path, are labelled separately in the reports. No actual MACE fine-tuning or QE material calculation is claimed by these probes.
