# Pyramid 0.4.0 independent review evidence

Baseline: `9c7dc4baba3c88544b85e254547b17b84a49196c` (`v0.4.0`).

See [overall review](../../docs/development_reports/INDEPENDENT_REVIEW_040_20260909.md) for severity, scope and correction order.

- `core_suite.log`: one independent run of the existing core suite; 458 passed, 1 deselected, with ASE/NumPy deprecation warnings. No real DFT or MACE run.
- `previous_probe_recheck.json`: re-execution of `../development_review_20260909/reproduce_findings.py` against 0.4.0. It confirms the configuration, normal complete-step momentum and incomplete-step count corrections. The direct Python API retry ledger remains undercounted in that probe; the backend review distinguishes this from fixed CLI paths.
- `export_probe.py` and `export_probe.json`: initial momentum, singlepoint control, failed relaxation export, orphan/replacement duplicate frames, and incomplete-tail inspection. Uses the existing `tests/unit/test_review_r1.py` analytic fixtures and creates new temporary directories only. Deliberate faults affect only those temporary runs.
- `material_recheck.json`: independent arithmetic from committed Al force rows and current/old events. FD residuals are recomputed from the archived derivative/force values, not from a new electronic-structure calculation. Actual MACE restore evidence is in the development reports' `contract_evidence` package and was not rerun here.
- `core_review.md`, `core_probe.py`, `core_probe.log`: closed original triggers and four residual core defects. The main reviewer reran all probe cases independently after receiving the component report. No source files were changed.
- `backend_review.md` and `pyramid_040_{backend,workflow,input}_tiny.{py,json,log}` (where present): the backend reviewer's targeted runs and outputs. The three scripts share a fake-executable helper and must remain together. The main reviewer inspected the resulting evidence and relevant implementation, without repeating the entire backend set.

From the repository root, using the existing environment:

```sh
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_040_20260909/export_probe.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_20260909/reproduce_findings.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_040_20260909/core_probe.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_040_20260909/pyramid_040_backend_tiny.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_040_20260909/pyramid_040_workflow_tiny.py
UV_OFFLINE=1 uv run --no-sync python analysis/development_review_040_20260909/pyramid_040_input_tiny.py
```

These are diagnostic scripts, not passing acceptance tests. A successful script exit means the observations were produced; the observed values include defects. The original implementation test helpers are reused deliberately so the check targets the same supported scenarios as the developer's regressions.
