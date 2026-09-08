# QE + MACE configuration skeleton

A configuration *shape* for a real-material adaptive run: Quantum
ESPRESSO as the reference engine and a MACE foundation model as the
surrogate, over a small periodic silicon cell. It exists to show how
backend options map to TOML and to be checked with `pyramid validate`
(parameter and path validation). It is **not** a verified materials recipe
— converged reference settings, pseudopotential choices and accuracy
targets are WP08's deliverable.

Prerequisites before `pyramid run` would work:

- `pw.x` on PATH (or set `reference.pw_cmd`), and the referenced
  pseudopotential files installed under the directory `pseudo_dir` points
  at (paths resolve relative to this file).
- `mace-torch` installed (`pip install 'pyraimd2[mace]'`) and a local MACE
  model file if `surrogate.model` names a path; a bare name like `small`
  refers to the MACE-MP foundation models and is downloaded by MACE itself
  on first use (validation never downloads anything).

What works offline today:

```sh
pyramid validate run.toml        # schema, structure, paths, backend contract
pyramid validate run.toml --probe-backends   # only on a machine with pw.x + torch
```

Notes:

- `reference` options are passed to the QE factory verbatim (`QeConfig`
  fields); `run_root` is supplied by the workflow (`<run dir>/calculations`).
- The recipe is explicit: PBE + Grimme D3 (`xc`, `dispersion`) — change it
  deliberately, not by editing defaults elsewhere.
- `pseudos` maps species to filenames resolved under `pseudo_dir`.
