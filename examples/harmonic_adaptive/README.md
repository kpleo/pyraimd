# Harmonic adaptive MD — runnable offline example

A complete, dependency-free adaptive NVE run: the builtin analytic
`harmonic-reference` engine and `harmonic-surrogate` model drive an H2O toy
molecule, exercising anchoring, surrogate acceptance, independent checks,
checkpoints, stop and resume. No external programs, no model downloads.

Run it (from anywhere — paths resolve against `run.toml`'s directory):

```sh
pyramid validate run.toml
pyramid run run.toml
pyramid inspect runs/harmonic-demo
pyramid export runs/harmonic-demo --force-source reference --output reference.extxyz
pyramid resume runs/harmonic-demo --steps 10
```

`pyramid init --template harmonic --output <dir>` writes exactly this pair
of files. The policy/verification numbers are demonstration values, not
accuracy recommendations for any material.
