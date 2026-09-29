# prepared_qe_launcher — prepare the QE environment once per Slurm job

An OPTIONAL, stdlib-only helper for allocations that make MANY QE calls
(plain reference MD, MTS reference boundaries) AND whose launch wrapper
re-runs a costly environment setup on every call. The default
`pw_cmd = ["pw.x"]` plain path does NOT re-source anything per call —
this example pays off only where your own wrapper scripts re-activate
modules/conda for each invocation; there it prepares once per allocation
and reuses the result. It changes no default behavior: core backends,
configuration schema, MD/SCF/cache, scratch and resume semantics are
untouched. Verify the prepared run's energies/forces against the plain
launch within tolerances YOU declare in advance (see the validation step
below; numerical behavior under MPI is compared to your own acceptance
tolerance, never assumed bit-for-bit).

**Acquisition.** The example ships with the source repository (and the
release tag's source tree on GitHub); it is NOT part of the installed
wheel or sdist — copy the directory from a checkout.

**Scope.** Explicit `mpirun`-style launches only (`mpirun -np N pw.x`),
one allocation at a time: the prepared state binds the job's
`SLURM_JOB_ID` and a content hash. Resume INSIDE the same allocation
works as usual; a resubmission is a NEW allocation and must re-prepare —
the prepared argv/hash changes the QE command identity, which then goes
through Pyramid's existing checkpoint/cache compatibility checks like
any other configuration change. There is no cross-job identity exemption
and no seamless cross-job resume promise.

## Files

- `launcher.py` — the subcommands (`prepare`, `run`, `render-config`);
  standard library only, no Pyramid import.
- `setup.sh` — your trusted environment setup template (edit it: module
  load, conda activate, thread pinning).
- `run.toml.template` — your run configuration with the ONE placeholder
  `@PREPARED_PW_CMD@` as `reference.pw_cmd` (edit everything else).
- `job_template.sbatch` — a Slurm invocation template wiring the steps.

## The three user steps

Submit from the example directory (Slurm then sets `SLURM_SUBMIT_DIR` to
it), or point `PYRAMID_EXAMPLE_DIR` at it explicitly — the job script is
copied to Slurm's spool, so it never locates resources relative to
itself; absolute paths and paths with spaces work throughout:

```sh
cd /path/to/prepared_qe_launcher && sbatch job_template.sbatch
# or: PYRAMID_EXAMPLE_DIR=/path/to/prepared_qe_launcher sbatch job_template.sbatch
```

1. **Install/prepare** (inside the allocated job): edit `setup.sh` to
   your site; the job script runs

   ```sh
   python3 launcher.py prepare --setup setup.sh --state "$STATE" \
       --mpi mpirun --pw pw.x --ranks "$SLURM_NTASKS"
   ```

   The setup script is sourced in a separate bash subprocess — your
   interactive environment is never modified. The COMPLETE managed set
   of runtime variables is recorded as the setup resolved it (PATH,
   LD_LIBRARY_PATH, LIBRARY_PATH, XML_CATALOG_FILES, OMP_/OPENBLAS_/MKL_*
   and CONDA_*/MINIFORGE* by default; `--allow-var NAME` extends it —
   credential-class names are refused even there). A REAL change OUTSIDE
   that set (added, modified or removed) aborts the prepare by variable
   name, never silently dropped; added or changed credential-class
   variables abort by name only. The state directory is created
   exclusively (any pre-existing one refused), mode 0700 with 0600
   files, bound to the current `SLURM_JOB_ID` and a content hash that
   covers the managed set and the allow policy. `prepare` also writes
   `pw_cmd.json` (the machine-readable argv with the prepare-time Python
   interpreter, the launcher's absolute path, the state path and the
   expected hash).

2. **Configure**: the job script renders your `run.toml.template` into a
   job-specific `run.job-$SLURM_JOB_ID.toml` NEXT TO the template — the
   placeholder `@PREPARED_PW_CMD@` becomes this job's prepared pw_cmd
   argv with every element properly TOML-quoted, so the rendered config's
   relative paths (structure, pseudos, `run.directory`) resolve exactly
   like the template's. The template itself is never modified, an
   existing output is refused, and the rendered config is validated with
   the project's own loader (`pyramid validate`) before any compute.

3. **Run and finish**: the job script runs `pyramid run` on the rendered
   config — no manual hash pasting after the job starts. On success or
   failure (EXIT/INT/TERM), cleanup removes ONLY what the job created
   (the job-private state parent and the rendered config); run results,
   pseudos, caches and pre-existing user files stay untouched. You only
   supply your own structure, QE parameters and setup before submission.

## What `run` rebuilds (the environment boundary)

At `run` time the managed set is rebuilt EXACTLY: prepared values win,
and managed variables NOT present in the prepared set are unset even
when set in the calling shell. Everything OUTSIDE the managed set keeps
inheriting from the current process (including scheduler job/step
identities, which are never recorded). The state does NOT freeze the
whole environment — it freezes the managed runtime set, bound by the
content hash; `run` refuses a mismatched job id or hash BEFORE spawning
anything, never printing environment values.

## Validation step (do this once per setup)

Run one singlepoint with your normal `pw_cmd` and one through the
prepared launcher on the same structure, and compare energies/forces
against numerical tolerances YOU declare in advance for your machine and
recipe (record the tolerance with the comparison; do not assume
bit-for-bit equality under different launch environments):

```sh
pyramid run run_singlepoint_plain.toml
pyramid run run_singlepoint_prepared.toml
# compare the committed energies/forces against your predeclared tolerance
```

## Honest notes

- This is an optimization for allocations that issue many QE calls under
  a per-call re-setup wrapper; with the plain default `pw_cmd` there is
  no such per-call cost to remove.
- The prepared environment is exactly what your `setup.sh` produces —
  the launcher does not check QE correctness, convergence or physics.
- `run` replaces itself with the MPI/pw.x process (`os.execvpe`): child
  exit codes and Ctrl-C propagate naturally, with no supervisor process
  and no shell in between.
