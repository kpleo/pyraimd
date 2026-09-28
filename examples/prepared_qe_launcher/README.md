# prepared_qe_launcher — prepare the QE environment once per Slurm job

An OPTIONAL, stdlib-only helper for allocations that make MANY QE calls
(plain reference MD, MTS reference boundaries): by default every
`pw_cmd` invocation re-sources the environment per call; this example
prepares it once per allocation and reuses it per call. It changes no
default behavior: core backends, configuration schema, MD/SCF/cache,
scratch and resume semantics are untouched. Use it only when re-sourcing
is measurably slow for you, and verify the results are numerically
identical (see the validation step below).

**Acquisition.** The example ships with the source repository (and the
release tag's source tree on GitHub); it is NOT part of the installed
wheel or sdist — copy the directory from a checkout.

**Scope.** Explicit `mpirun`-style launches only (`mpirun -np N pw.x`),
one allocation at a time: the prepared state is bound to the job's
`SLURM_JOB_ID` and to a content hash, so a resubmission (a new
allocation) must re-prepare. Pyramid's own engine cache, identity and
resume compatibility checks all still apply — the launcher only changes
HOW the process starts, never what is computed.

## Files

- `launcher.py` — the two subcommands (`prepare`, `run`); standard
  library only, no Pyramid import.
- `setup.sh` — your trusted environment setup template (edit it: module
  load, conda activate, thread pinning).
- `job_template.sbatch` — a Slurm invocation template wiring the three
  steps below.

## The three user steps

1. **Install/prepare** (inside the allocated job): edit `setup.sh` to
   your site, then

   ```sh
   python3 launcher.py prepare --setup setup.sh --state "$STATE" \
       --mpi mpirun --pw pw.x --ranks "$SLURM_NTASKS"
   ```

   The setup script is sourced in a separate bash subprocess — your
   interactive environment is never modified.  The recorded delta is
   allowlisted (PATH, LD_LIBRARY_PATH, LIBRARY_PATH, XML_CATALOG_FILES,
   OMP_/OPENBLAS_/MKL_* and CONDA_*/MINIFORGE* by default;
   `--allow-var NAME` extends it); variables the setup REMOVES are
   unset at run time; credential-class variables (KEY/TOKEN/SECRET/...)
   abort the prepare and are never stored or printed.  The state
   directory is mode 0700, never overwritten, and bound to the current
   `SLURM_JOB_ID`; prepare prints the state's content hash.

2. **Configure `reference.pw_cmd`** in your run configuration with the
   printed values (QE's `-in` argument is appended after `--` by the
   engine):

   ```toml
   [reference]
   backend = "qe"
   pw_cmd = ["/abs/path/launcher.py", "run", "--state",
             "/abs/path/STATE", "--expected-sha256", "<HASH>", "--"]
   ```

   (`launcher.py` as argv[0] needs its exec bit; or prefix with your
   Python interpreter as the first element.)  Before spawning anything,
   `run` refuses with a clear error if the job id or the state hash does
   not match — a stale or foreign state is never executed.

3. **Run and finish**: `pyramid run` as usual; resume works unchanged
   (Pyramid's identity checks still apply).  At job end, delete ONLY the
   example's job-private state directory (`rm -rf "$STATE"` in the
   template).

## Validation step (do this once per setup)

Prepare-then-run must be numerically identical to the plain launcher.
Run one singlepoint with your normal `pw_cmd` and one with the prepared
`pw_cmd` on the same structure, and compare the energies/forces (they
must match bit-for-bit on the same machine):

```sh
pyramid run run_singlepoint_plain.toml
pyramid run run_singlepoint_prepared.toml
# compare the committed energies/forces, e.g. via `pyramid export`
```

## Honest notes

- This is an optimization for allocations that issue many QE calls; if
  your jobs make a handful of calls, keep the default per-call launch.
- The prepared environment is exactly what your `setup.sh` produces —
  the launcher does not check QE correctness, convergence or physics.
- `run` replaces itself with the MPI/pw.x process (`os.execvpe`): child
  exit codes and Ctrl-C propagate naturally, with no supervisor process
  and no shell in between.
