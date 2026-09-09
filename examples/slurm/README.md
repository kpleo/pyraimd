# HPC templates and usage notes

`slurm_generic.sbatch` is the only cluster artifact Pyramid ships: a generic
template with `<PLACEHOLDER>` fields. Site profiles — account names, home
directory layouts, module/conda environment names, credentials — live on the
machines and in local notes, never in this repository.

How to use it:

1. Copy the template into your site's submit directory and fill every
   `<PLACEHOLDER>`: job name, partition, ranks, memory, wall time, log
   directory, environment setup (the commands that put `pw.x` and a python
   with `pyraimd2` on PATH), and the scratch work directory.
2. Environment: the run needs `pw.x` reachable from the same shell that runs
   `pyramid` (the QE engine shells out to it), plus the python environment
   that has `pyraimd2` (and `mace`/`torch` when the recipe uses MACE). On
   conda sites, activating the pyraimd2 env and prepending the QE env's
   `bin/` to PATH is the tested pattern.
3. One allocation per recipe: the whole workflow (singlepoint, relax, the
   MD modes, resume, export) runs serially inside one node job. Do not queue
   a job per MD step; do not oversubscribe ranks against the node shape.
4. Pseudopotentials and foundation-model files are inputs, not code: fetch
   them on the target machine (the recipe READMEs list the exact filenames)
   and reference them with recipe-relative paths.
