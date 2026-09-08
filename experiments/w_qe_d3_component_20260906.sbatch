#!/bin/bash
#SBATCH --job-name=w_fairness_d3_20260906
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:09:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail
umask 077
export LC_ALL=C
TASK_W=/data/home/df103967/df103967/cloud_projects/pyraimd2/experiments/w_fairness_d3_20260906/release_03
QE_W=/data/home/df103967/df103967/cloud_projects/pyraimd2/envs/qe
GFORTRAN_W=/data/app/gcc/gcc-10.2.0/bin/gfortran
LD_W=/data/home/df103967/df103967/cloud_projects/pyraimd2/envs/py312/bin/x86_64-conda-linux-gnu-ld
cd "$TASK_W"
test "$(date +%s)" -lt 1788666946
test "${SLURM_NTASKS}" -eq 28
mkdir "attempt_${SLURM_JOB_ID}"
ATTEMPT_W="$TASK_W/attempt_${SLURM_JOB_ID}"
cd "$ATTEMPT_W"
trap 'result_w=$?; date -u +%FT%TZ > ended_utc.txt; echo "$result_w" > exit_code.txt' EXIT
date -u +%FT%TZ > started_utc.txt
scontrol show job "$SLURM_JOB_ID" > allocation.txt
hostname > hostname.txt
printf '%s\n' "$SLURM_JOB_ID" > jobid.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$QE_W/bin:/data/app/gcc/gcc-10.2.0/bin:$PATH"
export LD_LIBRARY_PATH="$QE_W/lib:/data/app/gcc/gcc-10.2.0/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_FC="$GFORTRAN_W"
test "$(sha256sum "$QE_W/lib/libqe_dftd3.a" | cut -d ' ' -f 1)" = 8b4c223f0cfe528a4588ee3cb95567d812780db7bf9d09ec74f331e1642a0a43
(cd "$TASK_W" && sha256sum -c release_inputs.sha256) > input_verification.txt
"$GFORTRAN_W" --version > compiler_version.txt
"$LD_W" --version > linker_version.txt
test "$(sha256sum "$LD_W" | cut -d ' ' -f 1)" = 4bf70c257f4d889dd384c0018edebab8e77d85b4f01880a44449989ab75bcb76
mkdir toolchain
ln -s "$LD_W" toolchain/ld
"$GFORTRAN_W" "-B$ATTEMPT_W/toolchain/" -print-prog-name=ld > compiler_selected_linker.txt
test "$(cat compiler_selected_linker.txt)" = "$ATTEMPT_W/toolchain/ld"
"$QE_W/bin/mpifort" -show > mpi_compile_command.txt
include_w=("-I$QE_W/include/qe/qe_dftd3" "-I$QE_W/include/qe/qe_modules" "-I$QE_W/include/qe/qe_utilx" "-I$QE_W/include/qe/qe_lax")
libraries_w=()
for name_w in dftd3 modules utilx utilx_c modules_c lax device_lapack upflib xml; do
    libraries_w+=("$QE_W/lib/libqe_${name_w}.a")
done
sha256sum "${libraries_w[@]}" "$GFORTRAN_W" "$LD_W" "$QE_W/bin/mpifort" "$QE_W/bin/mpirun" > runtime_files.sha256
date -u +%FT%TZ > compile_started_utc.txt
set -x
"$QE_W/bin/mpifort" "-B$ATTEMPT_W/toolchain/" -O2 -fopenmp "${include_w[@]}" \
    "$TASK_W/w_qe_d3_component_20260906.f90" -o w_qe_d3_component.x \
    -Wl,--no-relax -Wl,--start-group "${libraries_w[@]}" -Wl,--end-group \
    "-L$QE_W/lib" "-Wl,-rpath,$QE_W/lib" -llapack -lblas -lfftw3 -lgomp -lm -ldl -lz \
    > compile.stdout 2> compile.stderr
set +x
date -u +%FT%TZ > compile_ended_utc.txt
sha256sum w_qe_d3_component.x > executable.sha256
ldd w_qe_d3_component.x > linked_libraries.txt
objdump -d w_qe_d3_component.x > disassembly.txt
if grep -E 'callq?.*<__dftd3_core_MOD_(rcov|r2r4)>' disassembly.txt > invalid_data_calls.txt; then
    echo 'Linker generated a call to D3 data; refusing numerical execution' >&2
    exit 78
fi
test "$(date +%s)" -lt 1788666946
date -u +%FT%TZ > compute_started_utc.txt
"$QE_W/bin/mpirun" -np 28 ./w_qe_d3_component.x < "$TASK_W/pair_qe_d3.in" > compute.stdout 2> compute.stderr
date -u +%FT%TZ > compute_ended_utc.txt
test "$(tail -1 w_d3_raw.dat)" = COMPLETE
sha256sum w_d3_raw.dat > result.sha256
(cd "$TASK_W" && sha256sum -c release_inputs.sha256) > final_input_verification.txt
sha256sum -c runtime_files.sha256 > final_runtime_verification.txt
