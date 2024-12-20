#!/bin/bash

module load GCC
module load OpenMPI

export OMP_NUM_THREADS=10      
NB_PROC=4                      

BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case/"
export SHALLOW_INPUT_DIR="$INPUT_PATH"
OUTPUT_PATH="../../output"
mkdir -p "$BIN_PATH"
mkdir -p "$OUTPUT_PATH"

# Ensure all processerus are killed
cleanup() {
    pkill -9 mpirun || true
    pkill -9 orted || true  # Kill Open MPI daemon processes
    sleep 2
}

cleanup
mpicc -O3 -fopenmp -o "$BIN_PATH/shallow_omp_mpi" shallow_omp_mpi.c tools_omp_mpi.c main_omp_mpi.c -lm

# Check compilation
if [ $? -eq 0 ]; then
    echo "Compilation succeed. Executiong OMP-MPI..."
    trap cleanup EXIT INT TERM
    mpirun -n $NB_PROC "$BIN_PATH/shallow_omp_mpi ${INPUT_PATH}/param_simple.txt"
else
    echo "Compilation error."
    exit 1
fi

cleanup
