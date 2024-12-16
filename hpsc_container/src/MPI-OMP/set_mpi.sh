#!/bin/bash

# Define number of MPI processes
export NB_PROC=5

# Define number of OpenMP threads per MPI process
export OMP_NUM_THREADS=4

# Define paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_src[0]}" )" && pwd )"
BIN_PATH="$SCRIPT_DIR/../../bin"
mkdir -p "$BIN_PATH"

# Compilation
mpicc -O3 -fopenmp -o "$BIN_PATH/shallow_mpi" shallow_mpi.c tools_mpi.c main_mpi.c -lm 

# Check if compilation succeed
if [ $? -eq 0 ]; then
    # Execution with MPI and OpenMP
    mpirun -n $NB_PROC "$BIN_PATH/shallow_mpi" param_simple.txt
else
    echo "Compilation error"
    exit 1
fi
