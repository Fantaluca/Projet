#!/bin/bash

# Define number of processes
export NB_PROC=1

# Define MPI paths if needed (uncomment and modify if using custom MPI installation)
#export MPI_ROOT=/usr/local/mpi
#export MPI_INCLUDE="$MPI_ROOT/include"
#export MPI_LIB="$MPI_ROOT/lib"

# Define paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN_PATH="$SCRIPT_DIR/../../bin"
mkdir -p "$BIN_PATH"

# Compilation
# If using custom MPI paths, use:
# gcc -O3 -fopenmp -I$MPI_INCLUDE -L$MPI_LIB -o "$BIN_PATH/shallow_mpi" shallow_mpi.c tools_mpi.c main_mpi.c -lm -lmpi
# Otherwise, use standard compilation:
mpicc -O3 -fopenmp -o "$BIN_PATH/shallow_mpi" shallow_mpi.c tools_mpi.c main_mpi.c -lm 

# Check if compilation succeed
if [ $? -eq 0 ]; then
    # Execution
    mpirun -n $NB_PROC "$BIN_PATH/shallow_mpi" param_simple.txt
else
    echo "Compilation error"
    exit 1
fi