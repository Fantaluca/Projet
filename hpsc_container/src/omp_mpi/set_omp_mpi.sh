#!/bin/bash
set -e

# Path settings
BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case/"
export SHALLOW_INPUT_DIR="$INPUT_PATH"

# Architecture settings
export OMP_NUM_THREADS=4
export NB_PROC=4

# Function to clean up MPI processes
cleanup() {
    pkill -9 mpirun || true
    pkill -9 orted || true  # Kill Open MPI daemon processes
    sleep 2 
}

cleanup

# Create temporary user if it doesn't exist
TMP_USER="mpirunner"
if ! id "$TMP_USER" &>/dev/null; then
    useradd -M -r $TMP_USER
fi

# Give temporary permissions
chown -R $TMP_USER:$TMP_USER /opt/hpsc_container

# Compilation
mpicc -O3 -fopenmp -o ${BIN_PATH}/shallow_omp_mpi shallow_omp_mpi.c tools_omp_mpi.c main_omp_mpi.c -lm

# Execute as temporary user
if [ $? -eq 0 ]; then
    # Utiliser trap pour nettoyer si le script est interrompu
    trap cleanup EXIT INT TERM
    su $TMP_USER -c "mpirun -n $NB_PROC ${BIN_PATH}/shallow_omp_mpi ${INPUT_PATH}/param_simple.txt"
else
    echo "Compilation error"
    exit 1
fi

# Restore permissions
chown -R root:root /opt/hpsc_container

cleanup