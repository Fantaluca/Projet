#!/bin/bash
set -e

# Path settings
BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case"

# Architecture settings
export OMP_NUM_THREADS=4
export NB_PROC=4

# Function to clean up MPI processes
cleanup() {
    pkill -9 mpirun || true
    pkill -9 orted || true  # Kill Open MPI daemon processes
    sleep 2  # Attendre que tous les processus soient bien terminés
}

# Nettoyer les processus existants avant de commencer
cleanup

# Create temporary user if it doesn't exist
TMP_USER="mpirunner"
if ! id "$TMP_USER" &>/dev/null; then
    useradd -M -r $TMP_USER
fi

# Give temporary permissions
chown -R $TMP_USER:$TMP_USER /opt/hpsc_container

# Compilation
mpicc -O3 -fopenmp -o ${BIN_PATH}/shallow_coriolis_pml shallow_coriolis_pml.c tools_coriolis_pml.c main_coriolis_pml.c -lm

# Execute as temporary user
if [ $? -eq 0 ]; then
    # Utiliser trap pour nettoyer si le script est interrompu
    trap cleanup EXIT INT TERM
    su $TMP_USER -c "mpirun -n $NB_PROC ${BIN_PATH}/shallow_coriolis_pml ${INPUT_PATH}/param_simple.txt"
else
    echo "Compilation error"
    exit 1
fi

# Restore permissions
chown -R root:root /opt/hpsc_container

cleanup