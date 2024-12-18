#!/bin/bash
set -e

# Architecture settings
export OMP_NUM_THREADS=4
export NB_PROC=5

# Compilation settings
PARAM_FILE="../../input_data/base_case/param_simple.txt"
BIN_PATH="../../bin/shallow_pml"

# Get MPI include path
MPI_INCLUDE_PATH=$(mpicc -show | grep -o -- '-I[^ ]*' | tr '\n' ' ')

# Create temporary user if it doesn't exist
TMP_USER="mpirunner"
if ! id "$TMP_USER" &>/dev/null; then
    useradd -M -r $TMP_USER
fi

# Give temporary permissions
chown -R $TMP_USER:$TMP_USER /opt/hpsc_container

# Compilation
mpicc -O3 -fopenmp -o ../../bin/shallow_pml shallow_pml.c tools_pml.c main_pml.c -lm

# Execute as temporary user
if [ $? -eq 0 ]; then
    su $TMP_USER -c "mpirun -n $NB_PROC ${BIN_PATH} ${PARAM_FILE}"
else
    echo "Compilation error"
    exit 1
fi

# Restore permissions
chown -R root:root /opt/hpsc_container