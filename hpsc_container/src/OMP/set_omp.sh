#!/bin/bash
set -e

# Path settings
BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case/"
export SHALLOW_INPUT_DIR="$INPUT_PATH"

# OpenMP settings
export OMP_NUM_THREADS=16

# Create temporary user if it doesn't exist
TMP_USER="omprunner"
if ! id "$TMP_USER" &>/dev/null; then
    useradd -M -r $TMP_USER
fi

# Give temporary permissions
chown -R $TMP_USER:$TMP_USER /opt/hpsc_container

# Compilation
gcc -O3 -fopenmp -o ${BIN_PATH}/shallow_omp shallow_omp.c tools_omp.c main_omp.c -lm

# Execute as temporary user
if [ $? -eq 0 ]; then
    su $TMP_USER -c "${BIN_PATH}/shallow_omp ${INPUT_PATH}/param_simple.txt"
else
    echo "Compilation error"
    exit 1
fi

# Restore permissions
chown -R root:root /opt/hpsc_container