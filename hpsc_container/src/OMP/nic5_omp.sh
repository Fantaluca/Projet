#!/bin/bash
set -e

# Path settings
BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case/"
export SHALLOW_INPUT_DIR="$INPUT_PATH"

# OpenMP settings
export OMP_NUM_THREADS=16

# Compilation
gcc -O3 -fopenmp -o ${BIN_PATH}/shallow_omp shallow_omp.c tools_omp.c main_omp.c -lm

if [ $? -eq 0 ]; then
    ${BIN_PATH}/shallow_omp ${INPUT_PATH}/param_simple.txt
else
    echo "Compilation error"
    exit 1
fi
