#!/bin/bash

export OMP_NUM_THREADS=10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN_PATH="$SCRIPT_DIR/../../bin"
mkdir -p "$BIN_PATH"

# Compilation
gcc -O3 -fopenmp -o "$BIN_PATH/shallow_omp" shallow_omp.c tools_omp.c main_omp.c -lm

# Check if compiation succeed
if [ $? -eq 0 ]; then
    # Execution
    "$BIN_PATH/shallow_omp" param_simple.txt
else
    echo "Compilation error"
    exit 1
fi