#!/bin/bash

export OMP_NUM_THREADS=10

SCRIPT_DIR="$( cd "$( dirname "${BASH_src[0]}" )" && pwd )"
BIN_PATH="$SCRIPT_DIR/../../bin"
mkdir -p "$BIN_PATH"

clang -O3 -fopenmp=libomp -o "$BIN_PATH/shallow_omp" shallow_omp.c tools_omp.c main_omp.c -lm -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include

if [ $? -eq 0 ]; then
    # Exécution
    "$BIN_PATH/shallow_omp" param_simple.txt
else
    echo "Compilation error"
    exit 1
fi
