#!/bin/bash
set -e

# Path settings
BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case/"
export SHALLOW_INPUT_DIR="$INPUT_PATH"

# Compilation
gcc -O3 -o ${BIN_PATH}/shallow_serial shallow_serial.c tools_serial.c  -lm

# Execute as temporary user
if [ $? -eq 0 ]; then
    ${BIN_PATH}/shallow_serial ${INPUT_PATH}/param_simple.txt
else
    echo "Compilation error"
    exit 1
fi