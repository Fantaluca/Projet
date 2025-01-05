#!/bin/bash
set -e

module load EasyBuild/2023a
module load Clang/18.1.8-GCCcore-12.3.0-CUDA-12.2.0

# Compilation settings
BIN_PATH="../../bin/shallow_gpu"
INPUT_PATH="../../input_data/base_case/"
OUTPUT_PATH="../../output"
export SHALLOW_INPUT_DIR="$INPUT_PATH"

# Create directories
mkdir -p "$OUTPUT_PATH"

# Disable OpenMP profiling to prevent trace file creation
unset LIBOMPTARGET_PROFILE
unset OMP_TOOL_LIBRARIES
unset OMP_TOOL

# Compile
echo "Starting compilation..."
clang -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda \
    -o "${BIN_PATH}" \
    shallow_gpu.c \
    tools_gpu.c \
    main_gpu.c \
    -lm 


if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    ${BIN_PATH} param_simple.txt
else
    echo "Compilation failed!"
    exit 1
fi