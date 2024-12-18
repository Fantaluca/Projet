#!/bin/bash
set -e

# Compilation settings
PARAM_FILE="../../input_data/base_case/param_simple.txt"
BIN_PATH="../../bin/shallow_gpu"

# OpenMP settings
export OMP_TARGET_OFFLOAD=MANDATORY

# Disable OpenMP profiling to prevent trace file creation
unset LIBOMPTARGET_PROFILE
unset OMP_TOOL_LIBRARIES
unset OMP_TOOL

# Compile
echo "Starting compilation..."
clang-18 -O3 \
    -fopenmp \
    -fopenmp-targets=nvptx64-nvidia-cuda \
    -Xopenmp-target -march=sm_89 \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -L/usr/lib/llvm-18/lib \
    --cuda-path=${CUDA_HOME} \
    -fopenmp-version=51 \
    -foffload-lto \
    -o "${BIN_PATH}" \
    shallow_gpu.c \
    tools_gpu.c \
    main_gpu.c \
    -lm -lomptarget -lomptarget.rtl.cuda

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    ${BIN_PATH} ${PARAM_FILE}
else
    echo "Compilation failed!"
    exit 1
fi