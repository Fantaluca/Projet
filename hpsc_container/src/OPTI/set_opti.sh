#!/bin/bash
set -e

# Compilation settings
NB_PROC=1
PARAM_FILE="../../input_data/base_case/param_simple.txt"
BIN_PATH="../../bin/shallow_opti"

# OpenMP memory settings
export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_PROFILE=0


# Get MPI include path
MPI_INCLUDE_PATH=$(mpicc -show | grep -o -- '-I[^ ]*' | tr '\n' ' ')

# Compile
echo "Starting compilation..."
clang-18 -O3 -g \
    -fopenmp \
    -fopenmp-targets=nvptx64-nvidia-cuda \
    -Xopenmp-target -march=sm_89 \
    ${MPI_INCLUDE_PATH} \
    -I${CUDA_HOME}/include \
    -L${CUDA_HOME}/lib64 \
    -L/usr/lib/llvm-18/lib \
    --cuda-path=${CUDA_HOME} \
    -fopenmp-version=51 \
    -foffload-lto \
    -o "${BIN_PATH}" \
    shallow_opti.c \
    tools_opti.c \
    main_opti.c \
    -lm -lomptarget -lomptarget.rtl.cuda -lmpi

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running..."
    cd ${SRC_DIR}
    mpirun --allow-run-as-root \
           -x LIBOMPTARGET_DEBUG=1 \
           -x OMP_TARGET_OFFLOAD \
           -x CUDA_VISIBLE_DEVICES \
           -n $NB_PROC "${BIN_PATH}" "${PARAM_FILE}"
else
    echo "Compilation failed!"
    exit 1
fi