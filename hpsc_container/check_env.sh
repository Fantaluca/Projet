#!/bin/bash
set -e

echo "=== GPU Environment Check ==="
date
nvidia-smi

echo -e "\n=== Compiler Information ==="
clang --version

echo -e "\n=== Environment Variables ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LIBOMPTARGET_PLUGIN_PATH: $LIBOMPTARGET_PLUGIN_PATH"
echo "LIBOMPTARGET_NVPTX_ARCH: $LIBOMPTARGET_NVPTX_ARCH"

echo -e "\n=== MPI Configuration ==="
MPI_INCLUDE_PATH=$(mpicc -show | grep -o -- '-I[^ ]*' | tr '\n' ' ')
echo "MPI Include Path: ${MPI_INCLUDE_PATH}"

# Function to check and print library dependencies
check_library() {
    local lib_name=$1
    local lib_path=$2
    echo -e "\nChecking $lib_name:"
    if [ -f "$lib_path" ]; then
        ldd "$lib_path"
    else
        echo "Library not found: $lib_path"
    fi
}

echo -e "\n=== Library Dependencies ==="
check_library "OpenMP Target" "/usr/lib/llvm-18/lib/libomptarget.so.18.1"
check_library "OpenMP CUDA RTL" "/usr/lib/llvm-18/lib/libomptarget.rtl.cuda.so.18.1"
check_library "MPI" "/usr/lib/x86_64-linux-gnu/libmpi.so"