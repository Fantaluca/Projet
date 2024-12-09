#!/bin/bash

# Debug avancé pour OpenMP
export LIBOMPTARGET_DEBUG=1
export LIBOMPTARGET_INFO=4
export LIBOMPTARGET_PROFILE=1
export LIBOMPTARGET_PLUGIN_DEBUG=1
export LIBOMPTARGET_PLUGIN_PROFILE=1

# Configuration CUDA plus détaillée
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_PATH=/usr/local/cuda-12.6
export CUDA_HOME=/usr/local/cuda-12.6

# Configuration OpenMP offloading
export LIBOMPTARGET_PLUGIN_PATH=/usr/lib/llvm-18/lib
export LIBOMPTARGET_PLUGIN=libomptarget.rtl.cuda.so
export LIBOMPTARGET_DEVICETYPE=nvidia
export LIBOMPTARGET_NVPTX_ARCH=sm_86
export LIBOMPTARGET_CUDA_DEVICE=0
export LIBOMPTARGET_CUDA_JIT=1

# Nouvelles variables pour forcer la détection
export OMP_TARGET_OFFLOAD=MANDATORY
export LIBOMPTARGET_STACK_SIZE=2048
export LIBOMPTARGET_HEAP_SIZE=2048

# Configuration pour le compilateur
export OMPI_CC=clang-18
export LIBRARY_PATH=/usr/lib/llvm-18/lib:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/nvvm/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/nvvm/lib64:$LD_LIBRARY_PATH
export PATH=/usr/lib/llvm-18/bin:/usr/local/cuda-12.6/bin:$PATH

# Vérification de l'environnement
echo "=== Environment Check ==="
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LIBOMPTARGET_PLUGIN_PATH: $LIBOMPTARGET_PLUGIN_PATH"
echo "LIBOMPTARGET_PLUGIN: $LIBOMPTARGET_PLUGIN"
echo "LIBOMPTARGET_NVPTX_ARCH: $LIBOMPTARGET_NVPTX_ARCH"
echo "========================"

# Répertoire contenant le binaire
NB_PROC=5
BIN_DIR="../../bin"
mkdir -p "$BIN_DIR"
BIN_PATH="$BIN_DIR/shallow_opti"

# Compilation avec options spécifiques
mpicc -O3 -g \
    -fopenmp \
    -fopenmp-targets=nvptx64-nvidia-cuda \
    -Xopenmp-target -march=sm_86 \
    -I/usr/local/cuda-12.6/include \
    -L/usr/local/cuda-12.6/lib64 \
    -L/usr/local/cuda-12.6/nvvm/lib64 \
    -L/usr/lib/llvm-18/lib \
    --cuda-path=/usr/local/cuda-12.6 \
    -Wl,-rpath,/usr/lib/llvm-18/lib \
    -Wl,-rpath,/usr/local/cuda-12.6/lib64 \
    -fopenmp-version=51 \
    -foffload-lto \
    -o "$BIN_PATH" \
    shallow_opti.c tools_opti.c main_opti.c \
    -lm -lomptarget -lomptarget.rtl.cuda

if [ $? -eq 0 ]; then
    echo "Compilation réussie!"
    echo "Vérification des bibliothèques liées..."
    ldd "$BIN_PATH"
    echo "Exécution du programme..."
    mpirun --mca orte_abort_on_non_zero_status 0 \
           -x LIBOMPTARGET_DEBUG=1 \
           -x LIBOMPTARGET_INFO=4 \
           -x LIBOMPTARGET_PLUGIN_DEBUG=1 \
           -x LIBOMPTARGET_PLUGIN \
           -x LIBOMPTARGET_PLUGIN_PATH \
           -x LIBOMPTARGET_NVPTX_ARCH \
           -x LIBOMPTARGET_CUDA_DEVICE \
           -x LIBOMPTARGET_CUDA_JIT \
           -x OMP_TARGET_OFFLOAD \
           -x LD_LIBRARY_PATH \
           -x CUDA_VISIBLE_DEVICES \
           -n $NB_PROC "$BIN_PATH" param_simple.txt 2>&1 | tee compile.log
else
    echo "Erreur de compilation!"
    echo "Tentative de compilation avec verbose pour plus de détails..."
    OMPI_CC=clang-18 mpicc -v -O3 -g \
        -fopenmp \
        -fopenmp-targets=nvptx64-nvidia-cuda \
        -Xopenmp-target -march=sm_86 \
        -I/usr/local/cuda-12.6/include \
        -L/usr/local/cuda-12.6/lib64 \
        -L/usr/local/cuda-12.6/nvvm/lib64 \
        -L/usr/lib/llvm-18/lib \
        --cuda-path=/usr/local/cuda-12.6 \
        -Wl,-rpath,/usr/lib/llvm-18/lib \
        -Wl,-rpath,/usr/local/cuda-12.6/lib64 \
        -fopenmp-version=51 \
        -foffload-lto \
        -o "$BIN_PATH" \
        shallow_opti.c tools_opti.c main_opti.c \
        -lm -lomptarget -lomptarget.rtl.cuda 2>&1 | tee compile_error.log
    exit 1
fi