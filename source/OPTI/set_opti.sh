#!/bin/bash

# Define number of processes
NB_PROC=5

# Répertoire contenant le binaire
BIN_DIR="../../bin"

# Création du répertoire bin s'il n'existe pas
mkdir -p "$BIN_DIR"

# Chemin complet du binaire
BIN_PATH="$BIN_DIR/shallow_opti"

# Variables d'environnement pour OpenMP
export LIBOMPTARGET_DEBUG=1
export LIBOMPTARGET_INFO=4

# Compilation avec Clang + MPI
OMPI_CC=clang mpicc -O3 -g \
    -fopenmp \
    -fopenmp-targets=nvptx64-nvidia-cuda \
    -Xopenmp-target -march=sm_86 \
    -o "$BIN_PATH" \
    shallow_opti.c tools_opti.c main_opti.c \
    -lm

# Vérification de la compilation
if [ $? -eq 0 ]; then
    echo "Compilation réussie!"
    echo "Exécution du programme..."
    # Exécution avec MPI
    mpirun -n $NB_PROC "$BIN_PATH" param_simple.txt
else
    echo "Erreur de compilation!"
    exit 1
fi