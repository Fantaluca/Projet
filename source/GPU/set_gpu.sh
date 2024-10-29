#!/bin/bash

# Répertoire contenant le binaire
BIN_DIR="../../bin"

# Création du répertoire bin s'il n'existe pas
mkdir -p "$BIN_DIR"

# Chemin complet du binaire
BIN_PATH="$BIN_DIR/shallow_gpu"

# Compilation avec Clang
clang -O3 \
    -fopenmp \
    -fopenmp-targets=nvptx64-nvidia-cuda \
    -o "$BIN_PATH" \
    shallow_gpu.c tools_gpu.c main_gpu.c \
    -lm

# Vérification de la compilation
if [ $? -eq 0 ]; then
    echo "Compilation réussie!"
    echo "Exécution du programme..."
    "$BIN_PATH" param_simple.txt
else
    echo "Erreur de compilation!"
    exit 1
fi