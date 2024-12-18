#!/bin/bash

# Chargement des modules nécessaires sur Nic5
module load GCC
module load OpenMPI

# Paramètres d'exécution
export OMP_NUM_THREADS=10       # Nombre de threads OpenMP par processus MPI
NB_PROC=4                       # Nombre de processus MPI

# Définition des chemins

BIN_PATH="../../bin"
INPUT_PATH="$../../input_data/base_case"
OUTPUT_PATH="../../output"
mkdir -p "$BIN_PATH"
mkdir -p "$OUTPUT_PATH"

# Fonction pour nettoyer les processus MPI en cas d'erreur ou d'interruption
cleanup() {
    pkill -9 mpirun || true
    pkill -9 orted || true  # Kill Open MPI daemon processes
    sleep 2
}

# Nettoyer les processus existants avant de commencer
cleanup

# Compilation avec GCC, OpenMP et MPI
mpicc -O3 -fopenmp -o "$BIN_PATH/shallow_omp_mpi" shallow_omp_mpi.c tools_omp_mpi.c main_omp_mpi.c -lm

# Vérification de la compilation
if [ $? -eq 0 ]; then
    echo "Compilation réussie. Exécution du programme OMP-MPI..."
    # Utiliser trap pour nettoyer si le script est interrompu
    trap cleanup EXIT INT TERM
    mpirun -n $NB_PROC "$BIN_PATH/shallow_omp_mpi" "param_simple.txt"
else
    echo "Erreur de compilation."
    exit 1
fi

# Nettoyage après l'exécution
cleanup
