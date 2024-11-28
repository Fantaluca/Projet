#!/bin/bash

# Usage : ./debug.sh shallow_mpi_exe shallow_mpi.c tools_mpi.c main_mpi.c

# Définir les chemins d'inclusion et de librairie pour MPI sur macOS
MSMPI_INC="/opt/homebrew/opt/open-mpi/include"
MSMPI_LIB64="/opt/homebrew/opt/open-mpi/lib"

# Vérifier si les répertoires existent
if [ ! -d "$MSMPI_INC" ]; then
  echo "Le répertoire MSMPI_INC n'existe pas"
fi

if [ ! -d "$MSMPI_LIB64" ]; then
  echo "Le répertoire MSMPI_LIB64 n'existe pas"
fi

# Vérification des arguments
if [ -z "$1" ]; then
  echo "Le nom de l'exécutable n'est pas fourni"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Le premier fichier source n'est pas fourni"
  exit 1
fi

if [ -z "$3" ]; then
  echo "Le deuxième fichier source n'est pas fourni"
  exit 1
fi

if [ -z "$4" ]; then
  echo "Le troisième fichier source n'est pas fourni"
  exit 1
fi

# Compilation avec mpicc (OpenMPI sous macOS), incluant OpenMP
mpicc -O3 -fopenmp -I"$MSMPI_INC" -L"$MSMPI_LIB64" -o "$1" "$2" "$3" "$4" -lm

if [ $? -ne 0 ]; then
  echo "La compilation a échoué"
  exit 1
fi

echo "Compilation terminée. Utilisez 'mpiexec -n <nombre_de_processus> $1' pour exécuter."
# mpiexec -n 4 shallow_mpi_exe param_simple.txt
