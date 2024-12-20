#!/bin/bash
#SBATCH --job-name="shallow_omp"
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=shallow_omp_%j.out
#SBATCH --error=shallow_omp_%j.err
#SBATCH --chdir=/home/ulg/info0939/ademeure/High-Performance-Scientific-Computing/hpsc_container/src/OMP
#SBATCH --exclusive


module load GCC
module load OpenMPI


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}




# Path definition
SCRIPT_DIR="$( pwd )"
BIN_PATH="$SCRIPT_DIR/../../bin"
INPUT_PATH="$SCRIPT_DIR/../../input_data/base_case/"
OUTPUT_PATH="$SCRIPT_DIR/../../output"

export SHALLOW_INPUT_DIR="$INPUT_PATH"

# Create directories
mkdir -p "$BIN_PATH"
mkdir -p "$OUTPUT_PATH"



# Compilation
gcc -O3 -fopenmp -o ${BIN_PATH}/shallow_omp shallow_omp.c tools_omp.c main_omp.c -lm

if [ $? -eq 0 ]; then
    srun --cpus-per-task=${OMP_NUM_THREADS} ${BIN_PATH}/shallow_omp param_simple.txt
else
    echo "Compilation error"
    exit 1
fi
