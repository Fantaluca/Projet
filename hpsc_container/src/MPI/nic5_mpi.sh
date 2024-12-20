#!/bin/bash
#SBATCH --job-name="shallow_omp_mpi"
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --output=shallow_mpi_%j.out
#SBATCH --error=shallow_mpi_%j.err
#SBATCH --chdir=/home/ulg/info0939/ademeure/High-Performance-Scientific-Computing/hpsc_container/src/MPI
#SBATCH --exclusive


module load GCC
module load OpenMPI


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-10}
NB_PROC=${SLURM_NTASKS:-4}



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
echo "Compiling..."
mpicc -O3 -fopenmp -o "$BIN_PATH/shallow_mpi" "$SCRIPT_DIR/shallow_mpi.c" "$SCRIPT_DIR/tools_mpi.c" "$SCRIPT_DIR/main_mpi.c" -lm

# Run
if [ $? -eq 0 ]; then
    echo "Compilation succeed. Exec MPI..."
    echo "nb of process MPI : ${NB_PROC}"
    mpirun -n ${NB_PROC} "$BIN_PATH/shallow_mpi" "param_simple.txt"
else
    echo "Error during compilation"
    exit 1
fi
