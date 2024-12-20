# Shallow Water Simulation with Multiple Implementation Options

This project implements shallow water simulations using various parallel computing approaches including GPU, OpenMP, MPI, and hybrid OpenMP+MPI implementations.

## Project Structure

```
.
├── src/
│   ├── coriolis_pml/     # PML implementation with Coriolis force
│   ├── gpu/              # GPU implementation
│   ├── mpi/              # Pure MPI implementation
│   ├── omp/              # OpenMP implementation
│   └── omp_mpi/          # Hybrid OpenMP+MPI implementation
├── bin/                  # Compiled executables
├── input_data/          # Input files and test cases
└── output/              # Simulation results
```

## Usage Instructions

> **Important Note**: When running on the nic5 cluster, you must use the `nic5_` prefixed scripts (e.g., `nic5_omp.sh`, `nic5_gpu.sh`) instead of the standard setup scripts.

1. Navigate to your chosen implementation directory:
   ```bash
   cd src/CHOSEN_TYPE
   ```
   Where `CHOSEN_TYPE` can be:
   - `coriolis_pml`
   - `gpu`
   - `mpi`
   - `omp`
   - `omp_mpi`

2. Run the setup script:
   ```bash
   ./set_CHOSEN_TYPE.sh     # Standard environment
   # OR
   ./nic5_CHOSEN_TYPE.sh    # On nic5 cluster
   ```
   For example, if using the OpenMP+MPI implementation:
   ```bash
   ./set_omp_mpi.sh        # Standard environment
   # OR
   ./nic5_omp_mpi.sh       # On nic5 cluster
   ```

## Configuration

Each implementation's setup script (`set_*.sh`) allows for manual configuration of execution parameters:

- For OpenMP implementations (`omp`, `omp_mpi`):
  ```bash
  export OMP_NUM_THREADS=4  # Number of OpenMP threads
  ```

- For MPI implementations (`mpi`, `omp_mpi`):
  ```bash
  export NB_PROC=4  # Number of MPI processes
  ```

You can modify these values in the setup script before execution to adjust the parallel configuration according to your system's capabilities.

## Input Data

The default input data path is configured in each setup script:
```bash
INPUT_PATH="../../input_data/base_case/"
```

You can modify this path in the setup script to use different input data sets.

## Output

Simulation results will be stored in the `output/` directory. Each run creates its own timestamped output files for post-processing and analysis.

## Requirements

- MPI implementation (OpenMPI recommended)
- OpenMP-compatible compiler
- CUDA toolkit (for GPU implementation)