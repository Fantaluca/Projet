## Table of Contents
- [Overview](#overview)
- [Mathematical Model](#mathematical-model)
- [Numerical Implementation](#numerical-implementation)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Building from Source](#building-from-source)

## Overview
This project was developed as part of the [INFO0939](https://people.montefiore.uliege.be/geuzaine/INFO0939/) High-Performance Scientific Computing course at the University of Liège. The main objective was to enhance the performance of a finite difference solver for shallow water equations through various parallel computing paradigms:

- Shared-memory parallelism using OpenMP
- Distributed computing using MPI
- GPU acceleration (upcoming feature)

## Mathematical Model

### Governing Equations

The shallow water equations solved in this project are:

$$ \frac{\partial \eta}{\partial t} = -\nabla \cdot (h\mathbf{u}) $$

$$ \frac{\partial \mathbf{u}}{\partial t} = -g\nabla \eta - \gamma\mathbf{u} $$

Where:
- $\eta$ : Water elevation
- $\mathbf{u}$ : Depth-averaged velocity vector
- $h$ : Water depth
- $g$ : Gravitational acceleration
- $\gamma$ : Friction coefficient

### Boundary and Initial Conditions

**Initial Conditions:**
- Zero free-surface elevation: $\eta^0_{i,j} = 0$
- Zero initial velocity: $u^0_{i,j} = v^0_{i,j} = 0, \forall(i, j)$

**Boundary Conditions:**
- Impermeable boundaries (left, bottom, right):
  - $u^n_{0,j} = 0$
  - $u^n_{imax,j} = 0$
  - $v^n_{i,0} = 0, \forall i, j$
- Top boundary condition:
  - $v^n_{i,jmax} = A \sin(2\pi ft), \forall i$

## Numerical Implementation

### Spatial Discretization

The solver employs a diamond scheme for spatial discretization:
- Elevation $\eta$ computed at cell centers
- Velocity components ($u$, $v$) computed on cell boundaries

### Temporal Integration

Using explicit Euler scheme, the discretized equations are:

$$ \frac{\eta_{i,j}^{n+1} - \eta_{i,j}^n}{\Delta t} = -\frac{h(\mathbf{x}^{u_{i+1,j}})u_{i+1,j}^n - h(\mathbf{x}^{u_{i,j}})u_{i,j}^n}{\Delta x} - \frac{h(\mathbf{x}^{v_{i,j+1}})v_{i,j+1}^n - h(\mathbf{x}^{v_{i,j}})v_{i,j}^n}{\Delta y} $$

$$ \frac{u_{i,j}^{n+1} - u_{i,j}^n}{\Delta t} = -g\frac{\eta_{i,j}^{n+1} - \eta_{i-1,j}^{n+1}}{\Delta x} - \gamma u_{i,j}^n $$

$$ \frac{v_{i,j}^{n+1} - v_{i,j}^n}{\Delta t} = -g\frac{\eta_{i,j}^{n+1} - \eta_{i,j-1}^{n+1}}{\Delta y} - \gamma v_{i,j}^n $$

Where velocity component locations are defined as:

$$ \mathbf{x}^{u_{i,j}} := (x_a + i\Delta x, y_a + (j + \frac{1}{2})\Delta y) $$

$$ \mathbf{x}^{v_{i,j}} := (x_a + (i + \frac{1}{2})\Delta x, y_a + j\Delta y) $$

## Features

- 🚀 OpenMP optimization for shared-memory parallelism
- 🌐 MPI implementation for distributed computing
- 📊 Comprehensive output for analysis and visualization
- 🔄 Flexible boundary condition handling
- 🎯 GPU acceleration support (coming soon)

## Installation

### Prerequisites

- C compiler (GCC recommended)
- OpenMP support
- MPI implementation:
  - Windows: Microsoft MPI
  - Linux: OpenMPI or MPICH

### System Requirements

- Windows 10/11 or Linux-based OS
- Multi-core processor for OpenMP optimization
- Network connection for MPI distributed computing

## Building from Source

### Windows Build

1. Navigate to the desired implementation directory:
```bash
cd source/OMP  # or MPI, GPU
```

2. Run the compilation script:
```bash
./set_omp.bat  # for OpenMP implementation
# or
./set_mpi.bat  # for MPI implementation
```

To adjust thread count, modify `OMP_NUM_THREADS` in the script.

### Linux Build

1. Navigate to the implementation directory:
```bash
cd source/OMP  # or MPI, GPU
```

2. Execute the build script:
```bash
./set_omp.sh  # for OpenMP implementation
# or
./set_mpi.sh  # for MPI implementation
```

To configure process count, modify `NB_PROC` in the script.

#### Custom MPI Installation

For non-standard MPI installations, uncomment and modify the following in the `.sh` script:
```bash
# Define MPI paths
export MPI_ROOT=/usr/local/mpi
export MPI_INCLUDE="$MPI_ROOT/include"
export MPI_LIB="$MPI_ROOT/lib"
```

