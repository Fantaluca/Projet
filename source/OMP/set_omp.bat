@echo off
set OMP_NUM_THREADS=10

REM Compilation
gcc -O3 -fopenmp -o "%~dp0..\..\bin\shallow_omp" shallow_omp.c tools_omp.c main_omp.c -lm

REM Execution avec le nombre de threads défini
"%~dp0..\..\bin\shallow_omp.exe" param_simple.txt