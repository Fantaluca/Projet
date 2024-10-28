@echo off
set MSMPI_INC="C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set MSMPI_LIB64="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
set NB_PROC=5

REM Compilation
gcc -mconsole -O3 -fopenmp -I%MSMPI_INC% -L%MSMPI_LIB64% -o ../../bin/shallow_mpi shallow_mpi.c tools_mpi.c main_mpi.c -lmsmpi -lm

REM Execution
mpiexec -n %NB_PROC% ../../bin/shallow_mpi.exe param_simple.txt