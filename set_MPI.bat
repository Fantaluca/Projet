@echo off
REM Assurez-vous que ces chemins correspondent à votre installation
set MSMPI_INC="C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set MSMPI_LIB64="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

REM Compilation avec gcc, incluant les options d'optimisation et OpenMP
gcc -O3 -fopenmp -I%MSMPI_INC% -L%MSMPI_LIB64% -o %1 %2 %3 -lmsmpi -lm

REM use 'mpiexec -n <nombre_de_processus> %1' pour exécuter.