@echo off
setlocal enabledelayedexpansion

REM Define paths and number of processes
set MSMPI_INC="C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set MSMPI_LIB64="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
set NB_PROC=4

REM Define bin path
set "BIN_PATH=%~dp0..\..\bin"
if not exist "!BIN_PATH!" mkdir "!BIN_PATH!"

REM Check if powershell is used
where powershell >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "EXE_PATH=!BIN_PATH!\shallow_mpi.exe"
) else (
    set "EXE_PATH=%BIN_PATH%\shallow_mpi.exe"
)

REM Compilation
gcc -mconsole -O3 -fopenmp -I%MSMPI_INC% -L%MSMPI_LIB64% -o "!EXE_PATH!" shallow_mpi.c tools_mpi.c main_mpi.c -lmsmpi -lm

REM Check compilation
if !ERRORLEVEL! EQU 0 (
    REM Execution
    mpiexec -n %NB_PROC% "!EXE_PATH!" param_simple.txt
    pause
) else (
    echo Compilation error
    pause
    exit /b 1
)

endlocal