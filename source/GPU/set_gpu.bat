@echo off
setlocal enabledelayedexpansion

REM Define nb of threads
set OMP_NUM_THREADS=10


set "BIN_PATH=%~dp0..\..\bin"
if not exist "!BIN_PATH!" mkdir "!BIN_PATH!"

REM Check if powershell is used
where powershell >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "EXE_PATH=!BIN_PATH!\shallow_omp.exe"
) else (
    set "EXE_PATH=%BIN_PATH%\shallow_omp.exe"
)

REM Compilation
gcc -O3 -fopenmp -o "!EXE_PATH!" shallow_omp.c tools_omp.c main_omp.c -lm

REM Check compilation
if !ERRORLEVEL! EQU 0 (
    REM Execution
    "!EXE_PATH!" param_simple.txt
    pause
) else (
    echo Compilation error
    pause
    exit /b 1
)

endlocal