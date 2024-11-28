@echo on
set MSMPI_INC="C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set MSMPI_LIB64="C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"

if not exist %MSMPI_INC% echo Le répertoire MSMPI_INC n'existe pas
if not exist %MSMPI_LIB64% echo Le répertoire MSMPI_LIB64 n'existe pas

REM Vérification des arguments
if "%1"=="" echo Le nom de l'exécutable n'est pas fourni
if "%2"=="" echo Le premier fichier source n'est pas fourni
if "%3"=="" echo Le deuxième fichier source n'est pas fourni

REM Compilation avec gcc, incluant les options d'optimisation et OpenMP
gcc -O3 -fopenmp -I%MSMPI_INC% -L%MSMPI_LIB64% -o %1 %2 %3 -lmsmpi -lm

if %errorlevel% neq 0 echo La compilation a échoué

echo Compilation terminée. Utilisez 'mpiexec -n <nombre_de_processus> %1' pour exécuter.
pause