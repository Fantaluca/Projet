#!/bin/bash
set -e

# Path settings
BIN_PATH="../../bin"
INPUT_PATH="../../input_data/base_case"

# Create temporary user if it doesn't exist
TMP_USER="omprunner"
if ! id "$TMP_USER" &>/dev/null; then
    useradd -M -r $TMP_USER
fi

# Give temporary permissions
chown -R $TMP_USER:$TMP_USER /opt/hpsc_container

# Compilation
gcc -O3 -o ${BIN_PATH}/shallow_serial shallow_serial.c tools_serial.c  -lm

# Execute as temporary user
if [ $? -eq 0 ]; then
    su $TMP_USER -c "${BIN_PATH}/shallow_serial ${INPUT_PATH}/param_simple.txt"
else
    echo "Compilation error"
    exit 1
fi

# Restore permissions
chown -R root:root /opt/hpsc_container