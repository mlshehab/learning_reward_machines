#!/bin/bash

# Check if at least one argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name> -depth <value>"
    exit 1
fi

# Define script path
SCRIPT="$1.py"

# Change directory to scripts
cd scripts || { echo "Error: Could not change to scripts directory."; exit 1; }

# Check if the script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: $SCRIPT does not exist."
    exit 1
fi

# Print which experiment is being run
echo "Running experiment: $1 with arguments ${@:2}"

# Run the experiment with the provided arguments
python "$SCRIPT" "${@:2}"
