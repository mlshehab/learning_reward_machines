#!/bin/bash

# Set -e option to exit immediately if any command fails
set -e

# Run automate.py with depth set to 5
echo "Running automate.py with depth 9..."
python automate_ne.py -depth 9
echo "automate.py completed."

# Run format_xml.py
echo "Running format_xml.py..."
python format_xml.py
echo "format_xml.py completed."

# Run SAT.py
echo "Running SAT.py..."
python SAT.py
echo "SAT.py completed."

echo "All scripts ran successfully!"
