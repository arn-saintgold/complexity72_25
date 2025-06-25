#!/bin/bash

# This script sets up a Python environment.

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ $? -eq 0 ]; then
    echo "Pip upgraded successfully."
else
    echo "Error: Failed to upgrade pip."
    exit
fi

pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "Python environment set up successfully."
else
    echo "Error: Failed to set up Python environment."
    exit
fi


