#!/bin/bash

# Bash settings: fail on any error and display all commands being run.
set -e
set -x

# Python must be 3.8 or higher.
python --version

# Set up a virtual environment.
python -m venv xmagical_testing
source xmagical_testing/bin/activate

# Install dependencies.
apt-get update
apt-get install -y gcc libgl1-mesa-dev gfortran libopenblas-dev liblapack-dev
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .[test]

N_CPU=$(grep -c ^processor /proc/cpuinfo)

# Run all tests.
pytest --durations=10 -n "${N_CPU}" tests

# Clean-up.
deactivate
rm -rf xmagical_testing/