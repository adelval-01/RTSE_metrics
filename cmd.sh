#!/bin/bash

# Stop on any error
set -e

# Create and activate conda environment
echo "Creating conda environment..."
conda env create -f rtse_environment.yml -n rtse_env

echo "Activating conda environment..."
# NOTE: 'conda activate' only works in interactive shells, so use 'source'
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rtse_env

# Install pip packages
echo "Installing pip packages..."
pip install -r rtse_requirements.txt

# Download and Unzip RTSE model data
echo "Downloading RTSE model data..."
gdown --fuzzy https://drive.google.com/file/d/109IRfgzH9bi1Jp3uYLed-2yZkEavh_6F/view?usp=drive_link
echo "Unzipping RTSE model data..."
unzip -o rtse_model.zip

# Run metrics evaluations
echo "Running latency measurements..."
python3 rtse_latency.py
python3 rtse_latency.py ./configs/config-3.json
python3 rtse_latency.py ./configs/config-4.json
python3 rtse_latency.py ./configs/config-5.json
python3 rtse_latency.py ./configs/config-5.json

# Zip the results
echo "Zipping results..."
zip -r latency_results.zip time_profiling

# Deactivate the conda environment
echo "Deactivating conda environment..."
conda deactivate

# Remove the conda environment
echo "Removing conda environment..."
conda env remove -n rtse_env -y

echo "All tasks completed successfully!"
