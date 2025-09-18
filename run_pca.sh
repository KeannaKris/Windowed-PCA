#!/bin/bash
#SBATCH --job-name=pca_conda
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err

# Explicitly set up conda environment
export PATH="$HOME/miniconda3/bin:$PATH"

# Verify python and packages before running
echo "Using Python: $(which python3)"
python3 -c 'import numpy; print("Numpy version:", numpy.__version__)'

# Run the actual analysis
echo "Starting PCA analysis..."
python3 pca_all.py
