#!/bin/bash
#SBATCH --job-name=pca_plots
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/plots_%j.out
#SBATCH --error=logs/plots_%j.err

# Set up conda environment
export PATH="$HOME/miniconda3/bin:$PATH"

# Install only plotly (skip kaleido since we're making HTML plots)
conda install -y plotly

echo "Starting plotting job..."
python pca_plots_v3.py
echo "Plotting completed"
