#!/bin/bash
#SBATCH --job-name=pca_plots_optimized
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/plots_opt_%j.out
#SBATCH --error=logs/plots_opt_%j.err

# Set up conda environment
export PATH="$HOME/miniconda3/bin:$PATH"

echo "Starting optimized plotting job..."
echo "Memory: 64GB allocated"
echo "Start time: $(date)"

python pca_plots_optimized.py

echo "Completed at: $(date)"
