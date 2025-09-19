#!/bin/bash
#SBATCH --job-name=multi_select_pca
#SBATCH --time=1:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/multi_select_%j.out
#SBATCH --error=logs/multi_select_%j.err

export PATH="$HOME/miniconda3/bin:$PATH"

echo "Starting multi-select PCA plotting job..."
python multi_select_pca.py
echo "Multi-select plotting completed"
