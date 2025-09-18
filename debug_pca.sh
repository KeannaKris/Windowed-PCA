#!/bin/bash
#SBATCH --job-name=pca_debug
#SBATCH --time=10:00
#SBATCH --mem=8G
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

export PATH="$HOME/miniconda3/bin:$PATH"

echo "Starting debug..."
python3 -c "
print('Testing imports...')
import allel
print('scikit-allel imported successfully')

print('Testing file access...')
import os
print('VCF exists:', os.path.exists('data/hprc8424.CHM13.merged.norm.vcf.gz'))
print('Population file exists:', os.path.exists('data/sample_ids.txt'))

print('Testing population file loading...')
with open('data/sample_ids.txt', 'r') as f:
    lines = f.readlines()
    print(f'Population file has {len(lines)} lines')
    print('First few lines:', lines[:3])

print('Debug complete')
"
