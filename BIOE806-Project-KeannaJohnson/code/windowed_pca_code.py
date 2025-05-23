# Improved Windowed PCA Analysis for Human Pangenome Data
# This script performs windowed PCA analysis on pangenome VCF data

import os
import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import gzip
import time
from collections import defaultdict

# Function to load VCF data
def load_vcf_data(vcf_path, chromosome=None, start=None, end=None):
    """
    Load VCF data, optionally filtering for a specific region
    
    Parameters:
    -----------
    vcf_path : str
        Path to VCF file
    chromosome : str, optional
        Chromosome to filter for (e.g., 'chr15')
    start : int, optional
        Start position for region
    end : int, optional
        End position for region
    
    Returns:
    --------
    dict
        Dictionary with callset data
    """
    print(f"Loading VCF data from {vcf_path}")
    fields = ['variants/CHROM', 'variants/POS', 'calldata/GT', 'samples']
    
    try:
        if chromosome:
            # Format region string
            chrom_without_prefix = chromosome.replace('chr', '')
            
            # If start and end are provided, use them
            if start is not None and end is not None:
                region = f"{chrom_without_prefix}:{start}-{end}"
                alt_region = f"chr{chrom_without_prefix}:{start}-{end}"
            else:
                region = f"{chrom_without_prefix}:"
                alt_region = f"chr{chrom_without_prefix}:"
            
            try:
                # Try first format
                callset = allel.read_vcf(vcf_path, fields=fields, region=region)
                if callset is None or len(callset.get('variants/POS', [])) == 0:
                    raise Exception("No variants found with first format")
            except Exception as e1:
                print(f"First format failed: {e1}")
                try:
                    # Try alternative format
                    callset = allel.read_vcf(vcf_path, fields=fields, region=alt_region)
                except Exception as e2:
                    raise Exception(f"Failed to load data with both formats: {e2}")
        else:
            # Load all data
            callset = allel.read_vcf(vcf_path, fields=fields)
            
        # Verify data was loaded
        if callset is None or 'variants/POS' not in callset or len(callset['variants/POS']) == 0:
            print("Warning: No variants found in the specified region")
            return None
            
        print(f"Successfully loaded data with {len(callset['samples'])} samples")
        print(f"Variant position range: {min(callset['variants/POS'])} - {max(callset['variants/POS'])}")
        print(f"Total variants: {len(callset['variants/POS'])}")
        return callset
    except Exception as e:
        print(f"Error loading VCF: {str(e)}")
        return None

# Function to assign samples to populations
def assign_populations(samples, population_file):
    """Assign samples to populations using a population mapping file"""
    pop_assignments = {}
    
    # Load population file
    if population_file and os.path.exists(population_file):
        print(f"Loading population assignments from {population_file}")
        with open(population_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    sample_id = parts[0]
                    population = parts[1]
                    pop_assignments[sample_id] = population
    else:
        print("Population file not found or not provided")
        return None
    
    # Count samples in each population
    pop_counts = defaultdict(int)
    for pop in pop_assignments.values():
        pop_counts[pop] += 1
    
    print("Population assignments:")
    for pop, count in pop_counts.items():
        print(f"  {pop}: {count} samples")
    
    return pop_assignments

# Function to perform PCA on genotype data
def process_window_pca(genotypes, scaler='patterson'):
    """Perform PCA on a window of genotypes"""
    try:
        # Convert to alternate allele counts
        ac = genotypes.to_n_alt()
        
        # Skip windows with no variation
        if ac.shape[0] == 0 or np.all(ac == ac[0, 0]):
            return None, 0
            
        # Perform PCA
        coords, model = allel.pca(ac, n_components=2, scaler=scaler)
        
        return (coords[:, 0], # PC1
                ac.shape[0])  # number of variants
    except Exception as e:
        print(f"Error in PCA: {str(e)}")
        return None, 0

# Function to perform windowed PCA analysis
def windowed_PCA(callset, window_size=10000, window_step=5000, min_variants=3):
    """Perform windowed PCA analysis for variants in callset"""
    if callset is None:
        print("No data to analyze")
        return None
    
    # Extract data
    chrom = callset['variants/CHROM']
    pos = callset['variants/POS']
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    samples = callset['samples']
    
    # Verify data
    if len(pos) == 0:
        print("No positions found in callset")
        return None
        
    print(f"Data Summary:")
    print(f"Total variants: {len(pos)}")
    print(f"Genotype shape: {genotypes.shape}")
    print(f"Number of samples: {len(samples)}")
    print(f"Position range: {min(pos)} - {max(pos)}")
    print(f"Number of unique positions: {len(np.unique(pos))}")
    
    # Initialize results
    pc1_values = []
    positions = []
    chromosomes = []
    n_variants = []
    
    # Get unique chromosomes
    unique_chroms = np.unique(chrom)
    print(f"Processing {len(unique_chroms)} chromosomes: {', '.join(unique_chroms)}")
    
    # Process windows
    total_windows = 0
    successful_windows = 0
    
    for current_chrom in unique_chroms:
        print(f"\nProcessing {current_chrom}")
        
        # Filter for current chromosome
        chrom_mask = chrom == current_chrom
        chrom_pos = pos[chrom_mask]
        chrom_genotypes = genotypes[chrom_mask]
        
        if len(chrom_pos) == 0:
            print(f"No variants found on {current_chrom}, skipping")
            continue
        
        print(f"Found {len(chrom_pos)} variants on {current_chrom}")
        print(f"Position range: {min(chrom_pos)} - {max(chrom_pos)}")
        
        # Calculate window starts
        min_pos = min(chrom_pos)
        max_pos = max(chrom_pos)
        window_starts = np.arange(min_pos, max_pos, window_step)
        
        print(f"Processing {len(window_starts)} windows from {min_pos} to {max_pos}")
        
        # Process windows
        for start in window_starts:
            total_windows += 1
            end = start + window_size
            window_mask = (chrom_pos >= start) & (chrom_pos < end)
            window_variants = np.sum(window_mask)
            
            # Debug output every 100 windows
            if total_windows % 100 == 0:
                print(f"Window {total_windows}: pos {start}-{end}, {window_variants} variants")
            
            if window_variants >= min_variants:
                window_geno = chrom_genotypes[window_mask]
                pc1, n_var = process_window_pca(window_geno)
                
                if pc1 is not None:
                    pc1_values.append(pc1)
                    positions.append(start + window_size//2)  # Use middle of window as position
                    chromosomes.append(current_chrom)
                    n_variants.append(n_var)
                    successful_windows += 1
            
            # Print progress periodically
            if total_windows % 1000 == 0:
                print(f"Processed {total_windows} windows, {successful_windows} successful")
    
    print(f"\nCompleted windowed PCA: {successful_windows} successful windows out of {total_windows} total")
    
    if successful_windows == 0:
        print("No successful windows, analysis failed")
        return None
        
    return {
        'pc1': pc1_values,
        'positions': positions,
        'chromosomes': chromosomes,
        'n_variants': n_variants,
        'samples': samples
    }

# Function to calculate population differentiation
def calculate_differentiation(pca_results, pop_assignments):
    """Calculate differentiation between populations for each window"""
    if pca_results is None or 'pc1' not in pca_results or len(pca_results['pc1']) == 0:
        print("No PCA results to analyze")
        return pd.DataFrame()
    
    results = []
    samples = pca_results['samples']
    
    for i, pc1 in enumerate(pca_results['pc1']):
        # Group PC1 values by population
        pop_pc1 = defaultdict(list)
        
        for j, sample in enumerate(samples):
            if sample in pop_assignments:
                pop = pop_assignments[sample]
                pop_pc1[pop].append(pc1[j])
        
        # Calculate means for each population
        pop_means = {pop: np.mean(vals) if vals else np.nan 
                     for pop, vals in pop_pc1.items()}
        
        # Calculate variance of population means (higher = more differentiation)
        valid_means = [m for m in pop_means.values() if not np.isnan(m)]
        if len(valid_means) > 1:
            diff_score = np.var(valid_means)
        else:
            diff_score = np.nan
        
        # Calculate pairwise differences between population means
        pairwise_diffs = {}
        pops = list(pop_means.keys())
        for i in range(len(pops)):
            for j in range(i+1, len(pops)):
                if not np.isnan(pop_means[pops[i]]) and not np.isnan(pop_means[pops[j]]):
                    pair_name = f"{pops[i]}_vs_{pops[j]}"
                    pairwise_diffs[pair_name] = abs(pop_means[pops[i]] - pop_means[pops[j]])
        
        # Store result
        result = {
            'chromosome': pca_results['chromosomes'][i],
            'position': pca_results['positions'][i],
            'diff_score': diff_score,
            'n_variants': pca_results['n_variants'][i]
        }
        
        # Add population means
        for pop, mean in pop_means.items():
            result[f'mean_{pop}'] = mean
        
        # Add pairwise differences
        for pair, diff in pairwise_diffs.items():
            result[f'diff_{pair}'] = diff
        
        results.append(result)
    
    return pd.DataFrame(results)

# Function to plot differentiation scores
def plot_differentiation(diff_results, chrom, output_file=None):
    """Create plot of differentiation scores across a chromosome"""
    if diff_results.empty:
        print("No data to plot")
        return None
    
    # Filter to chromosome
    plot_data = diff_results[diff_results['chromosome'] == chrom]
    
    if plot_data.empty:
        print(f"No data for chromosome {chrom}")
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add differentiation score line
    fig.add_trace(go.Scatter(
        x=plot_data['position'],
        y=plot_data['diff_score'],
        mode='lines',
        name='Differentiation Score',
        line=dict(color='blue', width=2)
    ))
    
    # Add points for higher differentiation
    high_diff = plot_data[plot_data['diff_score'] > plot_data['diff_score'].quantile(0.9)]
    fig.add_trace(go.Scatter(
        x=high_diff['position'],
        y=high_diff['diff_score'],
        mode='markers',
        name='High Differentiation',
        marker=dict(color='red', size=8)
    ))
    
    # Define regions of interest
    regions = {
        'SLC24A5': {'chr': '15', 'start': 48120000, 'end': 48145000},
        'EDAR': {'chr': '2', 'start': 108946000, 'end': 109016000},
        'LCT': {'chr': '2', 'start': 135787000, 'end': 136817000}
    }
    
    # Highlight known regions if present in this chromosome
    for name, region in regions.items():
        if region['chr'] == chrom or f"chr{region['chr']}" == chrom:
            fig.add_vrect(
                x0=region['start'],
                x1=region['end'],
                fillcolor="green",
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=name,
                annotation_position="top left"
            )
    
    # Update layout
    fig.update_layout(
        title=f"Population Differentiation on {chrom}",
        xaxis_title="Position",
        yaxis_title="Differentiation Score",
        hovermode="closest",
        template="plotly_white",
        height=600,
        width=1000
    )
    
    # Save if output file provided
    if output_file:
        fig.write_html(output_file)
    
    return fig

# Function to target specific regions
def analyze_regions(vcf_path, population_file, regions, window_size=5000, window_step=2000, min_variants=2, output_dir='results'):
    """
    Analyze specific genomic regions of interest
    
    Parameters:
    -----------
    vcf_path : str
        Path to VCF file
    population_file : str
        Path to population mapping file
    regions : dict
        Dictionary of regions to analyze
    window_size : int
        Size of windows in base pairs
    window_step : int
        Step size between windows
    min_variants : int
        Minimum variants per window
    output_dir : str
        Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each region
    for name, region in regions.items():
        chrom = region['chr']
        # Use a buffer around the region
        buffer = 500000  # 500kb buffer
        start = max(1, region['start'] - buffer)
        end = region['end'] + buffer
        
        print(f"\n========== Analyzing {name} region ({chrom}:{start}-{end}) ==========")
        
        # 1. Load VCF data for this region
        callset = load_vcf_data(vcf_path, chrom, start, end)
        if callset is None or 'variants/POS' not in callset or len(callset['variants/POS']) == 0:
            print(f"No data available for {name} region, skipping")
            continue
        
        # 2. Assign populations
        pop_assignments = assign_populations(callset['samples'], population_file)
        if pop_assignments is None:
            print(f"Failed to assign populations for {name} region, skipping")
            continue
        
        # 3. Run windowed PCA with smaller windows
        print(f"Running windowed PCA with window size {window_size}bp, step {window_step}bp")
        pca_results = windowed_PCA(callset, window_size, window_step, min_variants)
        if pca_results is None:
            print(f"PCA analysis failed for {name} region, skipping")
            continue
        
        # 4. Calculate differentiation
        diff_results = calculate_differentiation(pca_results, pop_assignments)
        if diff_results.empty:
            print(f"No differentiation results for {name} region")
            continue
            
        # 5. Save results
        region_file = os.path.join(output_dir, f"{name}_differentiation.csv")
        diff_results.to_csv(region_file, index=False)
        print(f"Saved differentiation results to {region_file}")
        
        # 6. Create visualization
        fig = plot_differentiation(diff_results, chrom)
        if fig:
            vis_file = os.path.join(output_dir, f"{name}_plot.html")
            fig.write_html(vis_file)
            print(f"Saved visualization to {vis_file}")
        
        # 7. Calculate stats for the specific region vs flanking
        region_windows = diff_results[
            (diff_results['position'] >= region['start']) & 
            (diff_results['position'] <= region['end'])
        ]
        
        flanking_windows = diff_results[
            ((diff_results['position'] >= start) & 
             (diff_results['position'] < region['start'])) | 
            ((diff_results['position'] > region['end']) & 
             (diff_results['position'] <= end))
        ]
        
        region_diff = region_windows['diff_score'].mean() if not region_windows.empty else np.nan
        flanking_diff = flanking_windows['diff_score'].mean() if not flanking_windows.empty else np.nan
        
        print(f"\n{name} Region Analysis:")
        print(f"  Windows in target region: {len(region_windows)}")
        print(f"  Windows in flanking regions: {len(flanking_windows)}")
        print(f"  Mean differentiation in target: {region_diff:.4f}")
        print(f"  Mean differentiation in flanking: {flanking_diff:.4f}")
        
        if not np.isnan(region_diff) and not np.isnan(flanking_diff) and flanking_diff > 0:
            ratio = region_diff / flanking_diff
            print(f"  Target/flanking ratio: {ratio:.2f}x")
            if ratio > 1.2:
                print(f"  {name} shows elevated differentiation!")
            else:
                print(f"  {name} does not show elevated differentiation")
        else:
            print(f"  Unable to calculate enrichment due to insufficient data")

# Main function
def run_analysis(vcf_path, population_file, target_chrom=None, 
                window_size=10000, window_step=5000, min_variants=2, 
                output_dir='results'):
    """Run the windowed PCA analysis pipeline"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    # 1. Load VCF data
    print(f"\n1. Loading VCF data" + (f" for {target_chrom}" if target_chrom else ""))
    callset = load_vcf_data(vcf_path, target_chrom)
    if callset is None:
        print("Failed to load VCF data, exiting")
        return None
    
    # 2. Assign populations
    print("\n2. Assigning samples to populations")
    pop_assignments = assign_populations(callset['samples'], population_file)
    if pop_assignments is None:
        print("Failed to assign populations, exiting")
        return None
    
    # 3. Run windowed PCA
    print(f"\n3. Running windowed PCA (window size: {window_size}, step: {window_step})")
    pca_results = windowed_PCA(callset, window_size, window_step, min_variants)
    if pca_results is None:
        print("No PCA results, exiting")
        return None
    
    # 4. Calculate differentiation
    print("\n4. Calculating population differentiation")
    diff_results = calculate_differentiation(pca_results, pop_assignments)
    
    # Save differentiation results
    chrom_suffix = f"_{target_chrom}" if target_chrom else "_all"
    diff_file = os.path.join(output_dir, f"differentiation{chrom_suffix}.csv")
    diff_results.to_csv(diff_file, index=False)
    print(f"Saved differentiation results to {diff_file}")
    
    # 5. Create visualizations by chromosome
    print("\n5. Creating visualizations")
    for chrom in diff_results['chromosome'].unique():
        fig = plot_differentiation(diff_results, chrom)
        if fig:
            vis_file = os.path.join(output_dir, f"differentiation_plot_{chrom}.html")
            fig.write_html(vis_file)
            print(f"Saved visualization for {chrom} to {vis_file}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.2f} seconds")
    
    return diff_results

# Main execution
if __name__ == "__main__":
    # Path to VCF file
    vcf_path = "data/hprc-v1.1-mc-grch38.vcfbub.a100k.wave.vcf.gz"
    
    # Path to population mapping file
    population_file = "data/population_mapping.txt"
    
    # Create results directory
    os.makedirs('data/results', exist_ok=True)
    
    # Define regions of interest
    regions = {
        'SLC24A5': {'chr': '15', 'start': 48120000, 'end': 48145000},
        'EDAR': {'chr': '2', 'start': 108946000, 'end': 109016000},
        'LCT': {'chr': '2', 'start': 135787000, 'end': 136817000}
    }
    
    # First, try to analyze the specific regions directly
    analyze_regions(
        vcf_path=vcf_path,
        population_file=population_file,
        regions=regions,
        window_size=5000,  # Smaller windows for fine-scale structure
        window_step=2000,  # Smaller steps for better resolution
        min_variants=2,    # Lower threshold for sparse data
        output_dir='data/results'
    )
    
    # Then, if resources allow, do the full chromosome analysis
    run_full_analysis = False  # Set to True if you want to run full chromosome analysis
    
    if run_full_analysis:
        # Define chromosomes to analyze
        target_chroms = ['15', '2']  # SLC24A5 on chr15, EDAR and LCT on chr2
        
        # Run analysis for each chromosome
        for chrom in target_chroms:
            print(f"\n========== Analyzing full chromosome {chrom} ==========")
            run_analysis(
                vcf_path=vcf_path,
                population_file=population_file,
                target_chrom=chrom,
                window_size=10000,  # Reduced from 50kb
                window_step=5000,   # Reduced from 25kb
                min_variants=2,     # Lower threshold may help with missing data
                output_dir='data/results'
            )