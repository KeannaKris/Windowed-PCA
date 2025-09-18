# Extract PC1 Values Across All Windows
import os
import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
from collections import defaultdict

# Define paths
vcf_path = "test_small.vcf.gz"
population_file = "data/sample_ids.txt"
results_dir = "data/results"
os.makedirs(results_dir, exist_ok=True)

# Parameters
window_size = 2000   # 2kb window
window_step = 1000   # 1kb step
min_variants = 1     # Just need 1 variant

# Load population mapping
def load_populations(population_file):
    """Load population assignments from file"""
    try:
        pop_assignments = {}
        
        with open(population_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"Read {len(lines)} lines from population file")
        
        # Skip header line
        for i, line in enumerate(lines[1:], 1):  # Start from line 1, skip header
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                sample_id = parts[0].strip()
                population = parts[1].strip()
                pop_assignments[sample_id] = population
            else:
                print(f"Warning: Line {i+1} has fewer than 2 columns: {line}")
        
        print(f"Loaded {len(pop_assignments)} population assignments")
        
        # Count populations
        pop_counts = defaultdict(int)
        for pop in pop_assignments.values():
            pop_counts[pop] += 1
        
        print("Population assignments:")
        for pop, count in pop_counts.items():
            print(f"  {pop}: {count} samples")
            
        return pop_assignments
        
    except Exception as e:
        print(f"Error loading population file: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load VCF data for a chromosome
def load_chromosome_data(chromosome):
    """Load VCF data for a chromosome"""
    print(f"\nLoading data for chromosome {chromosome}...")
    try:
        # Try different formats
        formats = [f"CHM13#0#chr{chromosome}", f"chr{chromosome}", f"{chromosome}"]
        
        for fmt in formats:
            try:
                print(f"Trying format: {fmt}")
                callset = allel.read_vcf(vcf_path, region=fmt, 
                                         fields=['variants/CHROM', 'variants/POS', 'calldata/GT', 'samples'])
                
                if callset and 'variants/POS' in callset and len(callset['variants/POS']) > 0:
                    print(f"Successfully loaded {len(callset['variants/POS'])} variants")
                    return callset
            except Exception as e:
                print(f"Error with format {fmt}: {e}")
        
        print(f"Failed to load data for chromosome {chromosome}")
        return None
    except Exception as e:
        print(f"Error loading chromosome data: {e}")
        return None

# NEW: Compute reference PC1 for Procrustes alignment
def compute_reference_pc1(callset, max_missing_rate=0.2, n_variants=500):
    """
    Compute a stable reference PC1 for Procrustes alignment
    """
    print("Computing reference PC1 for Procrustes alignment...")
    
    samples = callset['samples']
    pos = callset['variants/POS']
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    
    print(f"Starting with {len(pos)} variants and {len(samples)} samples")
    
    # Filter high-quality variants (low missing data)
    ac = genotypes.to_n_alt()
    missing_per_variant = np.isnan(ac).sum(axis=1) / ac.shape[1]
    good_variants = missing_per_variant <= max_missing_rate
    
    print(f"Filtering variants: kept {np.sum(good_variants)}/{len(good_variants)} with ≤{max_missing_rate*100}% missing data")
    
    if np.sum(good_variants) < 100:
        print("Too few high-quality variants, relaxing filter...")
        good_variants = missing_per_variant <= 0.5
    
    # Select subset for reference
    good_indices = np.where(good_variants)[0]
    step = max(1, len(good_indices) // n_variants)
    selected_indices = good_indices[::step][:n_variants]
    
    print(f"Selected {len(selected_indices)} variants for reference PCA")
    
    # Extract reference data
    ref_genotypes = genotypes[selected_indices]
    ref_ac = ref_genotypes.to_n_alt()
    
    # Handle missing data (simple zero imputation)
    if np.isnan(ref_ac).any() or np.isinf(ref_ac).any():
        ref_ac = np.nan_to_num(ref_ac, nan=0, posinf=0, neginf=0)
    
    # Remove non-variable variants
    variant_std = np.std(ref_ac, axis=1)
    variable_mask = variant_std > 1e-8
    ref_ac_final = ref_ac[variable_mask]
    
    print(f"Kept {ref_ac_final.shape[0]} variable variants for reference PCA")
    
    if ref_ac_final.shape[0] < 10:
        print("Too few variants for reliable reference PCA")
        return None
    
    # Compute reference PCA
    try:
        ref_coords, ref_model = allel.pca(ref_ac_final, n_components=1, scaler='patterson')
        reference_pc1 = ref_coords[:, 0]
        print(f"Reference PCA successful: PC1 range {np.min(reference_pc1):.3f} to {np.max(reference_pc1):.3f}")
        return reference_pc1
    except Exception as e:
        print(f"Reference PCA failed with Patterson scaler: {e}")
        try:
            ref_coords, ref_model = allel.pca(ref_ac_final, n_components=1, scaler='standard')
            reference_pc1 = ref_coords[:, 0]
            print(f"Reference PCA successful with standard scaler: PC1 range {np.min(reference_pc1):.3f} to {np.max(reference_pc1):.3f}")
            return reference_pc1
        except Exception as e2:
            print(f"Reference PCA failed completely: {e2}")
            return None

# UPDATED: Process windows and extract PC1 values with Procrustes alignment
def extract_pc1_by_windows(chromosome, callset, pop_assignments):
    """Extract PC1 values for all windows on a chromosome with Procrustes alignment"""
    if callset is None:
        print(f"No data available for chromosome {chromosome}")
        return None
    
    # Extract data
    samples = callset['samples']
    pos = callset['variants/POS']
    genotypes = allel.GenotypeArray(callset['calldata/GT'])
    
    print(f"Data summary:")
    print(f"  Variants: {len(pos)}")
    print(f"  Samples: {len(samples)}")
    print(f"  Position range: {min(pos)}-{max(pos)}")
    
    # NEW: Compute reference PC1 for alignment
    reference_pc1 = compute_reference_pc1(callset)
    
    if reference_pc1 is None:
        print("Failed to compute reference PC1, proceeding without Procrustes alignment")
        reference_pc1 = None
    
    # Calculate window starts
    window_starts = np.arange(min(pos), max(pos) + 1, window_step)
    print(f"Processing {len(window_starts)} windows")
    
    # Initialize results list
    all_pc1_data = []
    
    # Process each window
    total_windows = 0
    successful_windows = 0
    
    for start_pos in window_starts:
        end_pos = start_pos + window_size
        total_windows += 1
        
        # Find variants in this window
        window_mask = (pos >= start_pos) & (pos < end_pos)
        window_variants = np.sum(window_mask)
        
        if window_variants >= min_variants:
            window_geno = genotypes[window_mask]
            
            try:
                # Convert to alternate allele counts
                ac = window_geno.to_n_alt()
                
                # Skip windows with no variation
                if ac.shape[0] == 0 or np.all(ac == ac[0, 0]):
                    continue
                
                # Handle NaN and infinite values
                if np.isnan(ac).any() or np.isinf(ac).any():
                    ac = np.nan_to_num(ac, nan=0, posinf=0, neginf=0)
                
                # Remove non-variable variants
                variant_std = np.std(ac, axis=1)
                variable_mask = variant_std > 1e-8
                
                if np.sum(variable_mask) == 0:
                    continue
                    
                ac_final = ac[variable_mask]
                
                # Perform PCA
                try:
                    coords, model = allel.pca(ac_final, n_components=1, scaler='patterson')
                    pc1 = coords[:, 0]
                    
                except Exception as e:
                    # Try with standard scaler if Patterson fails
                    try:
                        coords, model = allel.pca(ac_final, n_components=1, scaler='standard')
                        pc1 = coords[:, 0]
                    except Exception as e2:
                        continue
                
                # NEW: Procrustes alignment - align to reference PC1
                if reference_pc1 is not None:
                    try:
                        correlation = np.corrcoef(pc1, reference_pc1)[0, 1]
                        
                        # If negative correlation, flip the sign
                        if correlation < 0:
                            pc1 = -pc1
                            
                    except Exception as e:
                        # If correlation fails, keep original orientation
                        pass
                
                # Create window data
                window_id = f"chr{chromosome}_{start_pos}_{end_pos}"
                window_center = start_pos + window_size // 2
                
                # Add data for each sample
                for i, sample in enumerate(samples):
                    pop = pop_assignments.get(sample, 'Unknown')
                    all_pc1_data.append({
                        'window_id': window_id,
                        'chromosome': chromosome,
                        'position': window_center,
                        'start': start_pos,
                        'end': end_pos,
                        'sample': sample,
                        'population': pop,
                        'PC1': pc1[i],
                        'n_variants': window_variants,
                        'method': 'procrustes_aligned' if reference_pc1 is not None else 'unaligned'
                    })
                
                successful_windows += 1
                    
            except Exception as e:
                continue
        
        # Print progress
        if total_windows % 100 == 0:
            print(f"Processed {total_windows} windows, {successful_windows} successful")
    
    print(f"Analysis complete: {successful_windows}/{total_windows} windows successful")
    
    if len(all_pc1_data) == 0:
        print("No PC1 data extracted")
        return None
    
    # Convert to DataFrame
    pc1_df = pd.DataFrame(all_pc1_data)
    
    # Save to CSV
    output_file = os.path.join(results_dir, f"chr{chromosome}_pc1_all_windows.csv")
    pc1_df.to_csv(output_file, index=False)
    print(f"Saved PC1 values to {output_file}")
    
    return pc1_df

# Main code to extract PC1 values for chromosomes
def extract_all_pc1_values(chromosomes):
    """Extract PC1 values for all windows on specified chromosomes"""
    print("\n=== Extracting PC1 Values for All Windows with Procrustes Alignment ===")
    
    # Load population assignments
    pop_assignments = load_populations(population_file)
    
    if not pop_assignments:
        print("Failed to load population assignments")
        return
    
    # Process each chromosome
    all_results = {}
    for chrom in chromosomes:
        print(f"\nProcessing chromosome {chrom}")
        
        # Load chromosome data
        callset = load_chromosome_data(chrom)
        
        if callset:
            # Extract PC1 values with Procrustes alignment
            pc1_data = extract_pc1_by_windows(chrom, callset, pop_assignments)
            
            if pc1_data is not None:
                all_results[chrom] = pc1_data
                
                # Create a summary of PC1 by population
                print("\nPC1 summary by population:")
                summary = pc1_data.groupby(['window_id', 'population'])['PC1'].mean().reset_index()
                summary_pivot = summary.pivot(index='window_id', columns='population', values='PC1')
                
                # Save summary
                summary_file = os.path.join(results_dir, f"chr{chrom}_pc1_summary.csv")
                summary_pivot.to_csv(summary_file)
                print(f"Saved PC1 summary to {summary_file}")
    
    # Create combined file with all chromosomes
    if all_results:
        combined_data = pd.concat(all_results.values())
        combined_file = os.path.join(results_dir, "all_chromosomes_pc1.csv")
        combined_data.to_csv(combined_file, index=False)
        print(f"\nSaved combined PC1 values to {combined_file}")

# Run the function to extract all PC1 values
if __name__ == "__main__":
    chromosomes_to_analyze = ['1']
    extract_all_pc1_values(chromosomes_to_analyze)
