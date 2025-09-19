#!/usr/bin/env python3
"""
Optimized visualization script for genome-wide PCA results with data thinning.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import os
from collections import defaultdict

def smart_thin_data(pop_data, target_points=5000):
    """
    Intelligently thin data for visualization while preserving patterns
    """
    if len(pop_data) <= target_points:
        return pop_data
    
    step = max(1, len(pop_data) // target_points)
    thinned_data = pop_data.iloc[::step].copy()
    
    print(f"  Thinned {len(pop_data):,} points to {len(thinned_data):,} points (step={step})")
    return thinned_data

# Create output directory
os.makedirs("plots_optimized", exist_ok=True)
print("Loading genomic PCA data...")

# Load the combined results with proper data type handling
data_path = "data/results/all_chromosomes_pc1.csv"
df = pd.read_csv(data_path, low_memory=False)
print(f"Loaded {len(df)} data points across {df['chromosome'].nunique()} chromosomes")

# Clean up population column
df = df.dropna(subset=['population'])
df['population'] = df['population'].astype(str)
df = df[df['population'].str.len() > 0]
df = df[~df['population'].isin(['nan', 'None', 'null'])]

print(f"After cleaning: {len(df)} data points")
print(f"Populations: {sorted(df['population'].unique())}")

# Calculate population means per window
print("Computing population means per window...")
pop_means = df.groupby(['window_id', 'chromosome', 'position', 'population'])['PC1'].mean().reset_index()

# Transparency level
TRANSPARENCY = 0.7

# Define colors for populations
population_colors = {
    "AFR": "rgba(30, 136, 229, 0.7)", "GWD": "rgba(30, 136, 229, 0.7)",
    "AMR": "rgba(255, 193, 7, 0.7)", "PUR": "rgba(255, 193, 7, 0.7)",
    "EAS": "rgba(76, 175, 80, 0.7)", "JPT": "rgba(76, 175, 80, 0.7)",
    "CHS": "rgba(76, 175, 80, 0.7)", "CHB": "rgba(76, 175, 80, 0.7)",
    "EUR": "rgba(244, 67, 54, 0.7)", "GBR": "rgba(244, 67, 54, 0.7)",
    "FIN": "rgba(244, 67, 54, 0.7)", "TSI": "rgba(244, 67, 54, 0.7)",
    "IBS": "rgba(244, 67, 54, 0.7)", "SAS": "rgba(156, 39, 176, 0.7)",
    "PJL": "rgba(156, 39, 176, 0.7)", "BEB": "rgba(156, 39, 176, 0.7)",
    "GIH": "rgba(156, 39, 176, 0.7)", "STU": "rgba(156, 39, 176, 0.7)",
    "ITU": "rgba(156, 39, 176, 0.7)", "REF": "rgba(117, 117, 117, 0.7)",
    "MKK": "rgba(121, 85, 72, 0.7)", "LWK": "rgba(96, 125, 139, 0.7)",
    "ACB": "rgba(63, 81, 181, 0.7)", "ASW": "rgba(0, 150, 136, 0.7)",
    "ESN": "rgba(255, 87, 34, 0.7)", "MSL": "rgba(139, 195, 74, 0.7)",
    "YRI": "rgba(205, 220, 57, 0.7)", "PEL": "rgba(255, 152, 0, 0.7)",
    "CLM": "rgba(255, 235, 59, 0.7)", "MXL": "rgba(233, 30, 99, 0.7)",
    "KHV": "rgba(33, 150, 243, 0.7)", "CDX": "rgba(0, 188, 212, 0.7)",
    "ASL": "rgba(255, 105, 180, 0.7)",
}

# Create genome position mapping for continuous x-axis
print("Creating genomic position mapping...")
chrom_order = [str(i) for i in range(1, 23)] + ['X', 'Y']
chrom_lengths = {}
chrom_offsets = {}
current_offset = 0

for chrom in chrom_order:
    chrom_data = pop_means[pop_means['chromosome'] == chrom]
    if len(chrom_data) > 0:
        chrom_max = chrom_data['position'].max()
        chrom_lengths[chrom] = chrom_max
        chrom_offsets[chrom] = current_offset
        current_offset += chrom_max + 10000000

# Add genome-wide position
pop_means['genome_position'] = pop_means.apply(
    lambda row: chrom_offsets.get(row['chromosome'], 0) + row['position'], axis=1
)

# =============================================================================
# 1. GENOME-WIDE COMPARISON OF ALL POPULATIONS WITH THINNING
# =============================================================================
print("Creating genome-wide population comparison plot...")
fig1 = go.Figure()

populations = sorted(pop_means['population'].unique())
for pop in populations:
    if pop in ['REF', 'NA', '', 'nan']:
        continue
    pop_data = pop_means[pop_means['population'] == pop].sort_values('genome_position')
    if len(pop_data) == 0:
        continue
        
    # Apply data thinning
    pop_data_thinned = smart_thin_data(pop_data, target_points=5000)
        
    color = population_colors.get(pop, f'rgba(99, 99, 99, {TRANSPARENCY})')
    fig1.add_trace(
        go.Scatter(
            x=pop_data_thinned['genome_position'],
            y=pop_data_thinned['PC1'],
            mode='lines',
            name=pop,
            line=dict(color=color, width=1.5),
            hovertemplate=
            f'{pop}<br>' +
            'Chr: %{customdata[0]}<br>' +
            'Position: %{customdata[1]:,}<br>' +
            'PC1: %{y:.4f}' +
            '<extra></extra>',
            customdata=pop_data_thinned[['chromosome', 'position']].values
        )
    )

# Add chromosome boundaries
for i in range(len(chrom_order) - 1):
    current_chrom = chrom_order[i]
    next_chrom = chrom_order[i + 1]
    
    if current_chrom in chrom_offsets and next_chrom in chrom_offsets:
        boundary_pos = chrom_offsets[current_chrom] + chrom_lengths.get(current_chrom, 0) + 5000000
        fig1.add_vline(x=boundary_pos, line_dash="dot", line_color="gray", line_width=1)

# Add chromosome labels
chrom_centers = []
chrom_labels = []
for chrom in chrom_order:
    if chrom in chrom_offsets:
        center = chrom_offsets[chrom] + chrom_lengths.get(chrom, 0) / 2
        chrom_centers.append(center)
        chrom_labels.append(f'Chr{chrom}')

fig1.update_layout(
    title="Population Structure (PC1) Across the Genome",
    xaxis_title="Genomic Position",
    yaxis_title="PC1 Value",
    width=1400,
    height=700,
    hovermode='closest',
    template='plotly_white',
    xaxis=dict(
        tickmode='array',
        tickvals=chrom_centers,
        ticktext=chrom_labels,
        tickangle=45
    )
)

fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='black')

print("Saving genome-wide plot...")
fig1.write_html(
    "plots_optimized/genome_wide_population_comparison.html",
    include_plotlyjs='cdn',
    config={'displayModeBar': False}
)
print("✓ Saved: plots_optimized/genome_wide_population_comparison.html")

# =============================================================================
# 2. INDIVIDUAL POPULATION TOGGLE VIEW WITH THINNING
# =============================================================================
print("Creating individual population toggle plot...")
fig2 = go.Figure()

non_ref_populations = [p for p in populations if p not in ['REF', 'NA', '', 'nan']]

for i, pop in enumerate(non_ref_populations):
    pop_data = pop_means[pop_means['population'] == pop].sort_values('genome_position')
    if len(pop_data) == 0:
        continue
        
    # Apply data thinning
    pop_data_thinned = smart_thin_data(pop_data, target_points=5000)
        
    color = population_colors.get(pop, f'rgba(99, 99, 99, {TRANSPARENCY})')
    fig2.add_trace(
        go.Scatter(
            x=pop_data_thinned['genome_position'],
            y=pop_data_thinned['PC1'],
            mode='lines',
            name=pop,
            line=dict(color=color, width=2),
            visible=(i == 0),
            hovertemplate=
            f'{pop}<br>' +
            'Chr: %{customdata[0]}<br>' +
            'Position: %{customdata[1]:,}<br>' +
            'PC1: %{y:.4f}' +
            '<extra></extra>',
            customdata=pop_data_thinned[['chromosome', 'position']].values
        )
    )

# Add chromosome boundaries
for i in range(len(chrom_order) - 1):
    current_chrom = chrom_order[i]
    next_chrom = chrom_order[i + 1]
    
    if current_chrom in chrom_offsets and next_chrom in chrom_offsets:
        boundary_pos = chrom_offsets[current_chrom] + chrom_lengths.get(current_chrom, 0) + 5000000
        fig2.add_vline(x=boundary_pos, line_dash="dot", line_color="gray", line_width=1)

# Create buttons for each population
buttons = []

# Add "Show All" button first
all_visible = [True] * len(non_ref_populations)
buttons.append(dict(
    label="Show All",
    method="update",
    args=[{"visible": all_visible},
          {"title": "PC1 Values - All Populations"}]
))

# Add individual population buttons
for i, pop in enumerate(non_ref_populations):
    visibility = [False] * len(non_ref_populations)
    visibility[i] = True

    buttons.append(dict(
        label=pop,
        method="update",
        args=[{"visible": visibility},
              {"title": f"PC1 Values for {pop} Across the Genome"}]
    ))

fig2.update_layout(
    title="PC1 Values - Toggle Populations",
    xaxis_title="Genomic Position",
    yaxis_title="PC1 Value",
    width=1400,
    height=700,
    hovermode='closest',
    template='plotly_white',
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=buttons,
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.01,
            xanchor="left",
            y=1.15,
            yanchor="top"
        ),
    ],
    xaxis=dict(
        tickmode='array',
        tickvals=chrom_centers,
        ticktext=chrom_labels,
        tickangle=45
    )
)

fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', zeroline=True, zerolinecolor='black')

print("Saving toggle plot...")
fig2.write_html(
    "plots_optimized/individual_population_toggle.html",
    include_plotlyjs='cdn',
    config={'displayModeBar': False}
)
print("✓ Saved: plots_optimized/individual_population_toggle.html")

# =============================================================================
# 3. CHROMOSOME-SPECIFIC COMPARISON SUBPLOTS WITH THINNING
# =============================================================================
print("Creating chromosome-specific comparison plots...")

fig3 = make_subplots(
    rows=4, cols=6,
    subplot_titles=[f'Chr {c}' for c in chrom_order],
    shared_yaxes=True,
    vertical_spacing=0.08,
    horizontal_spacing=0.05
)

for i, chrom in enumerate(chrom_order):
    row = (i // 6) + 1
    col = (i % 6) + 1
    
    chrom_data = pop_means[pop_means['chromosome'] == chrom]
    if len(chrom_data) == 0:
        continue
    
    for pop in non_ref_populations:
        pop_chrom_data = chrom_data[chrom_data['population'] == pop].sort_values('position')
        if len(pop_chrom_data) == 0:
            continue
            
        # Apply thinning for chromosome-specific plots
        pop_chrom_data_thinned = smart_thin_data(pop_chrom_data, target_points=500)
            
        color = population_colors.get(pop, f'rgba(99, 99, 99, {TRANSPARENCY})')
        fig3.add_trace(
            go.Scatter(
                x=pop_chrom_data_thinned['position'],
                y=pop_chrom_data_thinned['PC1'],
                mode='lines',
                name=pop,
                line=dict(color=color, width=1),
                showlegend=(i == 0),
                hovertemplate=f'{pop}<br>Pos: %{{x:,}}<br>PC1: %{{y:.3f}}<extra></extra>'
            ),
            row=row, col=col
        )

fig3.update_layout(
    title="Population Structure by Chromosome",
    width=1600,
    height=1000,
    template='plotly_white',
    legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
)

print("Saving chromosome-specific plots...")
fig3.write_html(
    "plots_optimized/chromosome_specific_comparison.html",
    include_plotlyjs='cdn',
    config={'displayModeBar': False}
)
print("✓ Saved: plots_optimized/chromosome_specific_comparison.html")

# =============================================================================
# 4. SUMMARY STATISTICS TABLE
# =============================================================================
print("Creating summary statistics...")

summary_stats = []
for pop in non_ref_populations:
    pop_data = pop_means[pop_means['population'] == pop]
    if len(pop_data) == 0:
        continue

    summary_stats.append({
        'Population': pop,
        'Mean_PC1': pop_data['PC1'].mean(),
        'Std_PC1': pop_data['PC1'].std(),
        'Min_PC1': pop_data['PC1'].min(),
        'Max_PC1': pop_data['PC1'].max(),
        'Windows': len(pop_data)
    })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv("plots_optimized/population_summary_statistics.csv", index=False)

print(f"\nSummary Statistics:")
print(summary_df.round(4))

print(f"\nOptimized plot files created:")
print("1. plots_optimized/genome_wide_population_comparison.html")
print("2. plots_optimized/individual_population_toggle.html") 
print("3. plots_optimized/chromosome_specific_comparison.html")
print("4. plots_optimized/population_summary_statistics.csv")
print("\n🎉 All optimized plots completed successfully!")
