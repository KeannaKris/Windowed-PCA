#!/usr/bin/env python3
"""
Multi-select population comparison script with improved layout and display.
Creates an interactive plot where users can select any combination of populations to compare.
"""
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

def smart_thin_data(pop_data, target_points=5000):
    """Intelligently thin data for visualization while preserving patterns"""
    if len(pop_data) <= target_points:
        return pop_data
    
    step = max(1, len(pop_data) // target_points)
    thinned_data = pop_data.iloc[::step].copy()
    
    print(f"  Thinned {len(pop_data):,} points to {len(thinned_data):,} points (step={step})")
    return thinned_data

# Create output directory
os.makedirs("plots_optimized", exist_ok=True)
print("Loading genomic PCA data...")

# Load and clean data
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

# Define colors for populations (better contrast and visibility)
population_colors = {
    "ACB": "#FF6B6B", "ASW": "#4ECDC4", "ESN": "#45B7D1", "GWD": "#96CEB4",
    "LWK": "#FFEAA7", "MSL": "#DDA0DD", "YRI": "#98D8C8", "MKK": "#F7DC6F",
    "BEB": "#BB8FCE", "GIH": "#85C1E9", "ITU": "#F8C471", "PJL": "#82E0AA",
    "STU": "#F1948A", "CHB": "#AED6F1", "CHS": "#A9DFBF", "CDX": "#FAD7A0",
    "JPT": "#D7BDE2", "KHV": "#A3E4D7", "CLM": "#F9E79F", "MXL": "#D5A6BD",
    "PEL": "#AED6F1", "PUR": "#F5B7B1", "FIN": "#D2B4DE", "GBR": "#A9CCE3",
    "IBS": "#A3E4D7", "TSI": "#F9E79F", "ASL": "#FF9FF3", "REF": "#BDC3C7"
}

# Create genome position mapping
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

# Add chromosome labels
chrom_centers = []
chrom_labels = []
for chrom in chrom_order:
    if chrom in chrom_offsets:
        center = chrom_offsets[chrom] + chrom_lengths.get(chrom, 0) / 2
        chrom_centers.append(center)
        chrom_labels.append(f'Chr{chrom}')

# =============================================================================
# MULTI-SELECT POPULATION COMPARISON
# =============================================================================
print("Creating multi-select population comparison plot...")
fig = go.Figure()

# Filter populations (exclude REF)
non_ref_populations = [p for p in sorted(pop_means['population'].unique()) 
                      if p not in ['REF', 'NA', '', 'nan']]

print(f"Creating plot for {len(non_ref_populations)} populations...")

# Add all populations to the plot
for i, pop in enumerate(non_ref_populations):
    print(f"Processing population {pop}...")
    pop_data = pop_means[pop_means['population'] == pop].sort_values('genome_position')
    if len(pop_data) == 0:
        continue
        
    # Apply data thinning
    pop_data_thinned = smart_thin_data(pop_data, target_points=3000)  # Reduced for better performance
        
    color = population_colors.get(pop, '#7F8C8D')  # Default gray
    
    fig.add_trace(
        go.Scatter(
            x=pop_data_thinned['genome_position'],
            y=pop_data_thinned['PC1'],
            mode='lines',
            name=pop,
            line=dict(color=color, width=2),
            visible='legendonly',  # Start with all populations hidden, user can click to show
            hovertemplate=
            f'<b>{pop}</b><br>' +
            'Chr: %{customdata[0]}<br>' +
            'Position: %{customdata[1]:,.0f}<br>' +
            'PC1: %{y:.3f}<br>' +
            '<extra></extra>',
            customdata=pop_data_thinned[['chromosome', 'position']].values,
            legendgroup=pop  # Group legend items
        )
    )

# Add chromosome boundaries
print("Adding chromosome boundaries...")
for i in range(len(chrom_order) - 1):
    current_chrom = chrom_order[i]
    next_chrom = chrom_order[i + 1]
    
    if current_chrom in chrom_offsets and next_chrom in chrom_offsets:
        boundary_pos = chrom_offsets[current_chrom] + chrom_lengths.get(current_chrom, 0) + 5000000
        fig.add_vline(
            x=boundary_pos, 
            line_dash="dot", 
            line_color="lightgray", 
            line_width=1,
            layer="below"
        )

# Improved layout with better spacing and readability
fig.update_layout(
    title={
        'text': "Interactive Population Structure Comparison<br><sub>Click population names in legend to show/hide. Multiple selections allowed.</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 16}
    },
    xaxis_title="Genomic Position",
    yaxis_title="PC1 Value",
    width=1600,  # Wider for better visibility
    height=900,  # Taller to prevent compression
    hovermode='closest',
    template='plotly_white',
    
    # Improved axis formatting
    xaxis=dict(
        tickmode='array',
        tickvals=chrom_centers,
        ticktext=chrom_labels,
        tickangle=45,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=False
    ),
    
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=2,
        # Add some padding to prevent compression
        range=[pop_means['PC1'].min() * 1.1, pop_means['PC1'].max() * 1.1]
    ),
    
    # Improved legend
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.02,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="lightgray",
        borderwidth=1,
        font=dict(size=11),
        itemclick="toggle",  # Allow toggling
        itemdoubleclick="toggleothers"  # Double-click to show only that population
    ),
    
    # Add some margin for the legend
    margin=dict(r=200, t=100, b=80, l=80),
    
    # Annotations with instructions
    annotations=[
        dict(
            text="💡 Instructions:<br>" +
                 "• Click legend items to show/hide populations<br>" +
                 "• Double-click to show only one population<br>" +
                 "• Hover over lines for detailed information",
            showarrow=False,
            x=1.02,
            y=0.02,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(240,240,240,0.8)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=10)
        )
    ]
)

print("Saving multi-select population comparison plot...")
fig.write_html(
    "plots_optimized/multi_select_population_comparison.html",
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'population_comparison',
            'height': 900,
            'width': 1600,
            'scale': 2
        }
    }
)

print("✓ Saved: plots_optimized/multi_select_population_comparison.html")

# Print summary statistics
print(f"\nPlot Summary:")
print(f"- {len(non_ref_populations)} populations available for selection")
print(f"- PC1 value range: {pop_means['PC1'].min():.2f} to {pop_means['PC1'].max():.2f}")
print(f"- Genomic coverage: {len(chrom_order)} chromosomes")
print(f"- Total data points after thinning: {len(non_ref_populations) * 3000:,}")

print("\n🎉 Multi-select population comparison plot completed!")
print("Usage: Click population names in the legend to toggle visibility.")
print("Double-click a population name to show only that population.")
