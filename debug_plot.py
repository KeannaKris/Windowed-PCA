import pandas as pd
import plotly.graph_objects as go
import os

print("=== DEBUGGING PCA PLOTTING ISSUES ===")

# Check if input file exists
data_path = "data/results/all_chromosomes_pc1.csv"
print(f"1. Checking for input file: {data_path}")

if os.path.exists(data_path):
    print("   ✓ File exists")
    try:
        df = pd.read_csv(data_path)
        print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        print(f"   First few rows:")
        print(df.head())
    except Exception as e:
        print(f"   ✗ Error reading file: {e}")
else:
    print("   ✗ File not found!")
    print("   Looking for other CSV files...")
    import glob
    csv_files = glob.glob("**/*.csv", recursive=True)
    for f in csv_files[:10]:  # Show first 10
        print(f"     Found: {f}")

# Check plotly
print("\n2. Testing plotly...")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6], mode='lines'))
    print("   ✓ Plotly working")
    
    # Test file creation
    os.makedirs("plots", exist_ok=True)
    fig.write_html("plots/debug_test.html")
    print("   ✓ Can create HTML files in plots/")
    
    if os.path.exists("plots/debug_test.html"):
        print("   ✓ Test file created successfully")
    else:
        print("   ✗ Test file not found after creation")
        
except Exception as e:
    print(f"   ✗ Plotly error: {e}")

# Check permissions
print("\n3. Checking permissions...")
print(f"   Current directory: {os.getcwd()}")
print(f"   Plots directory writable: {os.access('plots', os.W_OK) if os.path.exists('plots') else 'Directory does not exist'}")

print("\n=== DEBUG COMPLETE ===")
