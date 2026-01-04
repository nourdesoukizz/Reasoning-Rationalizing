#!/usr/bin/env python3
"""
Fix the results_dir not defined error in the final analysis notebook
"""

import json

# Read the final analysis notebook
with open('final_analysis_notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# Find and update the data loading cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'load_latest_results_silent' in source and 'LOADING EVALUATION RESULTS' in source:
            print(f"Found problematic cell at index {i}")
            
            # Fix the cell by adding results_dir definition at the beginning
            new_source = '''# Define results directory
results_dir = Path('results')

# Function to load with silent option
def load_latest_results_silent(pattern, silent=False):
    files = list(results_dir.glob(pattern))
    if not files:
        if not silent:
            print(f"⚠️  Not found: {pattern}")
        return None
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    if not silent:
        print(f"✅ Loaded: {latest_file.name} ({len(df)} records)")
    return df

# Print header
print("=" * 70)
print("LOADING EVALUATION RESULTS")
print("=" * 70)

# Load all available results
print("\\n📊 GEMINI MODEL RESULTS:")
print("-" * 50)

# Gemini - Baseline (with special handling for _baseline suffix)
gemini_baseline = None
for pattern in ['gemini_evaluation_results_*_baseline.csv', 'gemini_evaluation_results_*.csv', 'gemini_baseline_*.csv']:
    gemini_baseline = load_latest_results_silent(pattern, silent=True)
    if gemini_baseline is not None:
        print(f"✅ Baseline (NO hints): {len(gemini_baseline)} records")
        break
if gemini_baseline is None:
    print("⚠️  Baseline (NO hints): NOT FOUND")

# Gemini - Correct hints AFTER
gemini_hints_after = load_latest_results_silent('gemini_hints_evaluation_results_*.csv')
if gemini_hints_after is None:
    print("⚠️  Correct Hints AFTER: NOT FOUND")

# Gemini - Correct hints BEFORE
gemini_hints_before = load_latest_results_silent('gemini_hints_BEFORE_evaluation_*.csv')
if gemini_hints_before is None:
    print("⚠️  Correct Hints BEFORE: NOT FOUND")

# Gemini - Incorrect hints AFTER
gemini_ic_hints_after = load_latest_results_silent('gemini_ic_hints_evaluation_results_*.csv')
if gemini_ic_hints_after is None:
    print("⚠️  Incorrect Hints AFTER: NOT FOUND")

# Gemini - Incorrect hints BEFORE
gemini_ic_hints_before = load_latest_results_silent('gemini_ic_hints_BEFORE_evaluation_*.csv')
if gemini_ic_hints_before is None:
    print("⚠️  Incorrect Hints BEFORE: NOT FOUND")

print("\\n📊 OPENAI MODEL RESULTS:")
print("-" * 50)

# OpenAI - Baseline
openai_baseline = load_latest_results_silent('openai_evaluation_results_*.csv')
if openai_baseline is None:
    print("⚠️  Baseline (NO hints): NOT FOUND")

# OpenAI - Correct hints AFTER
openai_hints_after = load_latest_results_silent('openai_hints_evaluation_results_*.csv')
if openai_hints_after is None:
    print("⚠️  Correct Hints AFTER: NOT FOUND")

# OpenAI - Correct hints BEFORE
openai_hints_before = load_latest_results_silent('openai_hints_BEFORE_evaluation_*.csv')
if openai_hints_before is None:
    print("⚠️  Correct Hints BEFORE: NOT FOUND")

# OpenAI - Incorrect hints AFTER
openai_ic_hints_after = load_latest_results_silent('openai_ic_hints_evaluation_results_*.csv')
if openai_ic_hints_after is None:
    print("⚠️  Incorrect Hints AFTER: NOT FOUND")

# OpenAI - Incorrect hints BEFORE (now available!)
openai_ic_hints_before = load_latest_results_silent('openai_ic_hints_BEFORE_evaluation_*.csv')
if openai_ic_hints_before is None:
    print("⚠️  Incorrect Hints BEFORE: NOT FOUND")

# Summary
print("\\n" + "=" * 70)
print("DATA AVAILABILITY SUMMARY")
print("=" * 70)

total_datasets = 10
available_datasets = sum([
    gemini_baseline is not None,
    gemini_hints_after is not None,
    gemini_hints_before is not None,
    gemini_ic_hints_after is not None,
    gemini_ic_hints_before is not None,
    openai_baseline is not None,
    openai_hints_after is not None,
    openai_hints_before is not None,
    openai_ic_hints_after is not None,
    openai_ic_hints_before is not None
])

print(f"\\n📈 Available: {available_datasets}/{total_datasets} datasets")

if available_datasets == 10:
    print("\\n✅ ALL DATASETS AVAILABLE! Ready for complete analysis!")
else:
    # List missing datasets
    missing = []
    if gemini_baseline is None:
        missing.append("Gemini Baseline (no hints)")
    if gemini_hints_after is None:
        missing.append("Gemini Correct Hints AFTER")
    if gemini_hints_before is None:
        missing.append("Gemini Correct Hints BEFORE")
    if gemini_ic_hints_after is None:
        missing.append("Gemini Incorrect Hints AFTER")
    if gemini_ic_hints_before is None:
        missing.append("Gemini Incorrect Hints BEFORE")
    if openai_baseline is None:
        missing.append("OpenAI Baseline (no hints)")
    if openai_hints_after is None:
        missing.append("OpenAI Correct Hints AFTER")
    if openai_hints_before is None:
        missing.append("OpenAI Correct Hints BEFORE")
    if openai_ic_hints_after is None:
        missing.append("OpenAI Incorrect Hints AFTER")
    if openai_ic_hints_before is None:
        missing.append("OpenAI Incorrect Hints BEFORE")
    
    if missing:
        print("\\n⚠️ Missing datasets:")
        for m in missing:
            print(f"   - {m}")

print("\\n✅ Data loading complete!")'''
            
            # Replace the cell content
            notebook['cells'][i]['source'] = new_source.split('\n')
            print("Fixed the cell by adding results_dir = Path('results') at the beginning")
            
            # Save the notebook
            with open('final_analysis_notebook.ipynb', 'w') as f:
                json.dump(notebook, f, indent=1)
            print("✅ Saved updated notebook")
            break

# Also check if there's a cell that should define results_dir before this
found_definition = False
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if '# Define results directory' in source and 'results_dir = Path' in source and 'load_latest_results(' in source:
            # This is the original cell that defines both - might need fixing too
            if 'load_latest_results_silent' not in source:
                print(f"\nFound original data loading cell at index {i} that needs updating")
                # This cell needs the same fix we applied above
                notebook['cells'][i]['source'] = new_source.split('\n')
                found_definition = True
                
                # Save the notebook again
                with open('final_analysis_notebook.ipynb', 'w') as f:
                    json.dump(notebook, f, indent=1)
                print("✅ Also fixed the original data loading cell")

print("\n✅ Final analysis notebook fixed!")
print("\nThe error was caused by results_dir not being defined in the cell.")
print("It's now defined at the beginning of the cell as: results_dir = Path('results')")
print("\n🎉 You can now run the notebook without the NameError!")