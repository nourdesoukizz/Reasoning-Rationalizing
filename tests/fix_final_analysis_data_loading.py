#!/usr/bin/env python3
"""
Fix data loading section in final analysis notebook
"""

import json

# Read notebook
with open('final_analysis_notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# New data loading code
new_data_loading = """# Function to load with silent option
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

# Gemini - Try multiple patterns for baseline
gemini_baseline = None
for pattern in ['gemini_evaluation_results_*.csv', 'gemini_baseline_*.csv', 'gemini_no_hints_*.csv']:
    gemini_baseline = load_latest_results_silent(pattern, silent=True)
    if gemini_baseline is not None:
        print(f"✅ Baseline (NO hints): {len(gemini_baseline)} records")
        break
if gemini_baseline is None:
    print("⚠️  Baseline (NO hints): NOT FOUND - Need to generate from Gemini notebook")

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

# OpenAI - Incorrect hints BEFORE (try multiple patterns)
openai_ic_hints_before = None
for pattern in ['openai_ic_hints_BEFORE_*.csv', 'openai_incorrect_hints_BEFORE_*.csv']:
    openai_ic_hints_before = load_latest_results_silent(pattern, silent=True)
    if openai_ic_hints_before is not None:
        print(f"✅ Incorrect Hints BEFORE: {len(openai_ic_hints_before)} records")
        break
if openai_ic_hints_before is None:
    print("⚠️  Incorrect Hints BEFORE: NOT FOUND - Need to run IC-hints sections in OpenAI hints-before notebook")

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

# List missing datasets
missing = []
if gemini_baseline is None:
    missing.append("Gemini Baseline (no hints) - Run sections 1-10 in notebooks-hintsafter/gemini_evaluation_ide.ipynb")
if openai_ic_hints_before is None:
    missing.append("OpenAI Incorrect Hints BEFORE - Run sections 22-26 in notebooks-hintsbefore/openai_evaluation_ide_hintsbefore.ipynb")

if missing:
    print("\\n⚠️ To generate missing datasets:")
    for i, m in enumerate(missing, 1):
        print(f"   {i}. {m}")
        
# Create placeholder DataFrames for missing data to avoid errors
if gemini_baseline is None and openai_baseline is not None:
    print("\\n📌 Using empty DataFrame for missing Gemini baseline to avoid errors")
    gemini_baseline = pd.DataFrame(columns=openai_baseline.columns)
    
if openai_ic_hints_before is None and openai_ic_hints_after is not None:
    print("📌 Using empty DataFrame for missing OpenAI IC hints BEFORE to avoid errors")
    openai_ic_hints_before = pd.DataFrame(columns=openai_ic_hints_after.columns)

print("\\n✅ Data loading complete! Proceeding with available datasets...")"""

# Find and update the data loading cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Loading Gemini results' in source and 'load_latest_results' in source:
            print(f"Found data loading cell at index {i}")
            # Replace the cell content
            notebook['cells'][i]['source'] = new_data_loading.split('\n')
            print("Updated data loading cell")
            break

# Save the fixed notebook
with open('final_analysis_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\n✅ Fixed data loading in final analysis notebook!")
print("The notebook now:")
print("  - Clearly shows which datasets are available vs missing")
print("  - Provides instructions for generating missing data")
print("  - Creates placeholders to avoid errors when data is missing")
print("  - Shows data counts for all 5 conditions for each model")