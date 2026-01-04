#!/usr/bin/env python3
"""
Check and ensure that required exports are in place for:
1. Gemini baseline (no hints) - from notebooks-hintsafter/gemini_evaluation_ide.ipynb
2. OpenAI IC hints BEFORE - from notebooks-hintsbefore/openai_evaluation_ide_hintsbefore copy.ipynb
"""

import json
import os
from pathlib import Path

def check_gemini_baseline():
    """Check if Gemini baseline export exists in the notebook"""
    notebook_path = Path("notebooks-hintsafter/gemini_evaluation_ide.ipynb")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    print("=" * 70)
    print("CHECKING GEMINI BASELINE EXPORT")
    print("=" * 70)
    
    # Look for the export section
    found_export = False
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            # Check for baseline export
            if 'output_file = f"gemini_evaluation_results_{timestamp}.csv"' in source:
                print(f"✅ Found baseline export in cell {i}")
                print("   File pattern: gemini_evaluation_results_*.csv")
                found_export = True
                
                # Check if it's pointing to the right directory
                if '../results/' not in source and 'to_csv(output_file' in source:
                    print("⚠️  Export might not be saving to ../results/ directory")
                    print("   Fixing export path...")
                    
                    # Fix the export path
                    fixed_source = source.replace(
                        'output_file = f"gemini_evaluation_results_{timestamp}.csv"',
                        'output_file = f"../results/gemini_evaluation_results_{timestamp}.csv"'
                    )
                    notebook['cells'][i]['source'] = fixed_source.split('\n')
                    
                    # Save the fixed notebook
                    with open(notebook_path, 'w') as f:
                        json.dump(notebook, f, indent=1)
                    print("   ✅ Fixed export path to save in ../results/")
                else:
                    print("   ✅ Export path is correct")
                break
    
    if not found_export:
        print("❌ Baseline export not found - needs to be added")
    
    return found_export

def check_openai_ic_hints_before():
    """Check if OpenAI IC hints BEFORE export exists in the notebook"""
    notebook_path = Path("notebooks-hintsbefore/openai_evaluation_ide_hintsbefore copy.ipynb")
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    print("\n" + "=" * 70)
    print("CHECKING OPENAI IC HINTS BEFORE EXPORT")
    print("=" * 70)
    
    # Look for the IC hints export section
    found_export = False
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            # Check for IC hints BEFORE export
            if 'openai_ic_hints_BEFORE_evaluation' in source and 'to_csv' in source:
                print(f"✅ Found IC hints BEFORE export in cell {i}")
                print("   File pattern: openai_ic_hints_BEFORE_evaluation_*.csv")
                found_export = True
                
                # Check if the export has the right data
                if 'all_ic_results_df' in source or 'ic_results_df' in source:
                    print("   ✅ Export appears to have correct data source")
                else:
                    print("   ⚠️ May need to verify data source")
                break
    
    if not found_export:
        print("❌ IC hints BEFORE export not found")
        print("   Looking for section to add export...")
        
        # Find section 26 or 27 to add export
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'markdown':
                source = ''.join(cell.get('source', []))
                if '## 26' in source or '## 27' in source:
                    print(f"   Found section at cell {i}")
                    # Check next cell for code
                    if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                        code_cell = notebook['cells'][i+1]
                        source = ''.join(code_cell.get('source', []))
                        
                        # Add export code if not present
                        if 'to_csv' not in source and ('ic_metrics' in source or 'ic_results' in source):
                            export_code = '''
# Export IC hints BEFORE results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ic_file = f"../results/openai_ic_hints_BEFORE_evaluation_{timestamp}.csv"

# Prepare export dataframe
if 'all_ic_results_df' in locals():
    export_df = all_ic_results_df
elif 'ic_math_df' in locals() and 'ic_science_df' in locals():
    # Combine math and science results
    ic_math_df['domain'] = 'math'
    ic_science_df['domain'] = 'science'
    export_df = pd.concat([ic_math_df, ic_science_df], ignore_index=True)
else:
    print("Warning: IC results dataframe not found")
    export_df = pd.DataFrame()

if not export_df.empty:
    export_df.to_csv(ic_file, index=False)
    print(f"✅ IC hints BEFORE results saved to: {ic_file}")
else:
    print("⚠️ No IC hints results to export")
'''
                            # Append export code to the cell
                            new_source = source + "\n" + export_code
                            notebook['cells'][i+1]['source'] = new_source.split('\n')
                            
                            # Save the notebook
                            with open(notebook_path, 'w') as f:
                                json.dump(notebook, f, indent=1)
                            print("   ✅ Added export code to notebook")
                            found_export = True
                            break
    
    return found_export

def main():
    print("CHECKING REQUIRED EXPORTS FOR FINAL ANALYSIS")
    print("=" * 70)
    
    # Check both exports
    gemini_ok = check_gemini_baseline()
    openai_ok = check_openai_ic_hints_before()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if gemini_ok and openai_ok:
        print("✅ Both exports are configured correctly!")
        print("\nTo generate the missing data:")
        print("1. For Gemini baseline: Run sections 1-10 in notebooks-hintsafter/gemini_evaluation_ide.ipynb")
        print("2. For OpenAI IC hints BEFORE: Run sections 22-26 in notebooks-hintsbefore/openai_evaluation_ide_hintsbefore copy.ipynb")
    else:
        if not gemini_ok:
            print("❌ Gemini baseline export needs attention")
        if not openai_ok:
            print("❌ OpenAI IC hints BEFORE export needs attention")
        print("\nPlease review the notebooks and ensure exports are properly configured")

if __name__ == "__main__":
    main()