#!/usr/bin/env python3
"""
Verify all datasets are available for final analysis
"""

import pandas as pd
from pathlib import Path

results_dir = Path('results')

def load_latest_results(pattern):
    files = list(results_dir.glob(pattern))
    if not files:
        return None
    latest_file = max(files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest_file), latest_file.name

print("=" * 70)
print("VERIFYING ALL DATASETS FOR FINAL ANALYSIS")
print("=" * 70)

datasets = {
    "Gemini Baseline (NO hints)": ['gemini_evaluation_results_*_baseline.csv', 'gemini_evaluation_results_*.csv'],
    "Gemini Correct Hints AFTER": ['gemini_hints_evaluation_results_*.csv'],
    "Gemini Correct Hints BEFORE": ['gemini_hints_BEFORE_evaluation_*.csv'],
    "Gemini Incorrect Hints AFTER": ['gemini_ic_hints_evaluation_results_*.csv'],
    "Gemini Incorrect Hints BEFORE": ['gemini_ic_hints_BEFORE_evaluation_*.csv'],
    "OpenAI Baseline (NO hints)": ['openai_evaluation_results_*.csv'],
    "OpenAI Correct Hints AFTER": ['openai_hints_evaluation_results_*.csv'],
    "OpenAI Correct Hints BEFORE": ['openai_hints_BEFORE_evaluation_*.csv'],
    "OpenAI Incorrect Hints AFTER": ['openai_ic_hints_evaluation_results_*.csv'],
    "OpenAI Incorrect Hints BEFORE": ['openai_ic_hints_BEFORE_evaluation_*.csv']
}

available = 0
missing = []

for name, patterns in datasets.items():
    found = False
    for pattern in patterns:
        result = load_latest_results(pattern)
        if result:
            df, filename = result
            print(f"✅ {name:40} → {filename} ({len(df)} records)")
            available += 1
            found = True
            break
    if not found:
        print(f"❌ {name:40} → NOT FOUND")
        missing.append(name)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Available: {available}/10 datasets")

if available == 10:
    print("\n🎉 SUCCESS! All 10 datasets are available!")
    print("You can now run the final analysis notebook for complete results!")
else:
    print(f"\n⚠️ Missing {10-available} dataset(s):")
    for m in missing:
        print(f"  - {m}")

# Check accuracy for available datasets
print("\n" + "=" * 70)
print("QUICK ACCURACY CHECK")
print("=" * 70)

for name, patterns in datasets.items():
    for pattern in patterns:
        result = load_latest_results(pattern)
        if result:
            df, filename = result
            if 'is_correct' in df.columns:
                accuracy = df['is_correct'].mean() * 100
                print(f"{name[:30]:30} : {accuracy:5.1f}% accuracy")
            break