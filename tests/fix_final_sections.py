#!/usr/bin/env python3
"""
Fix sections 27-31 for comprehensive comparisons
"""

import json

# Read the notebook
with open('gemini_evaluation_ide_hintsbefore.ipynb', 'r') as f:
    notebook = json.load(f)

# Fix Section 27 - Three-Way Comparison
section_27_code = """# Comprehensive comparison: No Hints vs Correct Hints Before vs Incorrect Hints Before
import glob
import os

print("="*70)
print("📊 COMPREHENSIVE COMPARISON - ALL CONDITIONS")
print("="*70)

# Try to load baseline and hints-after results for full comparison
baseline_files = glob.glob("../results/gemini_evaluation_results_*.csv")
hints_after_files = glob.glob("../results/gemini_hints_evaluation_results_*.csv")
ic_hints_after_files = glob.glob("../results/gemini_ic_hints_evaluation_results_*.csv")

comparison_data = []

# 1. BASELINE (if available)
if baseline_files:
    latest_baseline = max(baseline_files, key=os.path.getctime)
    baseline_df = pd.read_csv(latest_baseline)
    baseline_metrics = calculate_metrics(baseline_df)
    comparison_data.append({
        'Condition': 'Baseline (No Hints)',
        'Overall (%)': baseline_metrics['overall_accuracy'],
        'Math (%)': baseline_metrics['by_domain']['math']['accuracy'],
        'Science (%)': baseline_metrics['by_domain']['science']['accuracy'],
        'Hint Position': 'N/A',
        'Hint Type': 'None'
    })
    print(f"✅ Loaded baseline from: {os.path.basename(latest_baseline)}")

# 2. CORRECT HINTS BEFORE (current notebook)
comparison_data.append({
    'Condition': 'Correct Hints BEFORE',
    'Overall (%)': hints_metrics['overall_accuracy'],
    'Math (%)': hints_metrics['by_domain']['math']['accuracy'],
    'Science (%)': hints_metrics['by_domain']['science']['accuracy'],
    'Hint Position': 'BEFORE',
    'Hint Type': 'Correct'
})

# 3. INCORRECT HINTS BEFORE (current notebook)
comparison_data.append({
    'Condition': 'Incorrect Hints BEFORE',
    'Overall (%)': ic_metrics['overall_accuracy'],
    'Math (%)': ic_metrics['by_domain']['math']['accuracy'],
    'Science (%)': ic_metrics['by_domain']['science']['accuracy'],
    'Hint Position': 'BEFORE',
    'Hint Type': 'Incorrect'
})

# 4. CORRECT HINTS AFTER (if available)
if hints_after_files:
    latest_hints_after = max(hints_after_files, key=os.path.getctime)
    hints_after_df = pd.read_csv(latest_hints_after)
    hints_after_metrics = calculate_metrics(hints_after_df)
    comparison_data.append({
        'Condition': 'Correct Hints AFTER',
        'Overall (%)': hints_after_metrics['overall_accuracy'],
        'Math (%)': hints_after_metrics['by_domain']['math']['accuracy'],
        'Science (%)': hints_after_metrics['by_domain']['science']['accuracy'],
        'Hint Position': 'AFTER',
        'Hint Type': 'Correct'
    })
    print(f"✅ Loaded correct hints-after from: {os.path.basename(latest_hints_after)}")

# 5. INCORRECT HINTS AFTER (if available)
if ic_hints_after_files:
    latest_ic_after = max(ic_hints_after_files, key=os.path.getctime)
    ic_after_df = pd.read_csv(latest_ic_after)
    ic_after_metrics = calculate_metrics(ic_after_df)
    comparison_data.append({
        'Condition': 'Incorrect Hints AFTER',
        'Overall (%)': ic_after_metrics['overall_accuracy'],
        'Math (%)': ic_after_metrics['by_domain']['math']['accuracy'],
        'Science (%)': ic_after_metrics['by_domain']['science']['accuracy'],
        'Hint Position': 'AFTER',
        'Hint Type': 'Incorrect'
    })
    print(f"✅ Loaded incorrect hints-after from: {os.path.basename(latest_ic_after)}")

# Create comprehensive DataFrame
full_comparison_df = pd.DataFrame(comparison_data)
full_comparison_df = full_comparison_df.round(2)

print("\\n📊 FULL COMPARISON TABLE:")
display(full_comparison_df)

# Analysis
print("\\n🎯 KEY FINDINGS:")
print("-"*50)

# Find best approach
best_row = full_comparison_df.loc[full_comparison_df['Overall (%)'].idxmax()]
print(f"✅ Best Performance: {best_row['Condition']} ({best_row['Overall (%)']:.2f}%)")

# Analyze hint positioning impact
before_correct = full_comparison_df[full_comparison_df['Condition'] == 'Correct Hints BEFORE']['Overall (%)'].values
after_correct = full_comparison_df[full_comparison_df['Condition'] == 'Correct Hints AFTER']['Overall (%)'].values

if len(before_correct) > 0 and len(after_correct) > 0:
    position_impact = before_correct[0] - after_correct[0]
    print(f"\\n📌 Hint Position Impact (Correct Hints):")
    print(f"   BEFORE: {before_correct[0]:.2f}%")
    print(f"   AFTER: {after_correct[0]:.2f}%")
    print(f"   Difference: {position_impact:+.2f}pp")
    
    if abs(position_impact) > 2:
        better_position = "BEFORE" if position_impact > 0 else "AFTER"
        print(f"   → Hints work better {better_position} questions")

# Analyze robustness to misinformation
before_incorrect = full_comparison_df[full_comparison_df['Condition'] == 'Incorrect Hints BEFORE']['Overall (%)'].values
after_incorrect = full_comparison_df[full_comparison_df['Condition'] == 'Incorrect Hints AFTER']['Overall (%)'].values

if len(before_incorrect) > 0 and len(after_incorrect) > 0:
    robustness_diff = before_incorrect[0] - after_incorrect[0]
    print(f"\\n🛡️ Robustness to Misinformation:")
    print(f"   Incorrect BEFORE: {before_incorrect[0]:.2f}%")
    print(f"   Incorrect AFTER: {after_incorrect[0]:.2f}%")
    print(f"   Difference: {robustness_diff:+.2f}pp")
    
    if abs(robustness_diff) > 2:
        more_robust = "BEFORE" if robustness_diff > 0 else "AFTER"
        print(f"   → Model more robust when incorrect hints {more_robust}")
"""

# Fix Section 28 - Visualizations
section_28_code = """# Create comprehensive visualizations comparing all conditions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Gemini Performance: Comprehensive Hints Analysis", fontsize=16, fontweight='bold')

# 1. Overall Comparison Bar Chart
ax = axes[0, 0]
conditions = full_comparison_df['Condition'].tolist()
accuracies = full_comparison_df['Overall (%)'].tolist()
colors = ['gray', 'green', 'red', 'lightgreen', 'lightcoral'][:len(conditions)]
bars = ax.bar(range(len(conditions)), accuracies, color=colors)
ax.set_xticks(range(len(conditions)))
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Overall Accuracy by Condition')
ax.set_ylim(0, 100)
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{acc:.1f}%', ha='center', fontweight='bold')

# 2. Math vs Science Comparison
ax = axes[0, 1]
width = 0.35
x = np.arange(len(conditions))
math_accs = full_comparison_df['Math (%)'].tolist()
sci_accs = full_comparison_df['Science (%)'].tolist()
ax.bar(x - width/2, math_accs, width, label='Math', color='steelblue')
ax.bar(x + width/2, sci_accs, width, label='Science', color='coral')
ax.set_xticks(x)
ax.set_xticklabels([c.replace(' ', '\\n') for c in conditions], fontsize=9)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Performance by Domain')
ax.legend()
ax.set_ylim(0, 100)

# 3. Hint Position Impact
ax = axes[0, 2]
if 'Correct Hints BEFORE' in conditions and 'Correct Hints AFTER' in conditions:
    position_data = full_comparison_df[full_comparison_df['Hint Type'] == 'Correct']
    x_pos = ['BEFORE', 'AFTER']
    y_pos = position_data['Overall (%)'].tolist()
    ax.bar(x_pos, y_pos, color=['darkgreen', 'lightgreen'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Correct Hints: Position Impact')
    ax.set_ylim(0, 100)
    for i, v in enumerate(y_pos):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'Hints-after data\\nnot available', 
            ha='center', va='center', transform=ax.transAxes)

# 4. Misinformation Impact
ax = axes[1, 0]
if 'Incorrect Hints BEFORE' in conditions and 'Incorrect Hints AFTER' in conditions:
    misinfo_data = full_comparison_df[full_comparison_df['Hint Type'] == 'Incorrect']
    x_mis = ['BEFORE', 'AFTER']
    y_mis = misinfo_data['Overall (%)'].tolist()
    ax.bar(x_mis, y_mis, color=['darkred', 'lightcoral'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Incorrect Hints: Position Impact')
    ax.set_ylim(0, 100)
    for i, v in enumerate(y_mis):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
else:
    ax.text(0.5, 0.5, 'IC hints-after\\ndata not available', 
            ha='center', va='center', transform=ax.transAxes)

# 5. Heatmap of all conditions
ax = axes[1, 1]
# Create matrix for heatmap
if len(full_comparison_df) >= 3:
    heatmap_data = []
    for _, row in full_comparison_df.iterrows():
        heatmap_data.append([row['Overall (%)'], row['Math (%)'], row['Science (%)']])
    
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(['Overall', 'Math', 'Science'])
    ax.set_yticklabels(conditions)
    ax.set_title('Performance Heatmap')
    
    # Add text annotations
    for i in range(len(conditions)):
        for j in range(3):
            text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}',
                         ha='center', va='center', color='black')
    
    plt.colorbar(im, ax=ax, label='Accuracy (%)')

# 6. Summary Statistics
ax = axes[1, 2]
ax.axis('off')
summary_text = f"📊 COMPREHENSIVE SUMMARY\\n\\n"
summary_text += f"Conditions Tested: {len(full_comparison_df)}\\n\\n"

# Best and worst
best_idx = full_comparison_df['Overall (%)'].idxmax()
worst_idx = full_comparison_df['Overall (%)'].idxmin()
summary_text += f"✅ Best: {full_comparison_df.loc[best_idx, 'Condition']}\\n"
summary_text += f"   ({full_comparison_df.loc[best_idx, 'Overall (%)']:.1f}%)\\n\\n"
summary_text += f"❌ Worst: {full_comparison_df.loc[worst_idx, 'Condition']}\\n"
summary_text += f"   ({full_comparison_df.loc[worst_idx, 'Overall (%)']:.1f}%)\\n\\n"

# Range
acc_range = full_comparison_df['Overall (%)'].max() - full_comparison_df['Overall (%)'].min()
summary_text += f"📏 Range: {acc_range:.1f}pp\\n\\n"

summary_text += "🎯 Key Finding:\\n"
if 'Correct Hints BEFORE' in conditions:
    before_acc = full_comparison_df[full_comparison_df['Condition'] == 'Correct Hints BEFORE']['Overall (%)'].values[0]
    summary_text += f"Hints BEFORE: {before_acc:.1f}%"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
        fontsize=11, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\\n✅ Comprehensive visualizations complete!")
"""

# Fix Section 30 - Export
section_30_code = """# Export comprehensive comparison results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save comprehensive comparison
comp_file = f"../results/gemini_comprehensive_hints_comparison_{timestamp}.csv"
full_comparison_df.to_csv(comp_file, index=False)
print(f"✅ Comprehensive comparison saved to: {comp_file}")

# Save detailed IC-hints results
ic_file = f"../results/gemini_ic_hints_BEFORE_evaluation_{timestamp}.csv"
all_ic_results_df.to_csv(ic_file, index=False)
print(f"✅ IC-hints BEFORE results saved to: {ic_file}")

# Create comprehensive report
report_file = f"../results/gemini_hints_position_analysis_{timestamp}.txt"
with open(report_file, 'w') as f:
    f.write("GEMINI MODEL - COMPREHENSIVE HINTS ANALYSIS\\n")
    f.write("="*60 + "\\n\\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
    f.write(f"Model: Gemini 2.0 Flash\\n\\n")
    
    f.write("CONDITIONS TESTED\\n")
    f.write("-"*40 + "\\n")
    for _, row in full_comparison_df.iterrows():
        f.write(f"{row['Condition']}: {row['Overall (%)']:.2f}%\\n")
    
    f.write("\\nKEY FINDINGS\\n")
    f.write("-"*40 + "\\n")
    
    # Best configuration
    best_idx = full_comparison_df['Overall (%)'].idxmax()
    f.write(f"Best Performance: {full_comparison_df.loc[best_idx, 'Condition']}\\n")
    f.write(f"Accuracy: {full_comparison_df.loc[best_idx, 'Overall (%)']:.2f}%\\n\\n")
    
    # Hint positioning impact
    if 'Correct Hints BEFORE' in full_comparison_df['Condition'].values and \
       'Correct Hints AFTER' in full_comparison_df['Condition'].values:
        before_acc = full_comparison_df[full_comparison_df['Condition'] == 'Correct Hints BEFORE']['Overall (%)'].values[0]
        after_acc = full_comparison_df[full_comparison_df['Condition'] == 'Correct Hints AFTER']['Overall (%)'].values[0]
        f.write("HINT POSITIONING IMPACT\\n")
        f.write(f"Correct Hints BEFORE: {before_acc:.2f}%\\n")
        f.write(f"Correct Hints AFTER: {after_acc:.2f}%\\n")
        f.write(f"Difference: {before_acc - after_acc:+.2f}pp\\n\\n")
    
    f.write("RECOMMENDATIONS\\n")
    f.write("-"*40 + "\\n")
    f.write("Based on the comprehensive analysis:\\n")
    f.write(f"• Use {full_comparison_df.loc[best_idx, 'Condition']} for best performance\\n")
    if before_acc > after_acc + 2:
        f.write("• Present hints BEFORE questions for this model\\n")
    elif after_acc > before_acc + 2:
        f.write("• Present hints AFTER questions for this model\\n")
    else:
        f.write("• Hint position has minimal impact on this model\\n")

print(f"✅ Comprehensive report saved to: {report_file}")
print("\\n📊 All results exported with clear labeling!")
"""

# Apply fixes
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell.get('source', []))
        
        # Fix Section 27
        if '## 27. Three-Way Comparative Analysis' in source:
            cell['source'] = ['## 27. Comprehensive Comparison: All Conditions\n',
                            '\n',
                            'Compare baseline, correct hints (before/after), and incorrect hints (before/after).']
            if i+1 < len(notebook['cells']):
                notebook['cells'][i+1]['source'] = section_27_code.split('\n')
            print("Fixed Section 27")
            
        # Fix Section 28
        elif '## 28. Visualizations - Three-Way Comparison' in source:
            cell['source'] = ['## 28. Visualizations - Comprehensive Comparison\n',
                            '\n',
                            'Visualize all conditions including hint positioning effects.']
            if i+1 < len(notebook['cells']):
                notebook['cells'][i+1]['source'] = section_28_code.split('\n')
            print("Fixed Section 28")
            
        # Fix Section 30
        elif '## 30. Export Three-Way Comparison Results' in source:
            cell['source'] = ['## 30. Export Comprehensive Results\n',
                            '\n',
                            'Save all comparison results with clear labeling.']
            if i+1 < len(notebook['cells']):
                notebook['cells'][i+1]['source'] = section_30_code.split('\n')
            print("Fixed Section 30")

# Save
with open('gemini_evaluation_ide_hintsbefore.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\nAll sections 27-31 fixed successfully!")