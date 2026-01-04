#!/usr/bin/env python3
"""
Add a final comprehensive comparison bar graph to the final analysis notebook
"""

import json

# Read the notebook
with open('final_analysis_notebook.ipynb', 'r') as f:
    notebook = json.load(f)

# Create the new graph cell content
new_graph_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 25. Comprehensive Model Comparison - All Conditions\n",
              "\n",
              "Final comparison showing both models across all experimental conditions with color-coded bars:\n",
              "- Gray: Baseline (no hints)\n",
              "- Green: Correct hints\n",
              "- Red: Incorrect hints"]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Comprehensive comparison of both models across all conditions\n",
        "fig, ax = plt.subplots(figsize=(16, 8))\n",
        "\n",
        "# Prepare data for both models\n",
        "conditions = ['Baseline\\n(No Hints)', 'Correct\\nHints After', 'Correct\\nHints Before', \n",
        "              'Incorrect\\nHints After', 'Incorrect\\nHints Before']\n",
        "\n",
        "# Calculate accuracies for Gemini\n",
        "gemini_accuracies = [\n",
        "    calculate_accuracy(gemini_baseline),          # Baseline\n",
        "    calculate_accuracy(gemini_hints_after),       # Correct After\n",
        "    calculate_accuracy(gemini_hints_before),      # Correct Before\n",
        "    calculate_accuracy(gemini_ic_hints_after),    # Incorrect After\n",
        "    calculate_accuracy(gemini_ic_hints_before)    # Incorrect Before\n",
        "]\n",
        "\n",
        "# Calculate accuracies for OpenAI\n",
        "openai_accuracies = [\n",
        "    calculate_accuracy(openai_baseline),          # Baseline\n",
        "    calculate_accuracy(openai_hints_after),       # Correct After\n",
        "    calculate_accuracy(openai_hints_before),      # Correct Before\n",
        "    calculate_accuracy(openai_ic_hints_after),    # Incorrect After\n",
        "    calculate_accuracy(openai_ic_hints_before)    # Incorrect Before\n",
        "]\n",
        "\n",
        "# Set up bar positions\n",
        "x = np.arange(len(conditions))\n",
        "width = 0.35\n",
        "\n",
        "# Define colors for each condition\n",
        "colors_gemini = ['#808080', '#43A047', '#66BB6A', '#EF5350', '#F44336']  # Gray, Green, Green, Red, Red\n",
        "colors_openai = ['#606060', '#2E7D32', '#4CAF50', '#D32F2F', '#C62828']  # Darker variants\n",
        "\n",
        "# Create bars for each model\n",
        "bars1 = ax.bar(x - width/2, gemini_accuracies, width, label='Gemini 2.0 Flash',\n",
        "               edgecolor='black', linewidth=1.5)\n",
        "bars2 = ax.bar(x + width/2, openai_accuracies, width, label='OpenAI GPT-4o-mini',\n",
        "               edgecolor='black', linewidth=1.5)\n",
        "\n",
        "# Apply colors to bars\n",
        "for bar, color in zip(bars1, colors_gemini):\n",
        "    bar.set_color(color)\n",
        "for bar, color in zip(bars2, colors_openai):\n",
        "    bar.set_color(color)\n",
        "\n",
        "# Add value labels on bars\n",
        "for bars in [bars1, bars2]:\n",
        "    for bar in bars:\n",
        "        height = bar.get_height()\n",
        "        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,\n",
        "                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
        "\n",
        "# Customize the plot\n",
        "ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')\n",
        "ax.set_xlabel('Experimental Condition', fontsize=13, fontweight='bold')\n",
        "ax.set_title('Comprehensive Model Comparison Across All Conditions', fontsize=16, fontweight='bold', pad=20)\n",
        "ax.set_xticks(x)\n",
        "ax.set_xticklabels(conditions, fontsize=11)\n",
        "ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)\n",
        "ax.grid(axis='y', alpha=0.3, linestyle='--')\n",
        "ax.set_ylim(0, 100)\n",
        "\n",
        "# Add horizontal line at 50% for reference\n",
        "ax.axhline(y=50, color='black', linestyle=':', alpha=0.5, linewidth=1, label='50% Reference')\n",
        "\n",
        "# Add a legend for colors\n",
        "from matplotlib.patches import Patch\n",
        "legend_elements = [\n",
        "    Patch(facecolor='#808080', edgecolor='black', label='Baseline (No Hints)'),\n",
        "    Patch(facecolor='#43A047', edgecolor='black', label='Correct Hints'),\n",
        "    Patch(facecolor='#EF5350', edgecolor='black', label='Incorrect Hints')\n",
        "]\n",
        "ax2_legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=11, \n",
        "                       title='Condition Types', title_fontsize=12)\n",
        "ax.add_artist(ax2_legend)  # Add second legend without removing first\n",
        "\n",
        "# Add model legend back\n",
        "ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Print summary statistics\n",
        "print('\\n' + '='*70)\n",
        "print('SUMMARY STATISTICS')\n",
        "print('='*70)\n",
        "\n",
        "print('\\n📊 Gemini 2.0 Flash:')\n",
        "print(f'  Baseline: {gemini_accuracies[0]:.1f}%')\n",
        "print(f'  Best Performance: {max(gemini_accuracies):.1f}% ({conditions[gemini_accuracies.index(max(gemini_accuracies))].replace(chr(10), \" \")})')\n",
        "print(f'  Worst Performance: {min(gemini_accuracies):.1f}% ({conditions[gemini_accuracies.index(min(gemini_accuracies))].replace(chr(10), \" \")})')\n",
        "print(f'  Average across all conditions: {np.mean(gemini_accuracies):.1f}%')\n",
        "\n",
        "print('\\n📊 OpenAI GPT-4o-mini:')\n",
        "print(f'  Baseline: {openai_accuracies[0]:.1f}%')\n",
        "print(f'  Best Performance: {max(openai_accuracies):.1f}% ({conditions[openai_accuracies.index(max(openai_accuracies))].replace(chr(10), \" \")})')\n",
        "print(f'  Worst Performance: {min(openai_accuracies):.1f}% ({conditions[openai_accuracies.index(min(openai_accuracies))].replace(chr(10), \" \")})')\n",
        "print(f'  Average across all conditions: {np.mean(openai_accuracies):.1f}%')\n",
        "\n",
        "print('\\n🔍 Key Insights:')\n",
        "baseline_diff = gemini_accuracies[0] - openai_accuracies[0]\n",
        "print(f'  • Baseline difference: Gemini leads by {baseline_diff:.1f}%')\n",
        "\n",
        "correct_impact_gemini = ((gemini_accuracies[1] + gemini_accuracies[2])/2 - gemini_accuracies[0])\n",
        "correct_impact_openai = ((openai_accuracies[1] + openai_accuracies[2])/2 - openai_accuracies[0])\n",
        "print(f'  • Correct hints impact: Gemini {correct_impact_gemini:+.1f}%, OpenAI {correct_impact_openai:+.1f}%')\n",
        "\n",
        "incorrect_impact_gemini = ((gemini_accuracies[3] + gemini_accuracies[4])/2 - gemini_accuracies[0])\n",
        "incorrect_impact_openai = ((openai_accuracies[3] + openai_accuracies[4])/2 - openai_accuracies[0])\n",
        "print(f'  • Incorrect hints impact: Gemini {incorrect_impact_gemini:+.1f}%, OpenAI {incorrect_impact_openai:+.1f}%')\n",
        "\n",
        "before_vs_after_correct_gemini = gemini_accuracies[2] - gemini_accuracies[1]\n",
        "before_vs_after_correct_openai = openai_accuracies[2] - openai_accuracies[1]\n",
        "print(f'  • Hints Before vs After (Correct): Gemini {before_vs_after_correct_gemini:+.1f}%, OpenAI {before_vs_after_correct_openai:+.1f}%')\n",
        "\n",
        "before_vs_after_incorrect_gemini = gemini_accuracies[4] - gemini_accuracies[3]\n",
        "before_vs_after_incorrect_openai = openai_accuracies[4] - openai_accuracies[3]\n",
        "print(f'  • Hints Before vs After (Incorrect): Gemini {before_vs_after_incorrect_gemini:+.1f}%, OpenAI {before_vs_after_incorrect_openai:+.1f}%')\n"
    ]
}

# Find where to insert the new cells (after the final heatmap, before conclusion)
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        if 'Conclusion' in source:
            insert_index = i
            break

if insert_index:
    # Insert the new cells before the conclusion
    notebook['cells'].insert(insert_index, new_code_cell)
    notebook['cells'].insert(insert_index, new_graph_cell)
    print(f"✅ Inserted new comprehensive comparison graph at position {insert_index}")
else:
    # Append at the end if conclusion not found
    notebook['cells'].append(new_graph_cell)
    notebook['cells'].append(new_code_cell)
    print("✅ Appended new comprehensive comparison graph at the end")

# Save the updated notebook
with open('final_analysis_notebook.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\n✅ Final analysis notebook updated with comprehensive comparison graph!")
print("\nThe new graph shows:")
print("  • Both models side by side")
print("  • All 5 conditions for each model")
print("  • Color coding: Gray (baseline), Green (correct hints), Red (incorrect hints)")
print("  • Accuracy percentages on each bar")
print("  • Detailed summary statistics below the graph")
print("\n🎉 The notebook now has a complete visual comparison of all experimental conditions!")