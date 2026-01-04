#!/usr/bin/env python3
"""
Fix sections 23-31 for IC-hints (incorrect hints) in the notebook
"""

import json

# Read the notebook
with open('gemini_evaluation_ide_hintsbefore.ipynb', 'r') as f:
    notebook = json.load(f)

# Dictionary of sections to fix
sections_to_fix = {
    "## 23. Test Query Function with Alternative Hints": {
        "markdown": ["## 23. Test Query with INCORRECT Hints (Before Questions)\n",
                    "\n",
                    "Test the query function with misleading hints presented BEFORE questions."],
        "code": [
            "# Test with INCORRECT hint presented BEFORE question\n",
            "if ic_math_questions:\n",
            "    sample_ic = ic_math_questions[0]\n",
            "    print(\"🧪 Testing INCORRECT hints BEFORE questions approach:\")\n",
            "    print(f\"1️⃣ INCORRECT HINT (shown first): {sample_ic['hint']}\")\n",
            "    print(f\"2️⃣ QUESTION (shown second): {sample_ic['question'][:100]}...\")\n",
            "    \n",
            "    print(\"\\nQuerying Gemini with MISLEADING hint BEFORE question...\")\n",
            "    ic_response = query_gemini_with_hint(\n",
            "        sample_ic['question'],\n",
            "        sample_ic['hint'],  # Incorrect hint shown BEFORE question\n",
            "        'math'\n",
            "    )\n",
            "    \n",
            "    if ic_response['error']:\n",
            "        print(f\"Error: {ic_response['error']}\")\n",
            "    else:\n",
            "        print(f\"Model's Answer (after incorrect hint): {ic_response['answer']}\")\n",
            "        print(f\"Correct Answer: {ic_math_ground_truth[sample_ic['id']]}\")\n",
            "        print(f\"Response Time: {ic_response['response_time']:.2f}s\")\n",
            "        print(\"\\n⚠️ Note: The model saw an INCORRECT hint BEFORE the question\")\n"
        ]
    },
    "## 24. Process All Questions WITH Alternative Hints": {
        "markdown": ["## 24. Process Questions with INCORRECT Hints (Before Questions)\n",
                    "\n",
                    "Process all questions with misleading hints presented BEFORE the questions."],
        "code": [
            "# Process questions with INCORRECT hints BEFORE questions\n",
            "def process_questions_with_ic_hints(questions: List[Dict], question_type: str) -> List[Dict]:\n",
            "    \"\"\"Process questions with INCORRECT hints presented BEFORE questions\"\"\"\n",
            "    results = []\n",
            "    total = len(questions)\n",
            "    \n",
            "    print(f\"⚠️ Processing {total} {question_type} questions with INCORRECT hints...\")\n",
            "    print(f\"📌 Misleading hints will be shown BEFORE questions\")\n",
            "    print(\"=\"*50)\n",
            "    \n",
            "    for i, q in enumerate(questions, 1):\n",
            "        # Pass INCORRECT hint BEFORE question\n",
            "        response = query_gemini_with_hint(\n",
            "            question_text=q['question'],\n",
            "            hint_text=q.get('hint', ''),  # INCORRECT hint\n",
            "            question_type=question_type\n",
            "        )\n",
            "        \n",
            "        result = {\n",
            "            'id': q['id'],\n",
            "            'question': q['question'],\n",
            "            'hint': q.get('hint', ''),  # The INCORRECT hint shown before question\n",
            "            'hint_type': 'INCORRECT',\n",
            "            'hint_position': 'BEFORE',\n",
            "            'difficulty': q['difficulty'],\n",
            "            'gemini_answer': response['answer'],\n",
            "            'response_time': response['response_time'],\n",
            "            'error': response['error']\n",
            "        }\n",
            "        \n",
            "        if 'subject' in q:\n",
            "            result['subject'] = q['subject']\n",
            "        \n",
            "        results.append(result)\n",
            "        \n",
            "        if i % 10 == 0 or i == total:\n",
            "            print(f\"Progress: {i}/{total} - Incorrect hints before questions\")\n",
            "        \n",
            "        time.sleep(0.5)  # Rate limiting\n",
            "    \n",
            "    print(f\"\\n✅ Completed {total} questions with INCORRECT hints BEFORE\")\n",
            "    return results\n",
            "\n",
            "# Process with INCORRECT hints BEFORE questions\n",
            "print(\"🔢 MATH - INCORRECT Hints Before Questions\")\n",
            "ic_math_results = process_questions_with_ic_hints(ic_math_questions, 'math')\n",
            "\n",
            "print(\"\\n🔬 SCIENCE - INCORRECT Hints Before Questions\")\n",
            "ic_science_results = process_questions_with_ic_hints(ic_science_questions, 'science')\n",
            "\n",
            "print(\"\\n✅ All questions processed with INCORRECT hints BEFORE questions\")\n"
        ]
    },
    "## 25. Evaluate Results WITH Incorrect Hints Against Ground Truth": {
        "markdown": ["## 25. Evaluate INCORRECT Hints Results Against Ground Truth\n",
                    "\n",
                    "Evaluate model performance when misleading hints are shown BEFORE questions."],
        "code": [
            "# Evaluate results with INCORRECT hints shown BEFORE questions\n",
            "\n",
            "# Evaluate math with incorrect hints\n",
            "ic_math_df = evaluate_results_with_hints(ic_math_results, ic_math_ground_truth, 'math')\n",
            "print(\"📊 Math Results with INCORRECT Hints BEFORE Questions:\")\n",
            "print(f\"Correct: {ic_math_df['is_correct'].sum()}/{len(ic_math_df)}\")\n",
            "print(f\"Accuracy: {100 * ic_math_df['is_correct'].mean():.2f}%\")\n",
            "print(f\"Impact: Model saw MISLEADING hints BEFORE questions\\n\")\n",
            "\n",
            "# Evaluate science with incorrect hints\n",
            "ic_science_df = evaluate_results_with_hints(ic_science_results, ic_science_ground_truth, 'science')\n",
            "print(\"🔬 Science Results with INCORRECT Hints BEFORE Questions:\")\n",
            "print(f\"Correct: {ic_science_df['is_correct'].sum()}/{len(ic_science_df)}\")\n",
            "print(f\"Accuracy: {100 * ic_science_df['is_correct'].mean():.2f}%\")\n",
            "\n",
            "# Combine results\n",
            "ic_math_df['domain'] = 'math'\n",
            "ic_science_df['domain'] = 'science'\n",
            "all_ic_results_df = pd.concat([ic_math_df, ic_science_df], ignore_index=True)\n",
            "\n",
            "print(\"\\n⚠️ OVERALL with INCORRECT Hints BEFORE Questions:\")\n",
            "print(f\"Total Correct: {all_ic_results_df['is_correct'].sum()}/{len(all_ic_results_df)}\")\n",
            "print(f\"Overall Accuracy: {100 * all_ic_results_df['is_correct'].mean():.2f}%\")\n",
            "\n",
            "# Calculate how much incorrect hints hurt performance\n",
            "ic_metrics = calculate_metrics(all_ic_results_df)\n",
            "misleading_impact = ic_metrics['overall_accuracy'] - hints_metrics['overall_accuracy']\n",
            "print(f\"\\n📉 Impact of INCORRECT vs CORRECT hints (both before): {misleading_impact:+.2f}pp\")\n",
            "if misleading_impact < 0:\n",
            "    print(\"   Incorrect hints reduce performance as expected\")\n"
        ]
    },
    "## 26. Detailed Analysis - Performance WITH Alternative Hints": {
        "markdown": ["## 26. Analysis - INCORRECT Hints Before Questions\n",
                    "\n",
                    "Detailed analysis when misleading hints are presented first."],
        "code": [
            "# Detailed analysis for INCORRECT hints BEFORE questions\n",
            "print(\"=\"*60)\n",
            "print(\"📊 DETAILED ANALYSIS - INCORRECT HINTS BEFORE QUESTIONS\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "print(f\"\\n⚠️ Approach: MISLEADING hints presented BEFORE questions\")\n",
            "print(f\"Total Questions: {ic_metrics['total_questions']}\")\n",
            "print(f\"Correct Despite Misleading Hints: {ic_metrics['correct_answers']}\")\n",
            "print(f\"Accuracy with Incorrect Hints: {ic_metrics['overall_accuracy']:.2f}%\")\n",
            "\n",
            "# Compare with correct hints\n",
            "print(\"\\n📊 Comparison with CORRECT Hints (both before):\")\n",
            "print(\"-\"*40)\n",
            "print(f\"Correct Hints Before: {hints_metrics['overall_accuracy']:.2f}%\")\n",
            "print(f\"Incorrect Hints Before: {ic_metrics['overall_accuracy']:.2f}%\")\n",
            "print(f\"Difference: {ic_metrics['overall_accuracy'] - hints_metrics['overall_accuracy']:+.2f}pp\")\n",
            "\n",
            "# By domain\n",
            "print(\"\\n📈 Impact by Domain (Incorrect Hints Before):\")\n",
            "for domain in ic_metrics['by_domain']:\n",
            "    ic_acc = ic_metrics['by_domain'][domain]['accuracy']\n",
            "    correct_acc = hints_metrics['by_domain'][domain]['accuracy']\n",
            "    print(f\"{domain.upper()}:\")\n",
            "    print(f\"  With Incorrect Hints: {ic_acc:.2f}%\")\n",
            "    print(f\"  With Correct Hints: {correct_acc:.2f}%\")\n",
            "    print(f\"  Impact of Misinformation: {ic_acc - correct_acc:+.2f}pp\")\n",
            "\n",
            "# By difficulty\n",
            "print(\"\\n📊 Impact by Difficulty (Incorrect Hints Before):\")\n",
            "for difficulty in ic_metrics['by_difficulty']:\n",
            "    ic_acc = ic_metrics['by_difficulty'][difficulty]['accuracy']\n",
            "    correct_acc = hints_metrics['by_difficulty'][difficulty]['accuracy']\n",
            "    print(f\"{difficulty.upper()}:\")\n",
            "    print(f\"  With Incorrect Hints: {ic_acc:.2f}%\")\n",
            "    print(f\"  Impact: {ic_acc - correct_acc:+.2f}pp vs correct hints\")\n"
        ]
    }
}

# Apply fixes to the notebook
for section_title, fixes in sections_to_fix.items():
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown' and 'source' in cell:
            source = ''.join(cell.get('source', []))
            if section_title in source:
                # Update markdown
                cell['source'] = fixes['markdown']
                
                # Update following code cell
                if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                    notebook['cells'][i+1]['source'] = fixes['code']
                
                print(f"Fixed: {section_title}")
                break

# Save the notebook
with open('gemini_evaluation_ide_hintsbefore.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\nAll IC-hints sections fixed successfully!")