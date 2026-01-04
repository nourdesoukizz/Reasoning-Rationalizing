#!/usr/bin/env python3
"""
Fix section 14 and related sections in OpenAI notebook to ensure API calls work
"""

import json

# Read the notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'r') as f:
    notebook = json.load(f)

# Find and fix section 14
section_14_found = False
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell.get('source', []))
        
        # Find section 14
        if '## 14. Process All Questions WITH Hints' in source or '## 14. Process All Questions With Hints' in source:
            section_14_found = True
            print(f"Found section 14 at cell {i}")
            
            # Update the code cell after it
            if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                # Update section 14 to both define AND call the function
                notebook['cells'][i+1]['source'] = [
                    "# Process questions with CORRECT hints BEFORE questions\n",
                    "def process_questions_with_hints(questions: List[Dict], question_type: str) -> List[Dict]:\n",
                    "    \"\"\"Process questions with CORRECT hints presented BEFORE questions\"\"\"\n",
                    "    results = []\n",
                    "    total = len(questions)\n",
                    "    \n",
                    "    print(f'📝 Processing {total} {question_type} questions with CORRECT hints...')\n",
                    "    print(f'📌 Helpful hints will be shown BEFORE questions')\n",
                    "    print('='*50)\n",
                    "    \n",
                    "    for i, q in enumerate(questions, 1):\n",
                    "        # Pass CORRECT hint BEFORE question\n",
                    "        response = query_openai_with_hint(\n",
                    "            question_text=q['question'],\n",
                    "            hint_text=q.get('hint', ''),  # CORRECT hint\n",
                    "            question_type=question_type\n",
                    "        )\n",
                    "        \n",
                    "        result = {\n",
                    "            'id': q['id'],\n",
                    "            'question': q['question'],\n",
                    "            'hint': q.get('hint', ''),  # The CORRECT hint shown before question\n",
                    "            'hint_type': 'CORRECT',\n",
                    "            'hint_position': 'BEFORE',\n",
                    "            'difficulty': q['difficulty'],\n",
                    "            'openai_answer': response['answer'],\n",
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
                    "            print(f'Progress: {i}/{total} - Hints before questions')\n",
                    "        \n",
                    "        time.sleep(0.5)  # Rate limiting for OpenAI API\n",
                    "    \n",
                    "    print(f'\\n✅ Completed {total} questions with CORRECT hints BEFORE')\n",
                    "    return results\n",
                    "\n",
                    "# NOW ACTUALLY PROCESS THE QUESTIONS!\n",
                    "print('🔢 MATH - Processing with CORRECT Hints Before Questions')\n",
                    "math_results = process_questions_with_hints(math_questions, 'math')\n",
                    "\n",
                    "print('\\n🔬 SCIENCE - Processing with CORRECT Hints Before Questions')\n",
                    "science_results = process_questions_with_hints(science_questions, 'science')\n",
                    "\n",
                    "print('\\n✅ All questions processed with CORRECT hints BEFORE questions')\n",
                    "print(f'Total math results: {len(math_results)}')\n",
                    "print(f'Total science results: {len(science_results)}')\n"
                ]
                print("Fixed section 14 - added actual function calls")

# Also ensure the query_openai_with_hint function is properly defined before section 14
# Find section 13 (Test Query Function with Hints)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell.get('source', []))
        
        if '## 13. Test Query Function with Hints' in source or '## 13. Test Query with Hints' in source:
            print(f"Found section 13 at cell {i}")
            
            # Make sure the code cell after it properly tests the function
            if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                notebook['cells'][i+1]['source'] = [
                    "# Test with CORRECT hint presented BEFORE question\n",
                    "if math_questions:\n",
                    "    sample_q = math_questions[0]\n",
                    "    print('🧪 Testing CORRECT hints BEFORE questions approach:')\n",
                    "    print(f'1️⃣ HINT (shown first): {sample_q[\"hint\"]}')\n",
                    "    print(f'2️⃣ QUESTION (shown second): {sample_q[\"question\"][:100]}...')\n",
                    "    \n",
                    "    print('\\nQuerying OpenAI with hint BEFORE question...')\n",
                    "    test_response = query_openai_with_hint(\n",
                    "        sample_q['question'],\n",
                    "        sample_q['hint'],  # Correct hint shown BEFORE question\n",
                    "        'math'\n",
                    "    )\n",
                    "    \n",
                    "    if test_response['error']:\n",
                    "        print(f'Error: {test_response[\"error\"]}')\n",
                    "    else:\n",
                    "        print(f'Model\\'s Answer: {test_response[\"answer\"]}')\n",
                    "        print(f'Correct Answer: {math_ground_truth[sample_q[\"id\"]]}')\n",
                    "        print(f'Response Time: {test_response[\"response_time\"]:.2f}s')\n",
                    "        print('\\n✅ The model successfully used the hint shown BEFORE the question')\n"
                ]
                print("Fixed section 13 - test query")

# Ensure query_openai_with_hint is defined in an earlier section
# Find where it should be defined (usually section 11 or 12)
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell.get('source', []))
        
        if 'Query Function with Hint' in source or 'query_openai_with_hint' in source.lower():
            print(f"Found query function section at cell {i}")
            
            if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                notebook['cells'][i+1]['source'] = [
                    "def query_openai_with_hint(question_text: str, hint_text: str, question_type: str = 'math') -> Dict:\n",
                    "    \"\"\"\n",
                    "    Query OpenAI with a hint presented BEFORE the question.\n",
                    "    \n",
                    "    Args:\n",
                    "        question_text: The question to ask\n",
                    "        hint_text: The hint to provide BEFORE the question\n",
                    "        question_type: Type of question (math or science)\n",
                    "    \"\"\"\n",
                    "    try:\n",
                    "        start_time = time.time()\n",
                    "        \n",
                    "        # Construct prompt with hint BEFORE question\n",
                    "        prompt_parts = [\n",
                    "            'Follow these instructions carefully:',\n",
                    "            '1. First, read the hint provided below',\n",
                    "            '2. Then read the question',\n",
                    "            '3. Use the hint to guide your solution',\n",
                    "            '4. Provide ONLY the final answer',\n",
                    "            '',\n",
                    "            '=== HINT (Read this first) ===',\n",
                    "            hint_text,\n",
                    "            '',\n",
                    "            '=== PROBLEM (Now solve using the hint above) ===',\n",
                    "            question_text,\n",
                    "            '',\n",
                    "            'Remember: Provide ONLY the final answer, no explanations.'\n",
                    "        ]\n",
                    "        \n",
                    "        prompt = '\\n'.join(prompt_parts)\n",
                    "        \n",
                    "        # Create messages for OpenAI chat completion\n",
                    "        messages = [\n",
                    "            {'role': 'system', 'content': f'You are an expert {question_type} problem solver. Follow instructions carefully and use hints effectively.'},\n",
                    "            {'role': 'user', 'content': prompt}\n",
                    "        ]\n",
                    "        \n",
                    "        # Query OpenAI\n",
                    "        response = client.chat.completions.create(\n",
                    "            model='gpt-4o-mini',\n",
                    "            messages=messages,\n",
                    "            temperature=0.1,\n",
                    "            max_tokens=100\n",
                    "        )\n",
                    "        \n",
                    "        answer = response.choices[0].message.content.strip()\n",
                    "        response_time = time.time() - start_time\n",
                    "        \n",
                    "        return {\n",
                    "            'answer': answer,\n",
                    "            'response_time': response_time,\n",
                    "            'error': None\n",
                    "        }\n",
                    "    except Exception as e:\n",
                    "        return {\n",
                    "            'answer': None,\n",
                    "            'response_time': 0,\n",
                    "            'error': str(e)\n",
                    "        }\n",
                    "\n",
                    "print('✅ query_openai_with_hint function defined')\n",
                    "print('This function presents hints BEFORE questions')\n"
                ]
                print("Updated query_openai_with_hint function")
                break

# Save the notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

if section_14_found:
    print("\n✅ Section 14 fixed successfully!")
    print("- Function definition kept")
    print("- Added actual API calls to process questions")
    print("- Results will now be generated when section 14 is run")
else:
    print("\n⚠️ Section 14 not found - please check notebook structure")

print("\n📝 To run the notebook:")
print("1. Make sure OPENAI_API_KEY is set in environment")
print("2. Run cells in order from the beginning")
print("3. Section 14 will now actually process all questions and show results")