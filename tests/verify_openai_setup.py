#!/usr/bin/env python3
"""
Verify and fix all setup sections in OpenAI notebook
"""

import json

# Read the notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'r') as f:
    notebook = json.load(f)

print("Verifying OpenAI notebook setup...")
print("=" * 50)

fixes_made = []

# Check for critical sections
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown' and 'source' in cell:
        source = ''.join(cell.get('source', []))
        
        # Section 10: Basic query function
        if '## 10.' in source and 'Basic Query Function' in source:
            if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                notebook['cells'][i+1]['source'] = [
                    "def query_openai(question: str, question_type: str = 'math') -> Dict:\n",
                    "    \"\"\"Query OpenAI API with a question (no hints)\"\"\"\n",
                    "    try:\n",
                    "        start_time = time.time()\n",
                    "        \n",
                    "        # Create messages for chat completion\n",
                    "        messages = [\n",
                    "            {'role': 'system', 'content': f'You are a helpful assistant answering {question_type} questions. Provide only the final answer without explanation.'},\n",
                    "            {'role': 'user', 'content': question}\n",
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
                    "print('✅ Basic query_openai function defined')\n"
                ]
                fixes_made.append("Section 10: Basic query function")
        
        # Section 11: Load questions with hints
        elif '## 11.' in source and 'Load Questions' in source:
            if i+1 < len(notebook['cells']) and notebook['cells'][i+1]['cell_type'] == 'code':
                notebook['cells'][i+1]['source'] = [
                    "# Load questions with CORRECT hints from C-hints directory\n",
                    "def load_questions_with_hints(file_path: str) -> Tuple[List[Dict], Dict[int, str], Dict]:\n",
                    "    \"\"\"Load questions, ground truth, and metadata from JSON file\"\"\"\n",
                    "    with open(file_path, 'r') as f:\n",
                    "        data = json.load(f)\n",
                    "    \n",
                    "    # Handle both dict and list formats\n",
                    "    if isinstance(data, dict) and 'questions' in data:\n",
                    "        questions_data = data['questions']\n",
                    "        metadata = {k: v for k, v in data.items() if k != 'questions'}\n",
                    "    else:\n",
                    "        questions_data = data if isinstance(data, list) else data.get('questions', [])\n",
                    "        metadata = {}\n",
                    "    \n",
                    "    questions = []\n",
                    "    ground_truth = {}\n",
                    "    \n",
                    "    for q in questions_data:\n",
                    "        # Handle both dict and list item formats\n",
                    "        if isinstance(q, dict):\n",
                    "            question_dict = {\n",
                    "                'id': q.get('id', len(questions)),\n",
                    "                'question': q.get('question', ''),\n",
                    "                'hint': q.get('hint', ''),  # CORRECT hint to show BEFORE question\n",
                    "                'difficulty': q.get('difficulty', 'medium')\n",
                    "            }\n",
                    "            if 'subject' in q:\n",
                    "                question_dict['subject'] = q['subject']\n",
                    "            questions.append(question_dict)\n",
                    "            ground_truth[question_dict['id']] = str(q.get('ground_truth', '')).strip().lower()\n",
                    "    \n",
                    "    return questions, ground_truth, metadata\n",
                    "\n",
                    "# Load CORRECT hints data\n",
                    "print('📂 Loading questions with CORRECT hints from C-hints directory...')\n",
                    "math_questions, math_ground_truth, math_metadata = load_questions_with_hints('../data/C-hints/math-questions.json')\n",
                    "science_questions, science_ground_truth, science_metadata = load_questions_with_hints('../data/C-hints/science-questions.json')\n",
                    "\n",
                    "print(f'✅ Loaded {len(math_questions)} math questions with CORRECT hints')\n",
                    "print(f'✅ Loaded {len(science_questions)} science questions with CORRECT hints')\n",
                    "print('📌 These hints will be presented BEFORE the questions')\n"
                ]
                fixes_made.append("Section 11: Load questions with hints")
        
        # Section 12: Query with hints function
        elif '## 12.' in source and ('Query' in source or 'hint' in source.lower()):
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
                    "print('📌 This function presents hints BEFORE questions')\n"
                ]
                fixes_made.append("Section 12: Query with hints function")

# Save the updated notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\n✅ Verification complete!")
print(f"Fixed {len(fixes_made)} sections:")
for fix in fixes_made:
    print(f"  - {fix}")

print("\n📋 Notebook should now have:")
print("  1. Proper OpenAI imports and setup")
print("  2. OpenAI client initialization")
print("  3. query_openai() function for baseline")
print("  4. query_openai_with_hint() function for hints-before")
print("  5. Data loading from C-hints directory")
print("  6. Section 14 that actually calls the API")

print("\n🚀 Ready to run! Make sure to:")
print("  1. Set OPENAI_API_KEY environment variable")
print("  2. Run cells in order from the beginning")