#!/usr/bin/env python3
"""
Complete fix for OpenAI notebook - ensures all sections use OpenAI API
"""

import json

# Read the notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'r') as f:
    notebook = json.load(f)

# Complete notebook sections with OpenAI-specific code
complete_sections = {
    0: {  # Title section
        "markdown": ["# OpenAI LLM Evaluation - Mathematical & Scientific Reasoning (Hints BEFORE Questions)\n",
                    "\n",
                    "This notebook evaluates OpenAI's GPT-4o-mini model on mathematical and scientific reasoning tasks.\n",
                    "\n", 
                    "**Key Feature**: This version presents hints BEFORE questions to test if providing guidance upfront improves model performance.\n",
                    "\n",
                    "## Evaluation Conditions:\n",
                    "1. **Baseline**: No hints provided\n",
                    "2. **Correct Hints BEFORE**: Helpful hints shown before questions\n", 
                    "3. **Incorrect Hints BEFORE**: Misleading hints shown before questions\n",
                    "\n",
                    "## Model Configuration:\n",
                    "- Model: GPT-4o-mini\n",
                    "- Temperature: 0.1 (for consistency)\n",
                    "- Max tokens: 100 (for concise answers)\n",
                    "- API: OpenAI Chat Completions\n",
                    "\n",
                    "## Hints-Before Approach:\n",
                    "In this notebook, all hints are presented BEFORE the questions to evaluate if upfront guidance affects model reasoning."]
    },
    1: {  # Setup section
        "code": ["import os\n",
                "import sys\n", 
                "import json\n",
                "import time\n",
                "from typing import Dict, List, Tuple, Optional\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from openai import OpenAI\n",
                "from pathlib import Path\n",
                "from datetime import datetime\n",
                "\n",
                "# Add parent directory to path\n",
                "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('.'))))\n",
                "\n",
                "# Set visualization style\n",
                "plt.style.use('seaborn-v0_8-darkgrid')\n",
                "sns.set_palette('husl')\n",
                "\n",
                "print('✅ All imports successful!')\n",
                "print(f'OpenAI library version: {OpenAI.__version__ if hasattr(OpenAI, \"__version__\") else \"Latest\"}')\n",
                "print(f'Working directory: {os.getcwd()}')"]
    },
    2: {  # API configuration
        "code": ["# Configure OpenAI API\n",
                "from openai import OpenAI\n",
                "import os\n",
                "\n",
                "# Initialize OpenAI client\n",
                "api_key = os.getenv('OPENAI_API_KEY')\n",
                "if not api_key:\n",
                "    print('⚠️ Warning: OPENAI_API_KEY not found in environment')\n",
                "    print('Please set it using: export OPENAI_API_KEY=\"your-key-here\"')\n",
                "    api_key = input('Enter your OpenAI API key: ').strip()\n",
                "\n",
                "# Create client instance\n",
                "client = OpenAI(api_key=api_key)\n",
                "\n",
                "# Test the connection\n",
                "try:\n",
                "    # Simple test to verify API key\n",
                "    test_response = client.chat.completions.create(\n",
                "        model='gpt-4o-mini',\n",
                "        messages=[{'role': 'user', 'content': 'Say \"Connected\"'}],\n",
                "        max_tokens=10,\n",
                "        temperature=0\n",
                "    )\n",
                "    print(f'✅ OpenAI API connected successfully!')\n",
                "    print(f'Model: gpt-4o-mini')\n",
                "    print(f'Response: {test_response.choices[0].message.content}')\n",
                "except Exception as e:\n",
                "    print(f'❌ Error connecting to OpenAI API: {e}')\n",
                "    print('Please check your API key and internet connection')"]
    }
}

# Helper function to update process_questions functions
def create_process_function(hint_type="correct"):
    """Create process questions function for OpenAI"""
    if hint_type == "correct":
        return [
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
            "    return results\n"
        ]
    else:  # incorrect
        return [
            "# Process questions with INCORRECT hints BEFORE questions\n",
            "def process_questions_with_ic_hints(questions: List[Dict], question_type: str) -> List[Dict]:\n",
            "    \"\"\"Process questions with INCORRECT hints presented BEFORE questions\"\"\"\n",
            "    results = []\n",
            "    total = len(questions)\n",
            "    \n",
            "    print(f'⚠️ Processing {total} {question_type} questions with INCORRECT hints...')\n",
            "    print(f'📌 Misleading hints will be shown BEFORE questions')\n",
            "    print('='*50)\n",
            "    \n",
            "    for i, q in enumerate(questions, 1):\n",
            "        # Pass INCORRECT hint BEFORE question\n",
            "        response = query_openai_with_hint(\n",
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
            "            print(f'Progress: {i}/{total} - Incorrect hints before questions')\n",
            "        \n",
            "        time.sleep(0.5)  # Rate limiting for OpenAI API\n",
            "    \n",
            "    print(f'\\n✅ Completed {total} questions with INCORRECT hints BEFORE')\n",
            "    return results\n"
        ]

# Apply specific section fixes
for idx, fixes in complete_sections.items():
    if idx < len(notebook['cells']):
        cell = notebook['cells'][idx]
        if 'markdown' in fixes and cell['cell_type'] == 'markdown':
            cell['source'] = fixes['markdown']
        elif 'code' in fixes and cell['cell_type'] == 'code':
            cell['source'] = fixes['code']

# Now fix all cells comprehensively
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        
        # Skip if empty
        if not source.strip():
            continue
            
        # Fix query functions
        if 'def query_gemini' in source or 'def query_openai' in source:
            if 'with_hint' in source:
                # Query with hint function
                cell['source'] = [
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
                    "        }\n"
                ]
            else:
                # Basic query function
                cell['source'] = [
                    "def query_openai(question: str, question_type: str = 'math') -> Dict:\n",
                    "    \"\"\"Query OpenAI API with a question\"\"\"\n",
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
                    "        }\n"
                ]
        
        # Fix process functions with correct hints
        elif 'def process_questions_with_hints' in source and 'INCORRECT' not in source:
            cell['source'] = create_process_function("correct")
        
        # Fix process functions with incorrect hints  
        elif 'def process_questions_with_ic_hints' in source or ('process_questions' in source and 'INCORRECT' in source):
            cell['source'] = create_process_function("incorrect")
        
        # Fix any remaining references
        else:
            # General replacements
            source = source.replace('gemini', 'openai')
            source = source.replace('Gemini', 'OpenAI')
            source = source.replace('GEMINI', 'OPENAI')
            source = source.replace('genai.', 'client.')
            source = source.replace('GenerativeModel', 'OpenAI')
            source = source.replace('generate_content', 'chat.completions.create')
            source = source.replace('gemini-2.0-flash-exp', 'gpt-4o-mini')
            source = source.replace('gemini-pro', 'gpt-4o-mini')
            source = source.replace('Google AI', 'OpenAI')
            source = source.replace('query_gemini', 'query_openai')
            
            cell['source'] = source.split('\\n') if source else []
    
    elif cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        
        # Update all text references
        source = source.replace('Gemini', 'OpenAI')
        source = source.replace('gemini', 'OpenAI')  
        source = source.replace('GEMINI', 'OPENAI')
        source = source.replace('Google AI', 'OpenAI')
        source = source.replace('Gemini 2.0 Flash', 'GPT-4o-mini')
        source = source.replace('gemini-2.0-flash', 'gpt-4o-mini')
        
        # Update specific OpenAI references
        source = source.replace('OpenAI API', 'OpenAI API')
        source = source.replace('OpenAI model', 'OpenAI GPT-4o-mini model')
        
        cell['source'] = source.split('\\n') if source else []

# Save the updated notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print('✅ OpenAI notebook completely updated!')
print('- All Gemini references replaced with OpenAI')
print('- API setup configured for OpenAI')
print('- Query functions updated for OpenAI Chat Completions')
print('- Model set to gpt-4o-mini')
print('- All comments and documentation updated')
print('- Process functions maintain hints-before approach')