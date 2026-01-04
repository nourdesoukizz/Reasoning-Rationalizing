#!/usr/bin/env python3
"""
Fix OpenAI notebook to properly use OpenAI API instead of Gemini
"""

import json
import re

# Read the notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'r') as f:
    notebook = json.load(f)

# Define the proper OpenAI API setup code
openai_setup = """import os
import sys
import json
import time
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('.'))))

print("✅ All imports successful!")"""

# Define the OpenAI API configuration
api_config = """# Configure OpenAI API
from openai import OpenAI
import os

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("⚠️ Warning: OPENAI_API_KEY not found in environment")
    print("Please set it using: export OPENAI_API_KEY='your-key-here'")
    api_key = input("Enter your OpenAI API key: ").strip()

client = OpenAI(api_key=api_key)

# Test the connection
try:
    # Simple test to verify API key
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API connected'"}],
        max_tokens=10
    )
    print(f"✅ OpenAI API connected successfully!")
    print(f"Model: gpt-4o-mini")
except Exception as e:
    print(f"❌ Error connecting to OpenAI API: {e}")"""

# Define the query function for OpenAI
query_function = """def query_openai(question: str, question_type: str = "math") -> Dict:
    \"\"\"Query OpenAI API with a question\"\"\"
    try:
        start_time = time.time()
        
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": f"You are a helpful assistant answering {question_type} questions. Provide only the final answer without explanation."},
            {"role": "user", "content": question}
        ]
        
        # Query OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content.strip()
        response_time = time.time() - start_time
        
        return {
            'answer': answer,
            'response_time': response_time,
            'error': None
        }
    except Exception as e:
        return {
            'answer': None,
            'response_time': 0,
            'error': str(e)
        }"""

# Define the query with hint function for OpenAI
query_with_hint_function = """def query_openai_with_hint(question_text: str, hint_text: str, question_type: str = "math") -> Dict:
    \"\"\"
    Query OpenAI with a hint presented BEFORE the question.
    
    Args:
        question_text: The question to ask
        hint_text: The hint to provide BEFORE the question
        question_type: Type of question (math or science)
    \"\"\"
    try:
        start_time = time.time()
        
        # Construct prompt with hint BEFORE question
        prompt_parts = [
            "Follow these instructions carefully:",
            "1. First, read the hint provided below",
            "2. Then read the question", 
            "3. Use the hint to guide your solution",
            "4. Provide ONLY the final answer",
            "",
            "=== HINT (Read this first) ===",
            hint_text,
            "",
            "=== PROBLEM (Now solve using the hint above) ===",
            question_text,
            "",
            "Remember: Provide ONLY the final answer, no explanations."
        ]
        
        prompt = "\\n".join(prompt_parts)
        
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": f"You are an expert {question_type} problem solver. Follow instructions carefully and use hints effectively."},
            {"role": "user", "content": prompt}
        ]
        
        # Query OpenAI with structured prompt
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=100
        )
        
        answer = response.choices[0].message.content.strip()
        response_time = time.time() - start_time
        
        return {
            'answer': answer,
            'response_time': response_time,
            'error': None
        }
    except Exception as e:
        return {
            'answer': None,
            'response_time': 0,
            'error': str(e)
        }"""

# Fix specific sections
sections_to_fix = {
    "# OpenAI LLM Evaluation - Mathematical & Scientific Reasoning (Hints BEFORE Questions)": {
        "type": "markdown",
        "content": ["# OpenAI LLM Evaluation - Mathematical & Scientific Reasoning (Hints BEFORE Questions)\\n",
                   "\\n",
                   "This notebook evaluates OpenAI's GPT-4o-mini model on mathematical and scientific reasoning tasks.\\n",
                   "\\n", 
                   "**Key Feature**: This version presents hints BEFORE questions to test if providing guidance upfront improves model performance.\\n",
                   "\\n",
                   "## Evaluation Conditions:\\n",
                   "1. **Baseline**: No hints provided\\n",
                   "2. **Correct Hints BEFORE**: Helpful hints shown before questions\\n", 
                   "3. **Incorrect Hints BEFORE**: Misleading hints shown before questions\\n",
                   "\\n",
                   "## Model Configuration:\\n",
                   "- Model: GPT-4o-mini\\n",
                   "- Temperature: 0.1 (for consistency)\\n",
                   "- Max tokens: 100 (for concise answers)"]
    },
    "## 1. Setup and Imports": {
        "type": "code",
        "content": openai_setup.split('\n')
    },
    "## 2. Configure OpenAI API": {
        "type": "code", 
        "content": api_config.split('\n')
    },
    "## 3. Basic Query Function": {
        "type": "code",
        "content": query_function.split('\n')
    },
    "## 4. Query Function with Hints": {
        "type": "code",
        "content": query_with_hint_function.split('\n')
    }
}

# Apply targeted fixes
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        
        # Check for sections to fix
        for section_title, fix in sections_to_fix.items():
            if section_title in source and fix['type'] == 'markdown':
                cell['source'] = fix['content']
                print(f"Fixed markdown section: {section_title[:50]}...")
                break
                
    elif cell['cell_type'] == 'code':
        # Check if this is right after a section we need to fix
        if i > 0 and notebook['cells'][i-1]['cell_type'] == 'markdown':
            prev_source = ''.join(notebook['cells'][i-1].get('source', []))
            
            for section_title, fix in sections_to_fix.items():
                if section_title in prev_source and fix['type'] == 'code':
                    cell['source'] = fix['content']
                    print(f"Fixed code section after: {section_title[:50]}...")
                    break

# Additional comprehensive replacements
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        
        # Fix any remaining Gemini references
        source = source.replace('gemini', 'openai')
        source = source.replace('Gemini', 'OpenAI')
        source = source.replace('GEMINI', 'OPENAI')
        
        # Fix model references
        source = source.replace('gpt-4o-mini-2.0-flash-exp', 'gpt-4o-mini')
        source = source.replace('model = genai.GenerativeModel', '# Using OpenAI client')
        source = source.replace('model.generate_content', 'client.chat.completions.create')
        
        # Update function calls
        source = source.replace('query_gemini_with_hint', 'query_openai_with_hint')
        source = source.replace('query_gemini', 'query_openai')
        
        # Update result processing
        source = source.replace('response.text', 'response.choices[0].message.content')
        
        cell['source'] = source.split('\n') if source else []
        
    elif cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        
        # Update all text references
        source = source.replace('Gemini', 'OpenAI')
        source = source.replace('gemini', 'OpenAI')
        source = source.replace('Google AI', 'OpenAI')
        source = source.replace('Gemini 2.0 Flash', 'GPT-4o-mini')
        
        cell['source'] = source.split('\n') if source else []

# Save the fixed notebook
with open('openai_evaluation_ide_hintsbefore copy.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("\n✅ OpenAI notebook has been fully updated!")
print("All Gemini references replaced with OpenAI")
print("API setup and query functions updated for OpenAI")
print("Model changed to gpt-4o-mini")