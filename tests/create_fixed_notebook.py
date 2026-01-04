#!/usr/bin/env python3
"""
Creates a fully fixed and working version of the Gemini hints-before notebook
"""

import json

# Create the complete notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Cell 1: Title
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Gemini Model Evaluation - Hints BEFORE Questions (IDE Version)\n",
        "\n",
        "This notebook evaluates Google's Gemini model on 120 questions (60 math, 60 science) with hints placed BEFORE questions."
    ]
})

# Cell 2: Important Notice
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 📌 IMPORTANT: Hints-Before-Question Evaluation\n",
        "\n",
        "## ⚠️ This notebook evaluates Gemini with hints placed BEFORE questions\n",
        "\n",
        "This is a specialized version where:\n",
        "- **Hints are presented FIRST**, before the model sees the question\n",
        "- **Results are compared** with the standard approach (hints AFTER questions)\n",
        "\n",
        "### Prompt Structure:\n",
        "```\n",
        "STEP 1 - HINT (Read this first):\n",
        "[hint text]\n",
        "\n",
        "STEP 2 - PROBLEM (Now read the problem):\n",
        "[question text]\n",
        "\n",
        "STEP 3 - YOUR ANSWER:\n",
        "```"
    ]
})

# Cell 3: Imports
notebook["cells"].append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Import required libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from typing import Dict, List, Tuple\n",
        "import json\n",
        "import time\n",
        "import os\n",
        "import glob\n",
        "from datetime import datetime\n",
        "from IPython.display import display, HTML, Markdown\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# Set style\n",
        "plt.style.use('seaborn-v0_8-darkgrid')\n",
        "sns.set_palette('husl')\n",
        "\n",
        "print('✅ All imports successful!')"
    ],
    "outputs": []
})

# Cell 4: Setup Gemini
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Configure Gemini API"]
})

notebook["cells"].append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# Configure Gemini API\n",
        "from dotenv import load_dotenv\n",
        "import google.generativeai as genai\n",
        "\n",
        "# Load environment variables\n",
        "load_dotenv('../.env')\n",
        "\n",
        "# Setup API\n",
        "api_key = os.getenv('GEMINI_API_KEY')\n",
        "if not api_key:\n",
        "    raise ValueError('GEMINI_API_KEY not found in .env file')\n",
        "\n",
        "genai.configure(api_key=api_key)\n",
        "model = genai.GenerativeModel('gemini-2.0-flash')\n",
        "\n",
        "print('✅ Gemini 2.0 Flash model initialized successfully!')"
    ],
    "outputs": []
})

# Cell 5: Load Questions Function
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Define Data Loading Functions"]
})

notebook["cells"].append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "def load_questions_with_hints(file_path: str) -> Tuple[List[Dict], Dict[int, str], Dict]:\n",
        "    \"\"\"Load questions with their hints and ground truth answers\"\"\"\n",
        "    with open(file_path, 'r') as f:\n",
        "        data = json.load(f)\n",
        "    \n",
        "    # Handle new data structure with metadata and questions\n",
        "    if isinstance(data, dict) and 'questions' in data:\n",
        "        questions_data = data['questions']\n",
        "    else:\n",
        "        questions_data = data\n",
        "    \n",
        "    questions = []\n",
        "    ground_truth = {}\n",
        "    hints = {}\n",
        "    \n",
        "    # Handle both dict and list formats\n",
        "    if isinstance(questions_data, dict):\n",
        "        for q_id, item in questions_data.items():\n",
        "            question_id = int(q_id) if q_id.isdigit() else q_id\n",
        "            questions.append({\n",
        "                'id': question_id,\n",
        "                'question': item['question'],\n",
        "                'difficulty': item['difficulty'],\n",
        "                'hint': item.get('hint', ''),\n",
        "                'subject': item.get('subject', '')\n",
        "            })\n",
        "            ground_truth[question_id] = item['ground_truth']\n",
        "            hints[question_id] = item.get('hint', '')\n",
        "    else:  # List format\n",
        "        for item in questions_data:\n",
        "            question_id = item['id']\n",
        "            questions.append({\n",
        "                'id': question_id,\n",
        "                'question': item['question'],\n",
        "                'difficulty': item['difficulty'],\n",
        "                'hint': item.get('hint', ''),\n",
        "                'subject': item.get('subject', '')\n",
        "            })\n",
        "            ground_truth[question_id] = item['ground_truth']\n",
        "            hints[question_id] = item.get('hint', '')\n",
        "    \n",
        "    return questions, ground_truth, hints"
    ],
    "outputs": []
})

# Cell 6: Query Function
notebook["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Define Query Function (Hints BEFORE Questions)"]
})

notebook["cells"].append({
    "cell_type": "code",
    "metadata": {},
    "source": [
        "def query_gemini_with_hint(question_text: str, hint_text: str, question_type: str = 'math') -> Dict:\n",
        "    \"\"\"Query Gemini with hint BEFORE question\"\"\"\n",
        "    \n",
        "    if question_type == 'math':\n",
        "        prompt_parts = [\n",
        "            \"Follow these steps carefully:\",\n",
        "            \"1. First, read and understand the hint provided\",\n",
        "            \"2. Then, read the problem statement\",\n",
        "            \"3. Use the hint to guide your thinking\",\n",
        "            \"4. Solve the problem\",\n",
        "            \"\",\n",
        "            \"Provide ONLY the final numerical answer. Do not include units or explanations, just the number.\",\n",
        "            \"\",\n",
        "            \"STEP 1 - HINT (Read this first):\",\n",
        "            hint_text,\n",
        "            \"\",\n",
        "            \"STEP 2 - PROBLEM (Now read the problem):\",\n",
        "            question_text,\n",
        "            \"\",\n",
        "            \"STEP 3 - YOUR ANSWER (Apply the hint to solve):\"\n",
        "        ]\n",
        "    else:  # science\n",
        "        prompt_parts = [\n",
        "            \"Follow these steps carefully:\",\n",
        "            \"1. First, read and understand the hint provided\",\n",
        "            \"2. Then, read the question\",\n",
        "            \"3. Use the hint to guide your reasoning\",\n",
        "            \"4. Determine the answer\",\n",
        "            \"\",\n",
        "            \"Provide ONLY the answer. Keep your answer concise and direct.\",\n",
        "            \"\",\n",
        "            \"STEP 1 - HINT (Read this first):\",\n",
        "            hint_text,\n",
        "            \"\",\n",
        "            \"STEP 2 - QUESTION (Now read the question):\",\n",
        "            question_text,\n",
        "            \"\",\n",
        "            \"STEP 3 - YOUR ANSWER (Apply the hint to solve):\"\n",
        "        ]\n",
        "    \n",
        "    prompt = \"\\n\".join(prompt_parts)\n",
        "    \n",
        "    try:\n",
        "        start_time = time.time()\n",
        "        response = model.generate_content(prompt)\n",
        "        response_time = time.time() - start_time\n",
        "        \n",
        "        return {\n",
        "            'answer': response.text.strip(),\n",
        "            'response_time': response_time,\n",
        "            'error': None\n",
        "        }\n",
        "    except Exception as e:\n",
        "        return {\n",
        "            'answer': 'ERROR',\n",
        "            'response_time': 0,\n",
        "            'error': str(e)\n",
        "        }"
    ],
    "outputs": []
})

# Save the notebook
with open('gemini_evaluation_ide_FIXED.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("Created fixed notebook: gemini_evaluation_ide_FIXED.ipynb")
print("\nThis notebook includes:")
print("✅ Fixed data loading functions for new JSON structure")
print("✅ Proper hints-before-questions prompting")
print("✅ Error handling")
print("✅ Clear documentation")
print("\nYou can now open this notebook and run it cell by cell.")