#!/usr/bin/env python3
"""
Test script to verify the Gemini notebook can run without errors
"""

import os
import sys
import json
from typing import Dict, List, Tuple
import time

print("Testing Gemini Hints-Before Notebook Setup...")
print("=" * 50)

# Test 1: Check Python version
print("\n1. Checking Python version...")
python_version = sys.version
print(f"   Python: {python_version}")
if sys.version_info < (3, 7):
    print("   ❌ Python 3.7+ required")
    sys.exit(1)
else:
    print("   ✅ Python version OK")

# Test 2: Check required packages
print("\n2. Checking required packages...")
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy', 
    'matplotlib': 'matplotlib.pyplot',
    'seaborn': 'seaborn',
    'google.generativeai': 'google.generativeai',
    'dotenv': 'dotenv'
}

missing_packages = []
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name} - NOT INSTALLED")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n   Please install missing packages: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# Test 3: Check .env file
print("\n3. Checking .env file...")
env_path = "../.env"
if os.path.exists(env_path):
    print(f"   ✅ .env file found")
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"   ✅ GEMINI_API_KEY found")
    else:
        print(f"   ❌ GEMINI_API_KEY not found in .env")
else:
    print(f"   ❌ .env file not found at {env_path}")

# Test 4: Check data files
print("\n4. Checking data files...")
data_dirs = {
    'C-hints': ['math-questions.json', 'science-questions.json'],
    'IC-hints': ['math-questions.json', 'science-questions.json'],
    'no-hints': ['math-questions.json', 'science-questions.json']
}

all_files_found = True
for dir_name, files in data_dirs.items():
    dir_path = f"../data/{dir_name}"
    if os.path.exists(dir_path):
        print(f"   ✅ {dir_name}/ directory found")
        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.exists(file_path):
                # Check if valid JSON
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        print(f"      ✅ {file} ({len(data)} questions)")
                except json.JSONDecodeError:
                    print(f"      ❌ {file} - Invalid JSON")
                    all_files_found = False
            else:
                print(f"      ❌ {file} - NOT FOUND")
                all_files_found = False
    else:
        print(f"   ❌ {dir_name}/ directory NOT FOUND")
        all_files_found = False

if not all_files_found:
    print("\n   Some data files are missing!")

# Test 5: Test Gemini API connection
print("\n5. Testing Gemini API connection...")
try:
    import google.generativeai as genai
    api_key = os.getenv('GEMINI_API_KEY')
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Test with a simple query
        test_prompt = "What is 2+2? Answer with just the number."
        response = model.generate_content(test_prompt)
        
        if response and response.text:
            print(f"   ✅ API connection successful")
            print(f"   Test response: {response.text.strip()}")
        else:
            print(f"   ❌ API returned empty response")
    else:
        print(f"   ⚠️  Cannot test API - no API key found")
except Exception as e:
    print(f"   ❌ API connection failed: {str(e)}")

# Test 6: Check results directory
print("\n6. Checking results directory...")
results_dir = "../results"
if os.path.exists(results_dir):
    print(f"   ✅ results/ directory exists")
    
    # Check for previous results files
    gemini_files = [f for f in os.listdir(results_dir) if f.startswith('gemini_')]
    if gemini_files:
        print(f"   Found {len(gemini_files)} existing Gemini result files")
else:
    print(f"   ⚠️  results/ directory not found - will be created when running notebook")

# Test 7: Memory check
print("\n7. System resources check...")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"   Available memory: {mem.available / (1024**3):.1f} GB")
    if mem.available < 1 * (1024**3):  # Less than 1GB
        print(f"   ⚠️  Low memory - may affect performance")
except ImportError:
    print(f"   (psutil not installed - skipping memory check)")

print("\n" + "=" * 50)
print("SUMMARY:")
print("=" * 50)

if missing_packages:
    print("❌ Missing packages need to be installed")
elif not api_key:
    print("⚠️  API key not configured - notebook will fail on API calls")
elif not all_files_found:
    print("⚠️  Some data files missing - notebook may fail on data loading")
else:
    print("✅ All checks passed! The notebook should run successfully.")
    print("\nTo run the notebook:")
    print("1. Open the notebook in Jupyter or your IDE")
    print("2. Run cells sequentially from top to bottom")
    print("3. Monitor API quota if using free tier")

print("\nNote: The notebook processes 120 questions, which may take time and API quota.")