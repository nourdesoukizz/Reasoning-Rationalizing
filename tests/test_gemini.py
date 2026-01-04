#!/usr/bin/env python3
"""
Test script to verify Gemini API connection
"""

import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv('../.env')

# Get API key
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file")
    sys.exit(1)

print(f"✅ API Key found: {api_key[:10]}...")

# Configure Gemini
try:
    genai.configure(api_key=api_key)
    print("✅ Gemini configured successfully")
except Exception as e:
    print(f"❌ Error configuring Gemini: {e}")
    sys.exit(1)

# Initialize model
try:
    model = genai.GenerativeModel('gemini-1.5-pro')
    print("✅ Model initialized successfully")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    sys.exit(1)

# Test query
try:
    print("\n📝 Testing with a simple math question...")
    response = model.generate_content("What is 2 + 2? Provide only the number.")
    print(f"✅ Response received: {response.text}")
except Exception as e:
    print(f"❌ Error during API call: {e}")
    print("\nPossible issues:")
    print("1. API key might be invalid or expired")
    print("2. Internet connection issues")
    print("3. Gemini API service might be down")
    print("4. Rate limits might be exceeded")
    sys.exit(1)

print("\n✨ All tests passed! Gemini API is working correctly.")