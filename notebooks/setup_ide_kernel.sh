#!/bin/bash

# Setup script for configuring Jupyter kernel for IDE use
# This ensures the IDE uses the same Python environment as the web Jupyter

echo "🔧 Setting up Jupyter kernel for IDE compatibility..."

# Python path where packages are installed
PYTHON_PATH="/Users/nourdesouki/.pyenv/versions/3.12.8/bin/python3"

# Check if Python exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Python not found at $PYTHON_PATH"
    echo "Please update the PYTHON_PATH variable in this script"
    exit 1
fi

echo "✅ Found Python at: $PYTHON_PATH"

# Install ipykernel if not already installed
echo "📦 Ensuring ipykernel is installed..."
$PYTHON_PATH -m pip install ipykernel

# Create and register the kernel
echo "🔄 Registering Jupyter kernel..."
$PYTHON_PATH -m ipykernel install --user --name gemini-env --display-name "Python 3.12 (Gemini)"

echo ""
echo "✅ Kernel setup complete!"
echo ""
echo "📝 To use in your IDE:"
echo "1. Open the notebook in your IDE"
echo "2. Select kernel: 'Python 3.12 (Gemini)'"
echo "3. This kernel has all required packages installed"
echo ""
echo "🔍 Verifying package installation..."

# Verify packages
$PYTHON_PATH -c "
import sys
print(f'Python: {sys.version}')
print(f'Path: {sys.executable}')
print()
print('Checking required packages:')
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'google.generativeai', 'dotenv']
for pkg in packages:
    try:
        __import__(pkg.replace('.', '_').replace('-', '_'))
        print(f'  ✅ {pkg}')
    except ImportError:
        print(f'  ❌ {pkg} - needs installation')
"

echo ""
echo "🎯 Next steps:"
echo "1. Run the notebook using the 'Python 3.12 (Gemini)' kernel"
echo "2. If your IDE doesn't show this kernel, restart the IDE"