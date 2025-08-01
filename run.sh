#!/bin/bash

# OpenVINO Object Detection Demo Runner
# This script sets up the environment and runs the demo

set -e

echo "🚀 OpenVINO Object Detection Demo"
echo "================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
if [ ! -f ".venv/pyvenv.cfg" ] || [ "requirements.txt" -nt ".venv/pyvenv.cfg" ]; then
    echo "📥 Installing dependencies..."
    uv pip install -r requirements.txt
fi

# Run the detector
echo "🎯 Starting object detection..."
python openvino_detector.py