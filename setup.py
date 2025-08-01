#!/usr/bin/env python3
"""
Setup script for OpenVINO Object Detection Demo
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"Command: {cmd}")
        print(f"Error: {e.stderr}")
        sys.exit(1)

def check_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("✅ uv is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ uv is not installed")
        return False

def install_uv():
    """Install uv package manager."""
    print("📦 Installing uv...")
    if sys.platform == "darwin":  # macOS
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh", "Installing uv")
    elif sys.platform.startswith("linux"):  # Linux
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh", "Installing uv")
    elif sys.platform == "win32":  # Windows
        run_command("powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"", "Installing uv")
    else:
        print(f"❌ Unsupported platform: {sys.platform}")
        print("Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

def setup_environment():
    """Set up the virtual environment and install dependencies."""
    print("🚀 Setting up OpenVINO Object Detection Demo")
    print("=" * 50)
    
    # Check if uv is installed
    if not check_uv_installed():
        install_uv()
        # Reload PATH to use newly installed uv
        if sys.platform != "win32":
            os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
    
    # Create virtual environment
    run_command("uv venv", "Creating virtual environment")
    
    # Install dependencies
    run_command("uv pip install -r requirements.txt", "Installing dependencies")
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo activate the environment and run the demo:")
    
    if sys.platform == "win32":
        print("  .venv\\Scripts\\activate")
    else:
        print("  source .venv/bin/activate")
    
    print("  python openvino_detector.py")
    print("\nFor the web demo, simply open index.html in your browser.")

if __name__ == "__main__":
    setup_environment()