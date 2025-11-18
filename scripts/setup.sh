#!/bin/bash
# Setup script for Inter-System Communication Language
# Automates the installation of all dependencies

set -e  # Exit on error

echo "=================================="
echo "Inter-System Communication Language"
echo "Automated Setup Script"
echo "=================================="
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

echo "Detected OS: $OS"
echo ""

# Install system dependencies
echo "Installing system dependencies..."
if [[ "$OS" == "linux" ]]; then
    sudo apt-get update
    sudo apt-get install -y build-essential cmake git
    sudo apt-get install -y libeigen3-dev
    sudo apt-get install -y python3 python3-pip python3-venv
    sudo apt-get install -y erlang
elif [[ "$OS" == "macos" ]]; then
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install Homebrew first."
        exit 1
    fi
    brew install cmake eigen gcc python@3.10 erlang
fi

echo "✓ System dependencies installed"
echo ""

# Setup Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Python environment ready"
echo ""

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo "✓ Rust installed"
else
    echo "✓ Rust already installed"
fi
echo ""

# Build the project
echo "Building the project..."
make all

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Activate Python venv: source venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. See README.md for usage examples"
echo ""
