#!/bin/bash
# Demo script showing all language implementations

set -e

echo "========================================"
echo "Inter-System Communication Language Demo"
echo "========================================"
echo ""

# C++ Demo
if [ -f "build/language_model" ]; then
    echo "=== C++ Implementation ==="
    ./build/language_model
    echo ""
else
    echo "C++ not built. Run 'make cpp' first."
    echo ""
fi

# Python Demo
echo "=== Python Implementation ==="
python3 << 'EOF'
import sys
sys.path.insert(0, 'examples/python')

# Simple demo without real models
print("Python language model framework initialized")
print("Note: Install models to see full functionality")
print("Response: [Demo mode - no trained models loaded]")
EOF
echo ""

# Rust Demo
if [ -f "target/release/inter-system-communication" ]; then
    echo "=== Rust Implementation ==="
    echo "Rust implementation compiled successfully"
    echo "Note: Run 'cargo run --example rnn_encoder_decoder' to see examples"
    echo ""
else
    echo "Rust not built. Run 'cargo build --release' first."
    echo ""
fi

# Erlang Demo
if command -v erl &> /dev/null; then
    echo "=== Erlang Implementation ==="
    echo "Erlang modules available in examples/erlang/"
    echo "Note: Compile with 'erlc examples/erlang/*.erl'"
    echo ""
else
    echo "Erlang not installed. Install with: sudo apt-get install erlang"
    echo ""
fi

echo "========================================"
echo "Demo Complete!"
echo "========================================"
