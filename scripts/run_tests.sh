#!/bin/bash
# Run all tests across all languages

set -e

echo "========================================"
echo "Running All Tests"
echo "========================================"
echo ""

# C++ Tests
if [ -f "build/tests/test_language_model" ]; then
    echo "=== C++ Tests ==="
    cd build/tests
    ./test_language_model
    cd ../..
    echo "✓ C++ tests passed"
    echo ""
else
    echo "✗ C++ tests not built"
    echo ""
fi

# Python Tests
echo "=== Python Tests ==="
if command -v pytest &> /dev/null; then
    PYTHONPATH=examples/python:$PYTHONPATH pytest tests/python -v --tb=short || true
    echo ""
else
    echo "pytest not installed. Install with: pip install pytest"
    echo ""
fi

# Rust Tests
echo "=== Rust Tests ==="
if command -v cargo &> /dev/null; then
    cargo test 2>&1 | grep -E "(test result|running)" || true
    echo "✓ Rust tests completed"
    echo ""
else
    echo "Rust not installed"
    echo ""
fi

echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Check output above for test results"
