# Makefile for Inter-System Communication Language Project

.PHONY: all clean install test docs help cpp python rust erlang

# Default target
all: help

help:
	@echo "Inter-System Communication Language - Build System"
	@echo "=================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make cpp        - Build C++ examples"
	@echo "  make python     - Setup Python environment"
	@echo "  make rust       - Build Rust examples"
	@echo "  make erlang     - Compile Erlang examples"
	@echo "  make test       - Run all tests"
	@echo "  make docs       - Generate documentation"
	@echo "  make clean      - Clean build artifacts"
	@echo "  make install    - Install all components"
	@echo ""

# C++ targets
cpp:
	@echo "Building C++ examples..."
	@mkdir -p build
	@cd build && cmake .. && make
	@echo "C++ build complete!"

cpp-clean:
	@rm -rf build
	@echo "C++ build artifacts cleaned!"

# Python targets
python:
	@echo "Setting up Python environment..."
	@python3 -m pip install --upgrade pip
	@pip install -r requirements.txt
	@echo "Python environment ready!"

python-test:
	@echo "Running Python tests..."
	@pytest tests/python/ -v
	@echo "Python tests complete!"

# Rust targets
rust:
	@echo "Building Rust examples..."
	@cargo build --release
	@echo "Rust build complete!"

rust-test:
	@echo "Running Rust tests..."
	@cargo test
	@echo "Rust tests complete!"

rust-clean:
	@cargo clean
	@echo "Rust build artifacts cleaned!"

# Erlang targets
erlang:
	@echo "Compiling Erlang examples..."
	@cd examples/erlang && erlc *.erl
	@echo "Erlang compilation complete!"

erlang-clean:
	@cd examples/erlang && rm -f *.beam
	@echo "Erlang build artifacts cleaned!"

# Test all
test: python-test rust-test
	@echo "All tests complete!"

# Documentation
docs:
	@echo "Generating documentation..."
	@cd build && make docs
	@cd docs && sphinx-build -b html . _build
	@echo "Documentation generated in build/docs and docs/_build!"

# Clean all
clean: cpp-clean rust-clean erlang-clean
	@find . -type d -name __pycache__ -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "All build artifacts cleaned!"

# Install
install: all
	@echo "Installing components..."
	@cd build && make install
	@pip install -e .
	@echo "Installation complete!"

# Format code
format:
	@echo "Formatting Python code..."
	@black examples/python/ src/python/
	@echo "Formatting Rust code..."
	@cargo fmt
	@echo "Code formatting complete!"

# Lint
lint:
	@echo "Linting Python code..."
	@flake8 examples/python/ src/python/
	@echo "Linting Rust code..."
	@cargo clippy
	@echo "Linting complete!"
