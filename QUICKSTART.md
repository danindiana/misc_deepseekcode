# Quick Start Guide

Get up and running with Inter-System Communication Language in 5 minutes!

## Prerequisites Check

```bash
# Check Python
python3 --version  # Should be 3.8+

# Check C++ compiler
gcc --version  # Should be 10+

# Check Rust (optional)
rustc --version

# Check Erlang (optional)
erl -version
```

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/danindiana/misc_deepseekcode.git
cd misc_deepseekcode

# Run automated setup
./scripts/setup.sh
```

### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/danindiana/misc_deepseekcode.git
cd misc_deepseekcode

# Install Python dependencies
pip install -r requirements.txt

# Build C++ examples
mkdir build && cd build
cmake ..
make
cd ..

# Build Rust examples (optional)
cargo build --release
```

## Quick Demo

Run the demo script to see all implementations:

```bash
./scripts/demo.sh
```

## Language-Specific Examples

### Python

```python
from examples.python.inter_system_communication import InterSystemCommunicationLanguage

# Initialize system
system = InterSystemCommunicationLanguage()

# Add models (when available)
# system.add_language_model('statistical', 'models/statistical_model.pkl')
# system.add_language_model('neural_network', 'models/neural_network_model.h5')

# Generate responses
responses = system.communicate("Hello, world!")
print(responses)
```

### C++

```bash
# Build and run
cd build
./language_model
```

Output:
```
Statistical Language Model initialized
Neural Network Language Model initialized
Optimized Response: Statistical response to: Hello, how are you?
```

### Rust

```bash
# Build and run examples
cargo build --release
cargo run --example rnn_encoder_decoder
```

### Erlang

```bash
# Compile modules
cd examples/erlang
erlc *.erl

# Run in Erlang shell
erl
```

```erlang
% In Erlang shell:
1> c(rnn_encdec).
2> rnn_encdec:encode(["hello", "world"]).
```

## Running Tests

### All Tests

```bash
./scripts/run_tests.sh
```

### Individual Tests

```bash
# C++ tests
cd build/tests
./test_language_model

# Python tests
pytest tests/python -v

# Rust tests
cargo test
```

## Project Structure

```
misc_deepseekcode/
├── examples/          # Example implementations
│   ├── cpp/          # C++ examples
│   ├── python/       # Python examples
│   ├── rust/         # Rust examples
│   └── erlang/       # Erlang examples
├── src/              # Source code
├── tests/            # Test suites
├── docs/             # Documentation
├── models/           # Trained models (create your own)
├── scripts/          # Utility scripts
└── build/            # Build directory (created by CMake)
```

## Next Steps

1. **Read the Documentation**: Check out `docs/user_manual.md` for detailed usage
2. **Explore Examples**: Look at language-specific examples in `examples/`
3. **Train Models**: Create your own models and save them in `models/`
4. **Run Tests**: Verify everything works with `./scripts/run_tests.sh`
5. **Contribute**: See `CONTRIBUTING.md` for contribution guidelines

## Common Issues

### Issue: CMake can't find Eigen3

```bash
# Ubuntu/Debian
sudo apt-get install libeigen3-dev

# macOS
brew install eigen
```

### Issue: Python import errors

```bash
# Make sure you're in the project root
cd /path/to/misc_deepseekcode

# Install dependencies
pip install -r requirements.txt

# Add to PYTHONPATH
export PYTHONPATH=$PWD/examples/python:$PYTHONPATH
```

### Issue: Rust compilation errors

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

## Getting Help

- **Documentation**: `docs/user_manual.md`
- **Issues**: https://github.com/danindiana/misc_deepseekcode/issues
- **Examples**: `examples/` directory

## Minimum Working Example

The simplest way to test the framework:

```bash
# Build C++ (easiest to test)
mkdir build && cd build
cmake ..
make
./language_model
```

You should see:
```
Statistical Language Model initialized
Neural Network Language Model initialized
Optimized Response: Statistical response to: Hello, how are you?
```

Success! You're ready to explore the framework. 🎉
