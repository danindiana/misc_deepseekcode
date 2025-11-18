# Contributing to Inter-System Communication Language

Thank you for your interest in contributing to the Inter-System Communication Language project! We welcome contributions from the community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/misc_deepseekcode.git
   cd misc_deepseekcode
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/danindiana/misc_deepseekcode.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new functionality
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance**: Optimize existing code
- **Examples**: Add new usage examples

### Finding Issues to Work On

- Check the [Issues](https://github.com/danindiana/misc_deepseekcode/issues) page
- Look for issues tagged with `good-first-issue` or `help-wanted`
- Comment on the issue to let others know you're working on it

## Development Setup

### Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# C++ dependencies (Ubuntu/Debian)
sudo apt-get install build-essential cmake libeigen3-dev

# Rust dependencies
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Erlang dependencies
sudo apt-get install erlang
```

### Build the Project

```bash
# Build all components
make all

# Or build specific components
make python
make cpp
make rust
make erlang
```

## Coding Standards

### Python

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://github.com/psf/black) for code formatting:
  ```bash
  black examples/python/ src/python/
  ```
- Use type hints where appropriate
- Maximum line length: 88 characters (Black default)

**Example:**
```python
def generate_response(self, input_text: str) -> str:
    """
    Generate a response for the given input text.

    Args:
        input_text: The input text to process

    Returns:
        Generated response string
    """
    # Implementation here
    pass
```

### C++

- Follow the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Use C++20 features where appropriate
- Use smart pointers instead of raw pointers
- Maximum line length: 100 characters

**Example:**
```cpp
class LanguageModel {
public:
    virtual ~LanguageModel() = default;

    /**
     * Generate a response for the given input text.
     *
     * @param inputText The input text to process
     * @return Generated response string
     */
    virtual std::string generateResponse(const std::string& inputText) = 0;
};
```

### Rust

- Follow the [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/)
- Use `cargo fmt` for formatting:
  ```bash
  cargo fmt
  ```
- Use `cargo clippy` for linting:
  ```bash
  cargo clippy -- -D warnings
  ```

**Example:**
```rust
/// Generate a response for the given input text.
///
/// # Arguments
///
/// * `input_text` - The input text to process
///
/// # Returns
///
/// Generated response string
pub fn generate_response(&self, input_text: &str) -> String {
    // Implementation here
}
```

### Erlang

- Follow [Erlang/OTP Design Principles](https://www.erlang.org/doc/design_principles/users_guide.html)
- Use meaningful variable names (CamelCase for variables)
- Maximum line length: 80 characters

**Example:**
```erlang
%% Generate a response for the given input text
%%
%% Args:
%%   InputText: The input text to process
%%
%% Returns:
%%   Generated response string
generate_response(InputText) ->
    % Implementation here
    ok.
```

## Testing

### Writing Tests

All new features and bug fixes should include tests.

#### Python Tests

```bash
# Run tests
pytest tests/python/ -v

# Run with coverage
pytest tests/python/ --cov=examples/python --cov-report=html
```

**Example test:**
```python
def test_generate_response():
    """Test response generation"""
    model = LanguageModel('statistical', 'path/to/model.pkl')
    response = model.generate_response("Hello")
    assert isinstance(response, str)
    assert len(response) > 0
```

#### C++ Tests

```bash
# Build and run tests
cd build
cmake ..
make
ctest
```

#### Rust Tests

```bash
# Run tests
cargo test

# Run with output
cargo test -- --nocapture
```

### Test Coverage

- Aim for at least 80% code coverage
- All public APIs must have tests
- Edge cases should be tested

## Documentation

### Code Documentation

- All public functions/methods must have docstrings
- Include parameter descriptions and return values
- Provide usage examples where appropriate

### User Documentation

- Update `docs/user_manual.md` for user-facing changes
- Update `docs/architecture.md` for architectural changes
- Add examples to `examples/` directory

### README Updates

- Update README.md if adding new features
- Update installation instructions if dependencies change
- Add new examples to the Quick Start section

## Pull Request Process

### Before Submitting

1. **Update your fork**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   make test
   ```

3. **Check code style**:
   ```bash
   make lint
   make format
   ```

4. **Update documentation** as needed

### Submitting the Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub

3. **Fill out the PR template** with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if applicable)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings introduced
```

### Review Process

- All PRs require at least one review
- Address reviewer comments promptly
- Keep the PR focused on a single feature/fix
- Update the PR based on feedback

## Issue Reporting

### Bug Reports

Include the following information:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Detailed steps
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**:
  - OS and version
  - Language version
  - Relevant dependencies
- **Error Messages**: Full error messages and stack traces
- **Screenshots**: If applicable

### Feature Requests

Include the following:

- **Problem**: What problem does this solve?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions considered
- **Additional Context**: Any other relevant information

## Community

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and feature requests
- **Pull Requests**: Follow the PR process above

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- The contributors page

## Questions?

If you have questions, please:
1. Check existing documentation
2. Search existing issues
3. Create a new discussion on GitHub
4. Reach out to maintainers

Thank you for contributing! 🎉
