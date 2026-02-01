# Contributing to Neo-Cortex

First off, thank you for considering contributing to Neo-Cortex! It's people like you that make this project better for everyone.

## Code of Conduct

By participating in this project, you agree to maintain a welcoming, inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, config files)
- **Describe the behavior you observed and what you expected**
- **Include your environment** (OS, Python version, dependencies)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List any alternatives you've considered**

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Make your changes** following the code style below
3. **Add tests** if you've added code that should be tested
4. **Ensure tests pass** with `python -m pytest tests/`
5. **Update documentation** if needed
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/NeoCortex.git
cd NeoCortex

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies

# Run tests
python -m pytest tests/
```

## Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for function signatures
- **Formatting**: Use `black` for formatting, `isort` for imports

```bash
# Format code
black service/
isort service/

# Check types
mypy service/
```

## Project Structure

```
NeoCortex/
├── service/           # Core Python package
│   ├── cortex_engine.py    # Main engine
│   ├── shared_engine.py    # Shared Memory
│   ├── session_engine.py   # Session Continuity
│   ├── health_engine.py    # Memory Health
│   ├── mcp_server.py       # MCP server
│   ├── api_server.py       # REST API
│   └── storage/            # Storage backends
├── tests/             # Test suite
├── docs/              # Documentation
└── assets/            # Images, banners
```

## Adding a New Feature

1. **Discuss first**: Open an issue to discuss major changes
2. **Branch naming**: Use `feature/`, `fix/`, `docs/` prefixes
3. **Small commits**: Make atomic commits with clear messages
4. **Tests required**: All new features need tests
5. **Documentation**: Update docs for user-facing changes

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=service --cov-report=html

# Run specific test file
python -m pytest tests/test_village.py

# Run specific test
python -m pytest tests/test_village.py::test_post_message
```

## Documentation

- **README.md**: High-level overview and quick start
- **ARCHITECTURE.md**: Technical design and internals
- **docs/API.md**: REST API reference
- **docs/TOOLS.md**: Tool schemas for function calling
- **Docstrings**: All public functions need docstrings

## Commit Messages

Follow conventional commits:

```
feat: add convergence detection to shared memory
fix: handle empty search results gracefully
docs: update REST API documentation
test: add tests for memory health engine
refactor: simplify storage adapter interface
```

## Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting)
3. Documentation must be updated if needed
4. Breaking changes need discussion

## Questions?

Feel free to open an issue with the `question` label or reach out to the maintainers.

---

Thank you for contributing!
