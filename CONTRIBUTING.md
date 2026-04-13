# Contributing to TinkerRL

Thank you for your interest in contributing to TinkerRL! This document provides guidelines for contributing to the benchmark suite.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/tinker-rl-lab.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
ruff check .
ruff format --check .
```

## Adding a New RL Library Implementation

TinkerRL is designed to be extended with new RL libraries. To add one:

1. Create a new file: `experiments/implementations/<library>_<task>.py`
2. Follow the existing pattern:
   - Import seed utilities: `from utils.seed import set_global_seed, get_seed_from_args`
   - Define the environment (use ArithmeticEnv as template)
   - Implement training loop with metrics logging
   - Support `--seed` CLI argument
3. Add the library to `requirements.txt` under an optional section
4. Add a section in `experiments/README.md`
5. Submit a PR with:
   - The implementation file
   - Results from at least 3 seeds
   - Brief description of the library and why it's included

### Implementation Checklist
- [ ] Uses `set_global_seed()` for reproducibility
- [ ] Supports `--seed` CLI argument
- [ ] Logs metrics in JSONL format to `experiments/results/`
- [ ] Includes docstring with library description
- [ ] Matches Tinker hyperparameters where applicable
- [ ] Has been tested locally

## Adding New Tasks

1. Define the task in `experiments/tasks/`
2. Provide:
   - Environment specification
   - Verifiable reward function
   - Floor and ceiling baselines (see BASELINES.md)
   - At least one reference implementation
3. Document in BASELINES.md

## Code Style

- Python 3.9+ compatible
- Follow PEP 8 (enforced by ruff)
- Type hints for function signatures
- Docstrings for all public functions (Google style)
- Maximum line length: 100 characters

## Commit Messages

Use conventional commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation
- `test:` adding tests
- `refactor:` code restructuring
- `ci:` CI/CD changes

Example: `feat: add SAC implementation for continuous control task`

## Pull Request Process

1. Ensure all tests pass: `pytest tests/`
2. Ensure code is formatted: `ruff format .`
3. Update documentation if needed
4. Fill in the PR template
5. Request review from maintainers

## Reporting Issues

- Use GitHub Issues
- Include: Python version, OS, library versions
- Provide minimal reproduction steps
- Include error logs

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
