"""Smoke tests verifying experiment files are well-formed without importing them."""

import ast
from pathlib import Path


EXP_DIR = (
    Path(__file__).resolve().parents[1] / "platform_hybrid" / "experiments" / "implementations"
)


def experiment_files():
    files = sorted(EXP_DIR.glob("*.py"))
    assert files, f"No experiment Python files found in {EXP_DIR}"
    return files


def test_experiments_parse():
    """All experiment Python files parse without syntax errors."""
    errors = []
    for path in experiment_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path.name}: {error}")
    assert not errors, f"Syntax errors in experiments: {errors}"


def test_experiments_have_seed():
    """All TRL experiment files contain seed management."""
    files = [path for path in experiment_files() if "trl" in path.name]
    assert files, f"No TRL experiment Python files found in {EXP_DIR}"
    missing = [
        path.name for path in files if "seed" not in path.read_text(encoding="utf-8").lower()
    ]
    assert not missing, f"Experiments missing seed management: {missing}"


if __name__ == "__main__":
    test_experiments_parse()
    test_experiments_have_seed()
    print("All experiment tests passed")
