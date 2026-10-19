"""Offline experiment smoke/setup tests; training libraries are never loaded."""

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils.seed


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


def test_experiments_have_seed_inventory_markers():
    """Smoke inventory: source mentions seed; this does not establish RNG propagation."""
    files = [path for path in experiment_files() if "trl" in path.name]
    assert files, f"No TRL experiment Python files found in {EXP_DIR}"
    missing = [
        path.name for path in files if "seed" not in path.read_text(encoding="utf-8").lower()
    ]
    assert not missing, f"Experiments missing seed management: {missing}"


def test_gsm8k_import_finds_checkout_root_from_another_directory(tmp_path):
    """Import the actual script in an isolated interpreter without provider libraries."""
    script = EXP_DIR / "trl_gsm8k_math.py"
    probe = """
import runpy
import sys
from pathlib import Path
from types import ModuleType

for name, exports in {
    "torch": [],
    "wandb": [],
    "datasets": ["load_dataset"],
    "transformers": ["AutoModelForCausalLM", "AutoTokenizer"],
    "trl": ["GRPOTrainer", "GRPOConfig"],
}.items():
    module = ModuleType(name)
    for export in exports:
        setattr(module, export, object())
    sys.modules[name] = module

script = Path(sys.argv[1])
namespace = runpy.run_path(str(script), run_name="offline_import_probe")
import utils.seed
assert Path(utils.seed.__file__).resolve() == script.parents[3] / "utils" / "seed.py"
assert sys.path[0] == str(script.parents[3])
assert namespace["get_seed_from_args"] is utils.seed.get_seed_from_args
assert namespace["set_global_seed"] is utils.seed.set_global_seed
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(("cli_seed", "seed"), [(None, 42), (0, 0), (73, 73)])
def test_gsm8k_setup_propagates_cli_seed_before_loading(cli_seed, seed, monkeypatch):
    """Execute setup with doubles; never import training libraries or fit a model."""
    tree = ast.parse((EXP_DIR / "trl_gsm8k_math.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    config_index = next(
        i
        for i, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "grpo_config"
    )
    main.body = main.body[: config_index + 1] + [
        ast.Return(value=ast.Name(id="grpo_config", ctx=ast.Load()))
    ]
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "utils.seed"
    ]
    setup = ast.fix_missing_locations(ast.Module(body=[*imports, main], type_ignores=[]))
    events = []

    def set_seed(value):
        events.append(("seed", value))

    def loaded(name, value):
        def load(*args, **kwargs):
            assert events[0] == ("seed", seed)
            events.append((name, kwargs))
            return value

        return load

    argv = ["trl_gsm8k_math.py"]
    if cli_seed is not None:
        argv.extend(["--seed", str(cli_seed)])
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(utils.seed, "set_global_seed", set_seed)
    ns = {
        "os": os,
        "torch": SimpleNamespace(bfloat16="fake_bfloat16"),
        "wandb": SimpleNamespace(init=loaded("wandb", None)),
        "AutoTokenizer": SimpleNamespace(
            from_pretrained=loaded("tokenizer", SimpleNamespace(eos_token="<eos>"))
        ),
        "AutoModelForCausalLM": SimpleNamespace(from_pretrained=loaded("model", object())),
        "load_gsm8k_dataset": loaded("dataset", []),
        "GRPOConfig": lambda **kwargs: kwargs,
    }
    exec(compile(setup, "<GSM8K setup fixture>", "exec"), ns)  # noqa: S102 - Execute checked-in setup with offline doubles only.
    config = ns["main"]()
    assert events[0] == ("seed", seed)
    assert config["seed"] == seed
    assert config["data_seed"] == seed
    assert dict(events)["wandb"]["config"]["seed"] == seed


if __name__ == "__main__":
    test_experiments_parse()
    test_experiments_have_seed_inventory_markers()
    print("All experiment tests passed")
