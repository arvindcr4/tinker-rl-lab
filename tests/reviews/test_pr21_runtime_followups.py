"""Offline reproductions for PR21 runtime integration comments."""

import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, **namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    for n in nodes:
        n.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)  # noqa: S102 -- execute selected local definitions only
    return SimpleNamespace(**namespace)


def test_curriculum_summary_consumes_renamed_result():
    tree = ast.parse((ROOT / "platform_hybrid/experiments/openings/curriculum_grpo.py").read_text())
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "k"
    ]
    summary = {}
    result = dict(
        near_zero_scalar_loss_fraction=0.1,
        oversample_factor=2,
        heldout_before=0.2,
        heldout_after=0.3,
        heldout_gain=0.1,
    )
    for loop in loops:
        exec(  # noqa: S102 -- isolated local summary consumer
            compile(ast.Module(body=[loop], type_ignores=[]), "<summary>", "exec"),
            {"run": SimpleNamespace(summary=summary), "mode": "baseline", "r": result},
        )
    assert summary["baseline/near_zero_scalar_loss_fraction"] == 0.1


@pytest.mark.parametrize("platform", ["modal", "google_colab"])
def test_corrected_manifest_hardware(platform):
    m = functions("platform_colab/publication_fixes/v2_final_analysis.py", ["hardware_label"])
    assert (
        m.hardware_label(
            {"execution_environment": platform, "manifest": {"hardware": {"gpu_names": ["A10G"]}}}
        )
        == f"A10G ({platform})"
    )
    assert m.hardware_label({"env": {"gpu": "A100", "platform": "legacy"}}) == "A100 (legacy)"
    for bad in (None, [], [""], [2]):
        with pytest.raises(ValueError, match="provenance"):
            m.hardware_label(
                {"execution_environment": platform, "manifest": {"hardware": {"gpu_names": bad}}}
            )


def test_every_sweep_config_has_explicit_reward_version():
    root = ROOT / "platform_hybrid/experiments/tool_use_zvf_sweep"
    m = functions(
        str(root.relative_to(ROOT) / "run_sweep.py"),
        ["reward_version_from_path"],
        yaml=yaml,
        Path=Path,
    )
    paths = list((root / "configs").glob("*.yaml"))
    assert len(paths) >= 14
    for path in paths:
        assert m.reward_version_from_path(path) in ("v1", "v2")
    tree = ast.parse((root / "gen_configs.py").read_text())
    template = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TEMPLATE" for t in n.targets)
    )
    assert 'reward_version: "{reward}"' in template


def test_reward_digest_binds_delegated_grader(tmp_path):
    m = functions(
        "platform_hybrid/experiments/tool_use_zvf_sweep/run_sweep.py",
        ["reward_source_manifest"],
        Path=Path,
        hashlib=hashlib,
        json=json,
    )
    wrapper = tmp_path / "tinker_atropos/environments/tool_use_tinker.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("wrapper")
    with pytest.raises(FileNotFoundError):
        m.reward_source_manifest(tmp_path)
    grader = tmp_path / "tinker_atropos/grading.py"
    grader.write_text("grader-v1")
    old = m.reward_source_manifest(tmp_path)
    grader.write_text("grader-v2")
    assert m.reward_source_manifest(tmp_path)["sha256"] != old["sha256"]


def test_math500_uses_boxed_math_grader():
    from platform_hybrid.experiments.research_contracts import math_reward

    m = functions(
        "platform_hybrid/experiments/p1b_audit_probe.py",
        ["reward_math500"],
        math_reward=math_reward,
    )
    assert m.reward_math500(r"Answer: \boxed{2\sqrt{2}}", r"2\sqrt{2}") == 1
    assert m.reward_math500("2", "2") == 0
    tree = ast.parse((ROOT / "platform_hybrid/experiments/p1b_audit_probe.py").read_text())
    tasks = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in n.targets)
    )
    value = next(
        v for k, v in zip(tasks.keys, tasks.values, strict=True) if ast.literal_eval(k) == "math500"
    )
    assert value.elts[1].id == "reward_math500"


def test_skyrl_launcher_recipe_matches_requested_model():
    from platform_hybrid.experiments.framework_contracts import read_skyrl_recipe

    source = read_skyrl_recipe(ROOT / "platform_hybrid/skyrl/configs/grpo_gsm8k_qwen3_8b.yaml")
    assert source["requested_config"]["trainer"]["policy"]["model"]["path"] == "Qwen/Qwen3-8B"
    with pytest.raises(ValueError, match="does not match"):
        read_skyrl_recipe(ROOT / "platform_hybrid/skyrl/configs/grpo_gsm8k.yaml")


def test_scalar_only_colab_cache_is_rejected(tmp_path):
    path = ROOT / "platform_colab/publication_fixes/samestack_v2_colab.py"
    spec = importlib.util.spec_from_file_location("colab_cache_test", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    target = tmp_path / "samestack-colab-v2-validated-reuse/smoke"
    target.mkdir(parents=True)
    cached = dict(
        protocol="samestack-colab-v2-validated-reuse",
        arm="ppo",
        seed=1,
        smoke=True,
        model=m.MODEL,
        n_steps=2,
        n_eval=32,
        group=1,
        n_gen=m.N_GEN,
        lr=m.LR,
        max_new=m.MAX_NEW,
        k_epochs=m.K_EPOCHS,
    )
    (target / "ppo_s1.json").write_text(json.dumps(cached))
    with gzip.open(target / "ppo_s1_texts.json.gz", "wt") as f:
        json.dump({"pre": ["x"] * 32, "post": ["x"] * 32}, f)
    with pytest.raises(ValueError, match="verified adapter"):
        m.run_arm("ppo", 1, str(tmp_path), smoke=True)


def test_wandb_summary_endpoints_are_promoted(monkeypatch):
    import pandas as pd

    run = SimpleNamespace(
        name="test",
        group="g",
        state="finished",
        config={"heldout_before": 999},
        summary={"heldout_before": 0.2, "heldout_after": 0.3, "evaluation_ids_hash": "paired"},
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=lambda: SimpleNamespace(runs=lambda _: [run]), _vram_patched=True),
    )
    m = functions(
        "platform_hybrid/experiments/10x_structural_ceiling/analyze_results.py",
        ["load_wandb_runs"],
        pd=pd,
    )
    result = m.load_wandb_runs().iloc[0]
    assert result["heldout_before"] == 0.2
    assert result["heldout_after"] == 0.3
    assert result["evaluation_ids_hash"] == "paired"
