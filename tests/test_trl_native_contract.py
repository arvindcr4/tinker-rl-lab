from pathlib import Path
import ast
import pytest
from platform_local.trl_integrations.config import TRLConfig
from platform_local.trl_integrations.trainer import generate_trl_train_script


def test_no_implicit_reward(tmp_path):
    with pytest.raises(ValueError, match="reward_callable"):
        generate_trl_train_script(
            TRLConfig(data={"train_data": ["train.json"]}), tmp_path / "train.py"
        )


def test_generated_settings_are_not_silently_dropped(tmp_path):
    config = TRLConfig(
        data={"train_data": ["train.json"], "max_response_length": 73},
        algorithm={"kl_coef": 0.123},
        optimizer={"beta2": 0.876},
        reward_callable="somewhere:reward",
        output_dir=str(tmp_path / "run"),
    )
    script = generate_trl_train_script(config, tmp_path / "train.py")
    tree = ast.parse(script)
    call = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "trainer_config" for t in n.targets)
    )
    kwargs = {k.arg: k.value for k in call.keywords}
    assert ast.literal_eval(kwargs["beta"]) == 0.123
    assert ast.literal_eval(kwargs["adam_beta2"]) == 0.876
    assert ast.literal_eval(kwargs["max_completion_length"]) == 73
    assert "get_last_checkpoint" not in script


import sys
import types
from importlib import metadata
from platform_local.trl_integrations.trainer import prepare_trl_run
from platform_local.unified.launcher import UnifiedLauncher


def native_config(tmp_path, **kwargs):
    data = tmp_path / "train.json"
    data.write_text('[ {"prompt": "synthetic"} ]')
    return TRLConfig(
        data={"train_data": [str(data)]},
        reward_callable="synthetic_rewards:reward",
        output_dir=str(tmp_path / "native-output"),
        **kwargs,
    )


@pytest.mark.parametrize("setting", ["reward_callable", "output_dir"])
def test_native_contract_requires_explicit_settings(tmp_path, setting):
    config = native_config(tmp_path)
    setattr(config, setting, None)
    with pytest.raises(ValueError, match=setting):
        generate_trl_train_script(config, tmp_path / "generated.py")
    assert not (tmp_path / "generated.py").exists()


@pytest.mark.parametrize("path", ["reward", "module:", ":reward", "a:reward.nested", "a-b:reward"])
def test_invalid_reward_reference_rejected_before_writing(tmp_path, path):
    config = native_config(tmp_path)
    config.reward_callable = path
    with pytest.raises(ValueError, match="module:attribute"):
        generate_trl_train_script(config, tmp_path / "generated.py")
    assert not (tmp_path / "generated.py").exists()


def test_unsupported_prompt_cap_and_evaluation_are_not_ignored(tmp_path):
    config = native_config(tmp_path)
    config.data.max_prompt_length = 512
    with pytest.raises(NotImplementedError, match="max_prompt_length"):
        generate_trl_train_script(config, tmp_path / "generated.py")
    config.data.max_prompt_length = None
    config.data.val_data = ["eval.json"]
    with pytest.raises(NotImplementedError, match="val_data"):
        generate_trl_train_script(config, tmp_path / "generated.py")


def install_reward(monkeypatch, value):
    module = types.ModuleType("synthetic_rewards")
    module.reward = value
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(metadata, "version", lambda name: "1.2.0")


def test_output_reservation_never_reuses_or_resumes(monkeypatch, tmp_path):
    reward = lambda **kwargs: [0]
    install_reward(monkeypatch, reward)
    output = tmp_path / "new"
    fn, path = prepare_trl_run("synthetic_rewards:reward", str(output))
    assert fn is reward
    assert Path(path) == output.resolve()
    sentinel = output / "checkpoint-999"
    sentinel.mkdir()
    with pytest.raises(FileExistsError, match="resume is unsupported"):
        prepare_trl_run("synthetic_rewards:reward", str(output))
    assert sentinel.exists()


def test_invalid_callable_and_version_leave_output_uncreated(monkeypatch, tmp_path):
    install_reward(monkeypatch, 12)
    output = tmp_path / "new"
    with pytest.raises(TypeError, match="callable"):
        prepare_trl_run("synthetic_rewards:reward", str(output))
    assert not output.exists()
    monkeypatch.setattr(metadata, "version", lambda name: "0.99")
    with pytest.raises(RuntimeError, match="trl==1.2.0"):
        prepare_trl_run("synthetic_rewards:reward", str(output))
    assert not output.exists()


def inert_runtime(monkeypatch, observed, reward):
    """Every generated script dependency is inert; no model/provider is imported."""
    install_reward(monkeypatch, reward)

    def model_load(*args, **kwargs):
        observed["model_loads"] = observed.get("model_loads", 0) + 1
        return object()

    class Trainer:
        def __init__(self, **kwargs):
            observed["trainer"] = kwargs
            self.state = types.SimpleNamespace(global_step=max(1, kwargs["args"].max_steps))

        def train(self, **kwargs):
            observed["train"] = kwargs

    def cfg(**kwargs):
        observed["config"] = kwargs
        return types.SimpleNamespace(**kwargs)

    def module(name, **attrs):
        obj = types.ModuleType(name)
        obj.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, obj)

    loader = types.SimpleNamespace(from_pretrained=model_load)
    module("torch", bfloat16="bf16", float16="fp16", float32="fp32")
    module(
        "transformers",
        AutoTokenizer=loader,
        AutoModelForCausalLM=loader,
        BitsAndBytesConfig=lambda **kwargs: kwargs,
        set_seed=lambda seed: observed.update(seed=seed),
    )
    module("datasets", load_dataset=lambda *a, **k: ["synthetic-row"])
    module("trl", GRPOConfig=cfg, GRPOTrainer=Trainer)
    module(
        "platform_local.unified.peft_utils",
        apply_peft_method=lambda model, **kw: model,
        save_bitfit_checkpoint=lambda *a, **k: None,
    )


def test_generated_execution_passes_explicit_reward_and_config(monkeypatch, tmp_path):
    observed = {}
    reward = lambda completions, **kwargs: [-1] * len(completions)
    inert_runtime(monkeypatch, observed, reward)
    config = native_config(
        tmp_path,
        report_to="none",
        seed=913,
        algorithm={"kl_coef": 0.12, "num_generations": 4},
        optimizer={"beta1": 0.8, "beta2": 0.87, "eps": 0.0001, "weight_decay": 0.2},
    )
    script = generate_trl_train_script(config, tmp_path / "generated.py")
    exec(compile(script, "<inert-generated>", "exec"), {})  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.
    assert observed["trainer"]["reward_funcs"] == [reward]
    assert observed["train"] == {"resume_from_checkpoint": None}
    assert observed["seed"] == 913
    actual = observed["config"]
    assert actual["beta"] == 0.12
    assert actual["num_generations"] == 4
    assert actual["adam_beta1"] == 0.8
    assert actual["adam_beta2"] == 0.87
    assert actual["adam_epsilon"] == 0.0001
    assert actual["weight_decay"] == 0.2
    assert actual["output_dir"] == str((tmp_path / "native-output").resolve())
    assert actual["seed"] == 913
    assert "prm_reward_fn" not in script
    assert "advantage_collapse_rate" not in script


def test_existing_output_fails_before_any_model_load(monkeypatch, tmp_path):
    observed = {}
    inert_runtime(monkeypatch, observed, lambda **k: [])
    config = native_config(tmp_path, report_to="none")
    Path(config.output_dir).mkdir()
    script = generate_trl_train_script(config, tmp_path / "generated.py")
    with pytest.raises(FileExistsError):
        exec(compile(script, "<inert-generated>", "exec"), {})  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.
    assert not observed


def test_launcher_does_not_report_epochs_as_observed_steps(monkeypatch, tmp_path):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    launcher = UnifiedLauncher()
    launcher.framework = "trl"
    launcher.model = "synthetic-model"
    launcher.epochs = 9
    launcher.train_data = ["synthetic.json"]
    launcher.trl_config = native_config(tmp_path)
    with pytest.raises(RuntimeError, match="receipt"):
        launcher._run_trl()


def test_actual_cli_to_generated_script_preserves_native_contract(monkeypatch, tmp_path):
    import subprocess
    from platform_local.unified.__main__ import main
    from platform_local.unified.canonical import CanonicalSpec

    observed = {}
    reward = lambda **kwargs: []
    inert_runtime(monkeypatch, observed, reward)
    wandb = types.ModuleType("wandb")
    wandb.init = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    def execute_inert_command(command, **kwargs):
        script = Path(command[1]).read_text()
        with monkeypatch.context() as env_patch:
            env_patch.setenv("TRL_RECEIPT_PATH", kwargs["env"]["TRL_RECEIPT_PATH"])
            exec(compile(script, "<inert-cli-generated>", "exec"), {})  # noqa: S102 - Execute only checked-in helper definitions to isolate optional imports.

    monkeypatch.setattr(subprocess, "run", execute_inert_command)
    (tmp_path / "synthetic.json").write_text('[ {"prompt": "synthetic"} ]')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unified",
            "--framework",
            "trl",
            "--train-data",
            "synthetic.json",
            "--reward-callable",
            "synthetic_rewards:reward",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    monkeypatch.setattr(UnifiedLauncher, "_resolve_spec", lambda self: CanonicalSpec())
    main()
    assert observed["trainer"]["reward_funcs"] == [reward]
    assert observed["config"]["output_dir"] == str((tmp_path / "run").resolve())
    assert observed["config"]["seed"] == CanonicalSpec().seed
    assert observed["config"]["beta"] == CanonicalSpec().beta
    assert (
        observed["config"]["gradient_accumulation_steps"]
        == CanonicalSpec().gradient_accumulation_steps
    )


def test_native_model_path_does_not_trigger_implicit_resume(tmp_path):
    config = native_config(tmp_path, model={"model_path": "previous-checkpoint"})
    with pytest.raises(NotImplementedError, match="model_path/resume"):
        generate_trl_train_script(config, tmp_path / "script.py")


def test_distributed_output_reservation_fails_explicitly(monkeypatch, tmp_path):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(NotImplementedError, match="one process"):
        prepare_trl_run("synthetic_rewards:reward", str(tmp_path / "run"))
    assert not (tmp_path / "run").exists()
    with pytest.raises(NotImplementedError, match="one process/GPU"):
        generate_trl_train_script(native_config(tmp_path, num_gpus=2), tmp_path / "script.py")


def test_explicit_scientific_contract_preserves_manifest_and_reward(tmp_path):
    config = TRLConfig(execution_contract="scientific", data={"train_data": ["synthetic.json"]})
    script = generate_trl_train_script(config, tmp_path / "scientific.py")
    ast.parse(script)
    assert "correctness_rewards(completions, answer)" in script
    assert "Checkpoint manifest differs" in script
    assert "trainer.state.global_step" in script
    assert "if len(tokens) > 512:" in script
    assert "prepare_trl_run(" not in script


def test_generation_aliases_share_one_budget_and_reject_conflicts():
    from pydantic import ValidationError

    for config in (TRLConfig(num_generations=4), TRLConfig(algorithm={"num_generations": 4})):
        assert config.num_generations == config.algorithm.num_generations == 4
    with pytest.raises(ValidationError, match="aliases must agree"):
        TRLConfig(num_generations=4, algorithm={"num_generations": 8})


def test_native_explicit_resume_is_rejected_before_script_write(tmp_path):
    config = native_config(tmp_path, resume_from_checkpoint="old/checkpoint-3")
    with pytest.raises(NotImplementedError, match="resume_from_checkpoint"):
        generate_trl_train_script(config, tmp_path / "script.py")
    assert not (tmp_path / "script.py").exists()


@pytest.mark.parametrize("step", [False, 0, -1, 2, 4.0])
def test_launcher_rejects_invalid_or_short_optimizer_receipts(monkeypatch, tmp_path, step):
    import json
    import subprocess

    def run(command, **kwargs):
        Path(kwargs["env"]["TRL_RECEIPT_PATH"]).write_text(
            json.dumps(
                {
                    "status": "completed",
                    "simulated": False,
                    "final_step": step,
                }
            )
        )

    monkeypatch.setattr(subprocess, "run", run)
    launcher = UnifiedLauncher()
    launcher.train_data = ["synthetic.json"]
    launcher.trl_config = native_config(tmp_path, max_steps=4)
    with pytest.raises(RuntimeError, match="receipt"):
        launcher._run_trl()


@pytest.mark.parametrize(("batch", "accum", "generations"), [(3, 1, 8), (2, 2, 3)])
def test_grpo_batch_geometry_rejected_at_configuration(batch, accum, generations):
    with pytest.raises(ValueError, match="divisible by num_generations"):
        TRLConfig(
            data={"train_batch_size": batch, "gradient_accumulation_steps": accum},
            algorithm={"num_generations": generations},
        )


def test_in_process_grpo_forwards_sampling_settings(monkeypatch):
    import sys
    from types import SimpleNamespace
    from platform_local.trl_integrations.trainer import create_grpo_trainer

    captured = {}

    def configuration(**kwargs):
        captured.update(kwargs)
        return kwargs

    monkeypatch.setitem(
        sys.modules,
        "trl",
        SimpleNamespace(GRPOConfig=configuration, GRPOTrainer=lambda **kwargs: kwargs),
    )
    config = TRLConfig(algorithm={"num_generations": 4, "temperature": 0.7})
    create_grpo_trainer(object(), object(), [], [], config)
    assert captured["num_generations"] == 4
    assert captured["temperature"] == 0.7
