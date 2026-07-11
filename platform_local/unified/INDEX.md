# unified/ — INDEX

**Purpose:** Common smoke-test launcher for integrated frameworks (skyrl, platform_tinker/atropos, verl, platform_modal/openrlhf, trl), plus a path for generating a runnable TRL GRPO script.

**Key files:**
- `__main__.py` — supported CLI interface and TRL script-generation path.
- `launcher.py` — framework dispatch and `TrainingResult` definitions.
- `peft_utils.py` — PEFT/BitFit configuration, model wrapping, and compact BitFit checkpoint helpers.

**Find it fast:**
- to validate framework dispatch → `python -m platform_local.unified --framework <skyrl|tinker|verl|openrlhf|trl> --model <m> --algorithm <grpo|ppo|...>`
- to generate runnable TRL GRPO code → add `--train-data <json> --generate-script <path>`
- to add a new framework → extend `FRAMEWORKS` + add a `_run_<name>` method in `launcher.py`
