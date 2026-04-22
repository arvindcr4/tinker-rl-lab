# BrowserGym/WebArena with Tinker

This path trains browser-control agents through the existing Atropos + Tinker
GRPO loop:

`Tinker trainer -> Atropos BrowserGymEnv -> Chromium BrowserGym task -> reward`

The current wrapper asks the model for a short one-shot BrowserGym action script,
executes those actions in Chromium, and logs benchmark reward plus browser
success metrics. This is intentionally a smoke-ready bridge; a later version can
move to fully reactive per-action sampling.

## Install BrowserGym dependencies

From the repo root:

```bash
uv pip install --python .venv-at/bin/python -e "atropos[browser]"

. .venv-at/bin/activate
python -m playwright install chromium
python -c "import nltk; nltk.download('punkt_tab')"
```

## MiniWoB smoke

MiniWoB needs the MiniWoB++ HTML assets. The helper keeps them outside the repo
and pins the same commit recommended by BrowserGym.

```bash
bash experiments/setup_browsergym_miniwob.sh
export MINIWOB_URL="file://$HOME/.cache/tinker-rl-lab/miniwob-plusplus/miniwob/html/miniwob/"
```

Run the Tinker smoke:

```bash
cd atropos
export TINKER_API_KEY="$TINKER_API_KEY"
export WANDB_API_KEY="$WANDB_API_KEY"
./run_experiment_generic.sh browsergym_tinker configs/browsergym_miniwob_qwen_8b_smoke.yaml
```

Expected W&B metrics include:

- `train/browser_success_rate`
- `train/browser_reward_mean`
- `train/browser_action_count_mean`
- `eval/browser_success_rate`
- `eval/browser_reward_mean`
- `eval/browser_action_count_mean`

## WebArena smoke

WebArena requires the benchmark sites to already be running. Configure the
BrowserGym `WA_*` URLs to match that deployment:

```bash
BASE_URL="http://your-webarena-host"
export WA_SHOPPING="$BASE_URL:8082/"
export WA_SHOPPING_ADMIN="$BASE_URL:8083/admin"
export WA_REDDIT="$BASE_URL:8080"
export WA_GITLAB="$BASE_URL:9001"
export WA_WIKIPEDIA="$BASE_URL:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:443"
export WA_HOMEPAGE="$BASE_URL:80"
export WA_FULL_RESET=""
export OPENAI_API_KEY="$OPENAI_API_KEY"
```

Then run:

```bash
cd atropos
export TINKER_API_KEY="$TINKER_API_KEY"
export WANDB_API_KEY="$WANDB_API_KEY"
./run_experiment_generic.sh browsergym_tinker configs/browsergym_webarena_qwen_8b_smoke.yaml
```

`OPENAI_API_KEY` and the NLTK `punkt_tab` resource are needed because some
WebArena evaluators use an LLM-based fuzzy match. MiniWoB does not need either.

## Files

- `tinker_atropos/environments/browsergym_tinker.py`
- `configs/browsergym_miniwob_qwen_8b_smoke.yaml`
- `configs/browsergym_webarena_qwen_8b_smoke.yaml`
- `../experiments/setup_browsergym_miniwob.sh`
