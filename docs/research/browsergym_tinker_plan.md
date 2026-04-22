# BrowserGym × Tinker — Integration Plan

_Research date: 2026-04-22._
_Sources: `ServiceNow/BrowserGym`, `thinking-machines-lab/tinker-cookbook`, tinker-docs, `web-arena-x/webarena`, `Farama-Foundation/miniwob-plusplus` (all Nia-indexed; see `nia-sources.md`)._

## TL;DR

Use **BrowserGym/MiniWoB** for the smoke loop, **WorkArena-L1** as the dev target, and **WebArena-verified** or **AssistantBench** for the paper-quality eval. Wire it into Tinker via the **`verifiers_rl` override pattern** — override `train.do_group_rollout` with a function that runs episodes through a `TinkerAsyncOpenAIClient` and converts results into a `TrajectoryGroup`. Drive observations through **AXTree-only** for ≤32k context; fall back to `pruned_html` only for tasks that need attributes.

## 1. Benchmark choice

| Benchmark | Max steps | Parallel seeds | Multi-tab | Cost to run | Recommendation |
|---|---|---|---|---|---|
| **MiniWoB** (`browsergym/miniwob.*`) | 10 | ✅ | ❌ | Local HTML, $0 | **Smoke loop.** Fastest iteration; ~125 tasks. |
| **WorkArena-L1** (`browsergym/workarena.*`) | 15 | ✅ | ❌ | ServiceNow dev instance | **Dev target.** Realistic enterprise flows, reproducible seeds. |
| **WorkArena-L2/L3** | 50 | ✅ | ✅ | ServiceNow | For agentic planning ablations. |
| **WebArena** (`browsergym/webarena.*`) | 30 | ❌ | ✅ | 5 Docker containers (shopping, reddit, gitlab, cms, map) | Paper eval; 812 tasks. |
| **WebArena-verified** | 30 | ❌ | ✅ | Same as WebArena | Preferred over raw WebArena (known task-label bugs fixed). |
| **VisualWebArena** | 30 | ❌ | ✅ | Vision-required | Only if using a VLM. |
| **AssistantBench** | 30 | ✅ | ✅ | Public web (flaky!) | Good real-web generalization eval. |

**Action for this repo**: the two smoke configs already committed (`browsergym_miniwob_qwen_8b_smoke.yaml`, `browsergym_webarena_qwen_8b_smoke.yaml`) are the right shape. Add a WorkArena-L1 config next.

## 2. MiniWoB setup (cheapest path)

```sh
# In BrowserGym root (already vendored)
make setup-miniwob            # clones + pins commit 7fd85d71; writes MINIWOB_URL to .env
source .env
```

Manual fallback:
```sh
git clone git@github.com:Farama-Foundation/miniwob-plusplus.git
git -C miniwob-plusplus reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
export MINIWOB_URL="http://localhost:8000/miniwob/"
# In one shell:
cd miniwob-plusplus/miniwob/html && python -m http.server 8000
```

Prefer **HTTP server over `file://`** — Chromium's CORS on `file://` breaks a subset of MiniWoB tasks (score silently drops).

## 3. BrowserGym action & observation contracts

**Action space** is the `HighLevelActionSet` with categories `{chat, infeas, bid, coord, nav, tab}` — use `bid`-only for MiniWoB, `bid + nav + tab` for WebArena. One action per `env.step(action_str)`; script is a list of stringified calls: `["click('a46')", "fill('a2164', 'Asset tag')"]`.

**Observation** (`obs` dict): after `default_obs_preprocessor`:
- `axtree_txt` — semantic, smallest, **default**.
- `pruned_html` — medium; needed for attribute-heavy tasks (forms).
- `dom_txt` — full; almost never worth it.
- `screenshot`, `chat_messages`, `goal_object`, `open_pages_*`, `last_action_error`.

**Reward** is task-specific via `_task_validate()`; MiniWoB and WebArena both return a scalar in `[0,1]`, terminal only (no dense per-step reward). Plan the RL reward around this — `max_reward` across the episode is a reasonable proxy if you want partial credit (see current `_run_browsergym_episode`).

**Recommendation**: prompt with `axtree_txt` + last action + last-action error; cap at ~24k tokens; keep screenshot off unless the base model is a VLM.

## 4. Tinker integration pattern

The clean pattern in the cookbook is **`verifiers_rl`** (`tinker_cookbook/recipes/verifiers_rl/`). It overrides `train.do_group_rollout` instead of shoehorning BrowserGym into Tinker's token-level `Env` interface:

```python
async def custom_do_group_rollout(builder, policy):
    sampling_client = cast(TinkerTokenCompleter, policy).sampling_client
    if shared_client is None:
        shared_client = TinkerAsyncOpenAIClient(sampling_client, renderer, tokenizer)
    else:
        shared_client.set_sampling_client(sampling_client)   # refresh after each policy step

    gen_sem = await maybe_semaphore(cfg.max_concurrent_generation)
    states = await bg_builder.run_group(
        group_inputs=builder.get_rollout_inputs(cfg.group_size),
        client=shared_client,
        gen_sampling_args={"max_tokens": cfg.max_tokens, "temperature": cfg.temperature},
        gen_sem=gen_sem,
    )
    return convert_states_to_trajectory_group(states)

train.do_group_rollout = custom_do_group_rollout
```

`convert_states_to_trajectory_group` turns each episode into a list of `Transition(ob, ac, reward, episode_done, metrics)` with step rewards = 0 and the final reward attached to `final_rewards_G`. GRPO centering across the group works out-of-the-box.

**Why this is better than the current `multihop_react_tinker`-style env**: current `browsergym_tinker.py` makes the model emit the full action script in one shot, then replays it. That throws away tree search / re-planning after errors — the reward is single-shot. The `verifiers_rl` pattern lets the model do **stepwise tool-call rollouts** where each `env.step` response is fed back into the context.

## 5. Concrete recommendations for this repo

1. **Keep the single-shot env (`browsergym_tinker.py`) for the smoke path** — cheap baseline, ~550 lines, already works. Rename to `browsergym_oneshot_tinker.py` for clarity.
2. **Add `browsergym_react_tinker.py`** modelled on `verifiers_rl` + `multihop_react_tinker`. Each env.step: LLM reads `axtree_txt + last_error`, emits one action, server runs it, next observation feeds back. Terminal reward from `_task_validate`.
3. **Parallelise rollouts**: replace the sequential `asyncio.to_thread(_build_initial_item, ...)` in `setup()` and `evaluate()` with `asyncio.gather(...)` bounded by a semaphore (cap at 4 Chromiums — RAM-heavy).
4. **Observation preset**: `axtree_txt` only, 24k-char cap, include `obs["last_action_error"]` every turn (drops hallucinated selectors fast).
5. **Reward shaping**: stick to terminal reward; add a small negative shaping term (−0.01 per invalid action exec) only if you see action-spam.
6. **Group size**: MiniWoB tasks reach ceiling fast at G=8; WorkArena needs G=16+ for non-degenerate GRPO signal.
7. **Eval harness**: `rollout_and_score_eval` + `supports_parallel_seeds=True` lets MiniWoB and WorkArena-L1 batch cleanly. WebArena needs one-at-a-time (shared Docker state).
8. **Gotcha**: WebArena / VisualWebArena require `full_reset()` on the instance at the start of each epoch — not each episode. Call once in `setup()`.

## 6. Configs to add next

- `atropos/configs/browsergym_workarena_l1_qwen_8b_smoke.yaml` — 15-step horizon, G=16, SNow dev instance env vars.
- `atropos/configs/browsergym_miniwob_react_qwen_8b_smoke.yaml` — targets the new stepwise env.
- Drop the `*_wandb_smoke.yaml` duplicates; consolidate via `TINKER_USE_WANDB=1` at runtime.

## 7. Open questions / risks

- **Playwright in a cluster**: each rollout worker needs its own Chromium. Docker ulimits + `/dev/shm` size kill it silently. Mount `/dev/shm: 2GB`.
- **WebArena reward reliability**: ~10% of WebArena-raw tasks have buggy validators. Prefer **WebArena-verified**.
- **AssistantBench** is flaky — real-web drift makes trained policies fragile over months. Good for generalization probe, bad for a reproducibility anchor.
- **MiniWoB commit drift**: pin `7fd85d71a4b60325c6585396ec4f48377d049838` hard; later commits silently change 3 tasks' HTML.

## 8. Next steps

- [ ] Write `browsergym_react_tinker.py` using the `verifiers_rl` override pattern.
- [ ] Add a `_probe_common.py` shared between the new env and `multihop_react_tinker` (config_init, chat-template, wandb_log, rollout_and_score_eval).
- [ ] Add WorkArena-L1 config + Docker-Compose file for local SNow dev instance.
- [ ] Run MiniWoB smoke with the stepwise env; verify terminal reward aligns with `gym.make(...).step(...)` return.
