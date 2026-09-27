"""E6 thin runner: native WebArena (web-arena-x/webarena @ dce04686) test loop, trained actor as the agent LLM.

Mirrors run.py::test() (same env, CoT prompt agent, early-stop rules, auto-login, evaluator_router), with:
  * agent LLM calls -> trained actor (OpenAI-compatible vLLM on Modal), non-thinking, temperature 0;
    OpenAI few-shot convention (role=system,name=example_user/assistant) mapped to user/assistant turns
    because the Qwen chat template rejects non-leading system messages;
  * native LLM judge (llm_fuzzy_match / llm_ua_match) -> openai/gpt-4.1 via OpenRouter
    (gpt-4-1106-preview is retired: 404 on OpenAI and OpenRouter, 2026-09-27);
  * tiktoken cl100k_base for observation truncation (encoding_for_model() rejects the actor id);
  * per-task wall-clock timeout; errors/timeouts recorded as score 0 (failure in denominator);
  * one JSON line per task in <result_dir>/results.jsonl; resumable.
Run from the webarena repo root with the venv active.
"""
import argparse, json, os, signal, subprocess, sys, tempfile, time, traceback
from pathlib import Path
sys.path.insert(0, os.getcwd())  # webarena repo root

import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")
tiktoken.encoding_for_model = lambda _m: _enc  # noqa: E731

import requests

ACTOR_URL = os.environ["TRAINED_ACTOR_BASE_URL"].rstrip("/") + "/v1/chat/completions"
ACTOR_KEY = os.environ["TRAINED_ACTOR_API_KEY"]
ACTOR_MODEL = "pavlov-public-portfolio-bf16"
JUDGE_MODEL = "openai/gpt-4.1"
OR_KEY = os.environ["OPENROUTER_API_KEY"]
class JudgeUnavailable(RuntimeError):
    """Judge provider has no credit (HTTP 402 / insufficient_quota). Task is recorded judge_pending, not scored."""


USAGE = {"actor_prompt": 0, "actor_completion": 0, "actor_calls": 0, "judge_calls": 0, "judge_cost": 0.0}


def _post(url, key, body, timeout):
    last = None
    for attempt in range(8):
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code == 402 or "insufficient_quota" in r.text or "credit" in r.text.lower():
                raise JudgeUnavailable(last)
            if r.status_code in (400, 401, 403, 404):
                break
        except JudgeUnavailable:
            raise
        except Exception as e:  # cold start / network
            last = repr(e)
        time.sleep(min(60, 5 * 2 ** attempt))
    raise RuntimeError(f"LLM call failed: {last}")


def _map_messages(messages):
    out = []
    for m in messages:
        name = m.get("name")
        if m["role"] == "system" and name == "example_user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "system" and name == "example_assistant":
            out.append({"role": "assistant", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def actor_chat(messages, model, temperature, max_tokens, top_p, context_length=0, stop_token=None):
    body = {"model": ACTOR_MODEL, "messages": _map_messages(messages), "temperature": 0.0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    d = _post(ACTOR_URL, ACTOR_KEY, body, timeout=900)
    u = d.get("usage") or {}
    USAGE["actor_prompt"] += u.get("prompt_tokens", 0)
    USAGE["actor_completion"] += u.get("completion_tokens", 0)
    USAGE["actor_calls"] += 1
    return d["choices"][0]["message"]["content"] or ""


def judge_chat(messages, model, temperature, max_tokens, top_p, context_length=0, stop_token=None):
    body = {"model": JUDGE_MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "top_p": top_p}
    d = _post("https://openrouter.ai/api/v1/chat/completions", OR_KEY, body, timeout=120)
    USAGE["judge_calls"] += 1
    USAGE["judge_cost"] += float((d.get("usage") or {}).get("cost") or 0)
    return d["choices"][0]["message"]["content"]


import llms, llms.utils  # noqa: E402
llms.utils.generate_from_openai_chat_completion = actor_chat
import evaluation_harness.helper_functions as hf  # noqa: E402
hf.generate_from_openai_chat_completion = judge_chat

from agent import PromptAgent, construct_agent  # noqa: E402
from browser_env import ActionTypes, ScriptBrowserEnv, create_stop_action  # noqa: E402
from browser_env.auto_login import get_site_comb_from_filepath  # noqa: E402
from browser_env.helper_functions import RenderHelper, get_action_description  # noqa: E402
from evaluation_harness import evaluator_router  # noqa: E402
import run as native_run  # noqa: E402  (early_stop is reused verbatim)


class TaskTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise TaskTimeout("task wall-clock timeout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task_ids", required=True, help="comma list or @file")
    p.add_argument("--result_dir", required=True)
    p.add_argument("--task_timeout", type=int, default=1500)
    a = p.parse_args()
    ids = open(a.task_ids[1:]).read().split() if a.task_ids.startswith("@") else a.task_ids.split(",")
    ids = [int(x) for x in ids if x.strip()]

    sys.argv = ["run.py", "--instruction_path", "agent/prompts/jsons/p_cot_id_actree_2s.json",
                "--provider", "openai", "--model", ACTOR_MODEL, "--mode", "chat", "--temperature", "0",
                "--max_tokens", "384", "--max_obs_length", "1920", "--max_steps", "30",
                "--result_dir", a.result_dir, "--test_start_idx", "0", "--test_end_idx", "812"]
    args = native_run.config()
    args.render_screenshot = False  # keep HTML trajectories text-only (disk); agent never sees screenshots
    native_run.prepare(args)
    rd = Path(a.result_dir)
    res_path = rd / "results.jsonl"
    done = set()
    if res_path.exists():
        done = {json.loads(l)["task_id"] for l in res_path.read_text().splitlines() if l.strip()}
    agent = construct_agent(args)
    thresholds = {"parsing_failure": args.parsing_failure_th, "repeating_action": args.repeating_action_failure_th}
    signal.signal(signal.SIGALRM, _alarm)

    for tid in ids:
        if tid in done:
            continue
        config_file = f"config_files/{tid}.json"
        rec = {"task_id": tid, "score": 0.0, "n_actions": 0, "stop": None, "error": None}
        t0 = time.time()
        u0 = dict(USAGE)
        env = None
        render_helper = None
        signal.alarm(a.task_timeout)
        try:
            env = ScriptBrowserEnv(headless=True, slow_mo=0, observation_type=args.observation_type,
                                   current_viewport_only=args.current_viewport_only,
                                   viewport_size={"width": args.viewport_width, "height": args.viewport_height},
                                   save_trace_enabled=False, sleep_after_execution=args.sleep_after_execution)
            render_helper = RenderHelper(config_file, args.result_dir, args.action_set_tag)
            with open(config_file) as f:
                _c = json.load(f)
            intent = _c["intent"]
            rec["sites"] = _c["sites"]
            rec["eval_types"] = _c["eval"]["eval_types"]
            if _c["storage_state"]:
                cookie_file_name = os.path.basename(_c["storage_state"])
                comb = get_site_comb_from_filepath(cookie_file_name)
                temp_dir = tempfile.mkdtemp()
                subprocess.run([sys.executable, "browser_env/auto_login.py", "--auth_folder", temp_dir,
                                "--site_list", *comb], check=False, timeout=300)
                _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                assert os.path.exists(_c["storage_state"]), "auto_login produced no storage_state"
                config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                with open(config_file, "w") as f:
                    json.dump(_c, f)
            agent.reset(config_file)
            trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info = {"observation": obs, "info": info}
            trajectory.append(state_info)
            meta_data = {"action_history": ["None"]}
            while True:
                flag, stop_info = native_run.early_stop(trajectory, args.max_steps, thresholds)
                if flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(trajectory, intent, meta_data=meta_data)
                    except ValueError as e:
                        action = create_stop_action(f"ERROR: {str(e)}")
                trajectory.append(action)
                action_str = get_action_description(action, state_info["info"]["observation_metadata"],
                                                    action_set_tag=args.action_set_tag,
                                                    prompt_constructor=agent.prompt_constructor
                                                    if isinstance(agent, PromptAgent) else None)
                render_helper.render(action, state_info, meta_data, args.render_screenshot)
                meta_data["action_history"].append(action_str)
                if action["action_type"] == ActionTypes.STOP:
                    rec["stop"] = action.get("answer", "")
                    break
                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                if terminated:
                    trajectory.append(create_stop_action(""))
                    break
            rec["n_actions"] = len(trajectory[1::2])
            rec["stop_reason"] = next((x for x in meta_data["action_history"][-1:]), None)
            rec["final_url"] = env.page.url
            evaluator = evaluator_router(config_file)
            try:
                rec["score"] = float(evaluator(trajectory=trajectory, config_file=config_file, page=env.page,
                                               client=env.get_page_client(env.page)))
            except JudgeUnavailable as e:  # graded later by offline_regrade.py (string/url evals only)
                rec["score"] = None
                rec["judge_pending"] = True
                rec["judge_error"] = str(e)[:200]
        except BaseException as e:  # noqa: BLE001  errors/timeouts are failures
            if isinstance(e, KeyboardInterrupt):
                raise
            rec["error"] = repr(e)[:500]
            rec["score"] = 0.0
            with open(rd / "error.txt", "a") as f:
                f.write(f"[task {tid}] {repr(e)}\n{traceback.format_exc()}\n")
        finally:
            signal.alarm(0)
            for closer in (lambda: render_helper and render_helper.close(), lambda: env and env.close()):
                try:
                    closer()
                except Exception:
                    pass
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["usage"] = {k: (USAGE[k] - u0[k]) for k in USAGE}
        rec["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(res_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
