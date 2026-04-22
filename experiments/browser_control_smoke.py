#!/usr/bin/env python3
"""LLM-controlled browser smoke benchmark using agent-browser.

This is intentionally small: it validates that an LLM controller can read live
browser snapshots, choose browser actions, and complete deterministic web tasks
through a real Chromium/CDP control path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from groq import Groq


PREFERRED_MODEL = os.environ.get("GROQ_MODEL", "kimi-k2-0905-preview")
FALLBACK_MODELS = [
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
]
PROJECT = "tinker-agentic-smoke"
GROUP = "agentic-wandb-smoke"


@dataclass
class BrowserTask:
    task_id: str
    path: str
    objective: str
    max_steps: int = 10


TASKS = [
    BrowserTask(
        task_id="checkout",
        path="/checkout.html",
        objective=(
            "Complete the checkout: enter full name Ada Lovelace, email "
            "ada@example.com, choose the Pro plan, set seats to 3, accept "
            "terms, then submit the order."
        ),
        max_steps=9,
    ),
    BrowserTask(
        task_id="inventory_search",
        path="/inventory.html",
        objective=(
            "Find the Qwen3-8B model in the inventory and open/select its "
            "details."
        ),
        max_steps=7,
    ),
    BrowserTask(
        task_id="settings",
        path="/settings.html",
        objective=(
            "Configure the browser agent settings: choose Browser Agent, set "
            "browsing depth to 4, enable browser tools, and save settings."
        ),
        max_steps=8,
    ),
]


CHECKOUT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Checkout Task</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; line-height: 1.45; }
    label { display: block; margin: 14px 0; }
    input, select, button { font: inherit; padding: 8px; margin-top: 4px; }
    button { cursor: pointer; }
    #status { margin-top: 20px; padding: 12px; border: 1px solid #999; }
  </style>
</head>
<body data-success="false">
  <h1>Checkout</h1>
  <form id="checkout-form">
    <label>Full name
      <input id="full-name" aria-label="Full name" autocomplete="off">
    </label>
    <label>Email address
      <input id="email" aria-label="Email address" type="email" autocomplete="off">
    </label>
    <label>Plan
      <select id="plan" aria-label="Plan">
        <option value="">Choose a plan</option>
        <option value="starter">Starter</option>
        <option value="pro">Pro</option>
        <option value="enterprise">Enterprise</option>
      </select>
    </label>
    <label>Seats
      <input id="seats" aria-label="Seats" type="number" min="1" value="1">
    </label>
    <label>
      <input id="terms" aria-label="Accept terms" type="checkbox">
      Accept terms
    </label>
    <button id="submit-order" type="submit">Submit order</button>
  </form>
  <div id="status" role="status">Waiting for order.</div>
  <script>
    document.querySelector("#checkout-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const name = document.querySelector("#full-name").value.trim();
      const email = document.querySelector("#email").value.trim();
      const plan = document.querySelector("#plan").value;
      const seats = document.querySelector("#seats").value.trim();
      const terms = document.querySelector("#terms").checked;
      const ok = name === "Ada Lovelace" && email === "ada@example.com" &&
        plan === "pro" && seats === "3" && terms;
      document.body.dataset.success = String(ok);
      document.querySelector("#status").textContent = ok
        ? "Order confirmed for Ada Lovelace on Pro seats 3."
        : "Order incomplete. Check name, email, plan, seats, and terms.";
    });
  </script>
</body>
</html>
"""


INVENTORY_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Inventory Task</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 820px; margin: 40px auto; line-height: 1.45; }
    input, button { font: inherit; padding: 8px; }
    table { border-collapse: collapse; width: 100%; margin-top: 18px; }
    th, td { border: 1px solid #aaa; padding: 10px; text-align: left; }
    #status { margin-top: 20px; padding: 12px; border: 1px solid #999; }
  </style>
</head>
<body data-success="false">
  <h1>Model Inventory</h1>
  <label>Search inventory
    <input id="search" aria-label="Search inventory" autocomplete="off">
  </label>
  <table aria-label="Inventory results">
    <thead><tr><th>Model</th><th>Capability</th><th>Action</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <div id="details" aria-live="polite"></div>
  <div id="status" role="status">No model selected.</div>
  <script>
    const models = [
      ["Llama-3.1-8B", "math reasoning"],
      ["Qwen3-8B", "browser and tool control"],
      ["Qwen3-4B", "compact reasoning"],
      ["HumanEval Runner", "code execution"]
    ];
    const rows = document.querySelector("#rows");
    function render() {
      const query = document.querySelector("#search").value.toLowerCase();
      rows.innerHTML = "";
      models
        .filter(([name, cap]) => (name + " " + cap).toLowerCase().includes(query))
        .forEach(([name, cap]) => {
          const tr = document.createElement("tr");
          tr.innerHTML = `<td>${name}</td><td>${cap}</td><td><button aria-label="Open ${name} details">Open ${name} details</button></td>`;
          tr.querySelector("button").addEventListener("click", () => {
            document.querySelector("#details").textContent = `${name}: ${cap}`;
            const ok = name === "Qwen3-8B";
            document.body.dataset.success = String(ok);
            document.querySelector("#status").textContent = ok
              ? "Selected Qwen3-8B for browser and tool control."
              : `Selected ${name}, not the target model.`;
          });
          rows.appendChild(tr);
        });
    }
    document.querySelector("#search").addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""


SETTINGS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Settings Task</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; line-height: 1.45; }
    label { display: block; margin: 14px 0; }
    input, select, button { font: inherit; padding: 8px; margin-top: 4px; }
    #status { margin-top: 20px; padding: 12px; border: 1px solid #999; }
  </style>
</head>
<body data-success="false">
  <h1>Agent Settings</h1>
  <form id="settings-form">
    <label>Agent profile
      <select id="profile" aria-label="Agent profile">
        <option value="reasoning">Reasoning Agent</option>
        <option value="browser">Browser Agent</option>
        <option value="coding">Coding Agent</option>
      </select>
    </label>
    <label>Browsing depth
      <input id="depth" aria-label="Browsing depth" type="number" min="1" max="6" value="1">
    </label>
    <label>
      <input id="browser-tools" aria-label="Enable browser tools" type="checkbox">
      Enable browser tools
    </label>
    <button id="save-settings" type="submit">Save settings</button>
  </form>
  <div id="status" role="status">Settings not saved.</div>
  <script>
    document.querySelector("#settings-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const profile = document.querySelector("#profile").value;
      const depth = document.querySelector("#depth").value.trim();
      const tools = document.querySelector("#browser-tools").checked;
      const ok = profile === "browser" && depth === "4" && tools;
      document.body.dataset.success = String(ok);
      document.querySelector("#status").textContent = ok
        ? "Settings saved: Browser Agent depth 4 tools enabled."
        : "Settings incomplete. Check profile, depth, and tools.";
    });
  </script>
</body>
</html>
"""


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_site(site_dir: Path) -> None:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "checkout.html").write_text(CHECKOUT_HTML, encoding="utf-8")
    (site_dir / "inventory.html").write_text(INVENTORY_HTML, encoding="utf-8")
    (site_dir / "settings.html").write_text(SETTINGS_HTML, encoding="utf-8")


def start_server(site_dir: Path, port: int) -> ThreadingHTTPServer:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(site_dir), **kwargs)

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_browser(session: str, args: list[str], timeout: float = 30) -> dict[str, Any]:
    cmd = ["agent-browser", "--session", session, *args]
    start = time.time()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "duration_s": time.time() - start,
    }


def snapshot(session: str) -> str:
    result = run_browser(session, ["snapshot", "-i", "-c"], timeout=30)
    if result["returncode"] != 0:
        return f"SNAPSHOT_ERROR:\n{result['stderr']}\n{result['stdout']}"
    return result["stdout"][-8000:]


def success_state(session: str) -> dict[str, Any]:
    js = (
        "({success: document.body.dataset.success === 'true', "
        "status: document.querySelector('#status')?.innerText || '', "
        "url: location.href, title: document.title})"
    )
    result = run_browser(session, ["eval", js], timeout=15)
    text = f"{result['stdout']}\n{result['stderr']}"
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "success": False,
            "status": "Could not parse success state",
            "raw": text[-1000:],
        }
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"success": False, "status": "Invalid success JSON", "raw": text[-1000:]}


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in model response: {text[:500]}")
    decoder = json.JSONDecoder()
    parsed, _ = decoder.raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def choose_action(
    client: Groq,
    model: str,
    objective: str,
    state: dict[str, Any],
    page_snapshot: str,
    history: list[dict[str, Any]],
    last_error: str | None,
) -> dict[str, Any]:
    history_text = json.dumps(history[-5:], indent=2) if history else "[]"
    error_text = last_error or "none"
    prompt = f"""Objective:
{objective}

Current page state:
{json.dumps(state, indent=2)}

Recent actions:
{history_text}

Last execution error:
{error_text}

Interactive browser snapshot:
{page_snapshot}

Return exactly one JSON object. Allowed actions:
{{"action":"fill","ref":"@e1","text":"value"}}
{{"action":"click","ref":"@e1"}}
{{"action":"check","ref":"@e1"}}
{{"action":"uncheck","ref":"@e1"}}
{{"action":"select","ref":"@e1","value":"visible option text or option value"}}
{{"action":"press","key":"Enter"}}
{{"action":"wait","ms":500}}
{{"action":"finish"}}

Use only refs that appear in the snapshot. If the state already shows success,
return {{"action":"finish"}}."""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise browser-control agent. Choose the next "
                    "single action needed to complete the task. Return only JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0,
    )
    return extract_json(response.choices[0].message.content or "")


def resolve_model(client: Groq) -> tuple[str, str | None]:
    available = {model.id for model in client.models.list().data}
    if PREFERRED_MODEL in available:
        return PREFERRED_MODEL, None
    for fallback in FALLBACK_MODELS:
        if fallback in available:
            return fallback, (
                f"Preferred Groq model {PREFERRED_MODEL!r} is unavailable; "
                f"using {fallback!r}."
            )
    raise RuntimeError(
        "No configured Groq controller model is available. "
        f"Preferred={PREFERRED_MODEL!r}; fallbacks={FALLBACK_MODELS!r}; "
        f"available={sorted(available)!r}"
    )


def execute_action(session: str, action: dict[str, Any]) -> dict[str, Any]:
    kind = str(action.get("action", "")).lower()
    if kind == "finish":
        return {"returncode": 0, "stdout": "finish", "stderr": "", "cmd": ["finish"]}
    if kind == "wait":
        ms = int(action.get("ms", 500))
        return run_browser(session, ["wait", str(ms)], timeout=max(5, ms / 1000 + 5))
    if kind == "press":
        return run_browser(session, ["press", str(action.get("key", "Enter"))])
    ref = str(action.get("ref", "")).strip()
    if not ref:
        return {"returncode": 2, "stdout": "", "stderr": "Missing ref", "cmd": []}
    if kind == "click":
        return run_browser(session, ["click", ref])
    if kind == "check":
        return run_browser(session, ["check", ref])
    if kind == "uncheck":
        return run_browser(session, ["uncheck", ref])
    if kind == "fill":
        return run_browser(session, ["fill", ref, str(action.get("text", ""))])
    if kind == "select":
        return run_browser(session, ["select", ref, str(action.get("value", ""))])
    return {"returncode": 2, "stdout": "", "stderr": f"Unknown action: {kind}", "cmd": []}


def run_task(
    client: Groq,
    model: str,
    session: str,
    base_url: str,
    task: BrowserTask,
    screenshot_dir: Path,
) -> dict[str, Any]:
    task_url = base_url + task.path
    run_browser(session, ["open", task_url], timeout=45)
    run_browser(session, ["wait", "500"], timeout=10)

    history: list[dict[str, Any]] = []
    last_error: str | None = None
    final_state: dict[str, Any] = {}

    for step in range(task.max_steps):
        final_state = success_state(session)
        if final_state.get("success"):
            break

        page_snapshot = snapshot(session)
        try:
            action = choose_action(
                client,
                model,
                task.objective,
                final_state,
                page_snapshot,
                history,
                last_error,
            )
        except Exception as exc:
            history.append({"step": step, "model_error": repr(exc)})
            last_error = repr(exc)
            continue

        execution = execute_action(session, action)
        history.append(
            {
                "step": step,
                "action": action,
                "returncode": execution["returncode"],
                "stdout": execution["stdout"][-1000:],
                "stderr": execution["stderr"][-1000:],
            }
        )
        last_error = None if execution["returncode"] == 0 else execution["stderr"][-1000:]
        if action.get("action") == "finish":
            break
        run_browser(session, ["wait", "250"], timeout=10)

    final_state = success_state(session)
    screenshot_path = screenshot_dir / f"{task.task_id}.png"
    shot = run_browser(session, ["screenshot", str(screenshot_path)], timeout=30)

    return {
        "task_id": task.task_id,
        "objective": task.objective,
        "url": task_url,
        "success": bool(final_state.get("success")),
        "status": final_state.get("status"),
        "steps": len(history),
        "history": history,
        "screenshot": str(screenshot_path),
        "screenshot_returncode": shot["returncode"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/results/browser_control_smoke.json")
    parser.add_argument("--site-dir", default="experiments/results/browser_control_site")
    parser.add_argument("--screenshot-dir", default="experiments/results/browser_control_screenshots")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--session", default=f"browser-control-smoke-{int(time.time())}")
    args = parser.parse_args()

    if "GROQ_API_KEY" not in os.environ:
        raise RuntimeError("GROQ_API_KEY is required for the LLM browser controller")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    site_dir = Path(args.site_dir)
    screenshot_dir = Path(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    write_site(site_dir)

    port = find_free_port()
    server = start_server(site_dir, port)
    base_url = f"http://127.0.0.1:{port}"
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    controller_model, model_warning = resolve_model(client)
    if model_warning:
        print(model_warning)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=PROJECT,
            group=GROUP,
            name=(
                "browser-control-"
                f"{controller_model.replace('/', '-').replace('.', '-')}"
                "-agent-browser-smoke"
            ),
            config={
                "controller_model": controller_model,
                "preferred_model": PREFERRED_MODEL,
                "browser_driver": "agent-browser",
                "task_count": len(TASKS),
                "base_url": base_url,
                "model_warning": model_warning,
            },
        )

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    try:
        for task in TASKS:
            task_result = run_task(
                client,
                controller_model,
                args.session,
                base_url,
                task,
                screenshot_dir,
            )
            results.append(task_result)
            if wandb_run:
                import wandb

                wandb.log(
                    {
                        f"browser_control/{task.task_id}/success": int(task_result["success"]),
                        f"browser_control/{task.task_id}/steps": task_result["steps"],
                    }
                )
                if Path(task_result["screenshot"]).exists():
                    wandb.log(
                        {
                            f"browser_control/{task.task_id}/screenshot": wandb.Image(
                                task_result["screenshot"]
                            )
                        }
                    )
    finally:
        run_browser(args.session, ["close"], timeout=10)
        server.shutdown()

    pass_rate = sum(1 for r in results if r["success"]) / len(results)
    summary = {
        "experiment": "browser_control_smoke",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "preferred_model": PREFERRED_MODEL,
        "controller_model": controller_model,
        "model_warning": model_warning,
        "browser_driver": "agent-browser",
        "task_count": len(results),
        "success_count": sum(1 for r in results if r["success"]),
        "pass_rate": pass_rate,
        "tasks": results,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if wandb_run:
        import wandb

        wandb.log(
            {
                "browser_control/pass_rate": pass_rate,
                "browser_control/success_count": summary["success_count"],
                "browser_control/task_count": summary["task_count"],
            }
        )
        wandb.save(str(output_path))
        wandb.finish(exit_code=0 if pass_rate == 1.0 else 1)

    print(json.dumps(summary, indent=2))
    return 0 if pass_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
