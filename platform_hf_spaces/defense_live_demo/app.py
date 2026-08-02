from __future__ import annotations

import ast
import json
import operator
import os
from pathlib import Path
from typing import Any

import gradio as gr
from huggingface_hub import InferenceClient

from offline_verify import verify


ROOT = Path(__file__).resolve().parent
SNAPSHOT = json.loads((ROOT / "evidence_snapshot.json").read_text(encoding="utf-8"))
ROUTER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")


def evidence_for_groups(group_sizes: list[int], seed: int = 123) -> dict[str, Any]:
    rows = [
        row
        for row in SNAPSHOT["claim_2_matched_budget"]
        if row["group_size"] in group_sizes and seed == 123
    ]
    return {"seed": seed, "matched_budget": True, "runs": rows}


# Deliberately small, safe arithmetic evaluator for tool execution.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float:
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("Only numeric arithmetic is allowed")

    return walk(ast.parse(expression, mode="eval"))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_run_evidence",
            "description": "Look up the bundled matched-budget W&B evidence for one or more rollout group sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_sizes": {
                        "type": "array",
                        "items": {"type": "integer", "enum": [2, 16]},
                    },
                    "seed": {"type": "integer", "enum": [123]},
                },
                "required": ["group_sizes", "seed"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a numeric arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def _message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return message
    return {"content": str(message)}


def _execute_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "lookup_run_evidence":
        return evidence_for_groups(arguments.get("group_sizes", [2, 16]), arguments.get("seed", 123))
    if name == "calculator":
        return {"expression": arguments["expression"], "result": safe_calculate(arguments["expression"])}
    return {"error": f"Unknown tool: {name}"}


def deterministic_tool_fallback(reason: str) -> tuple[str, str, str]:
    call = {"name": "lookup_run_evidence", "arguments": {"group_sizes": [2, 16], "seed": 123}}
    result = _execute_tool(call["name"], call["arguments"])
    payload = {"tool_call": call, "tool_result": result}
    status = f"⚠️ **DETERMINISTIC FALLBACK** — no hosted response was used. Reason: `{reason}`"
    explanation = (
        "The fallback calls the same local evidence function. It returns the two frozen W&B run IDs "
        "for the matched-budget comparison, so the demonstration remains reproducible during a network failure."
    )
    return status, json.dumps(payload, indent=2), explanation


def hosted_tool_call(prompt: str) -> tuple[str, str, str]:
    if not HF_TOKEN:
        return deterministic_tool_fallback("HF_TOKEN is not configured in this Space")
    try:
        client = InferenceClient(provider="auto", api_key=HF_TOKEN)
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence assistant. You must use a supplied tool before answering. "
                        "For comparisons of G=2 and G=16, call lookup_run_evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=300,
            temperature=0.0,
        )
        message = response.choices[0].message
        message_data = _message_dict(message)
        calls = message_data.get("tool_calls") or []
        if not calls:
            raise RuntimeError("Router response contained no tool call")

        executed = []
        for item in calls:
            function = item.get("function", {})
            args = function.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            executed.append(
                {
                    "tool_call": {"name": function.get("name"), "arguments": args},
                    "tool_result": _execute_tool(function.get("name", ""), args),
                }
            )
        status = f"✅ **LIVE HF ROUTER** — `{ROUTER_MODEL}` generated the tool call; this Space executed it locally."
        explanation = (
            "This is a real hosted model decision followed by deterministic local tool execution. "
            "The linked project LoRA is training evidence, while the live serving model is named explicitly above."
        )
        return status, json.dumps(executed, indent=2), explanation
    except Exception as exc:
        return deterministic_tool_fallback(f"hosted call failed: {type(exc).__name__}: {exc}")


MATH_FALLBACK_QUESTION = (
    "A training budget allows 128 sampled completions. Compare group sizes G=2 and G=16: "
    "how many prompts can each setting process, and what is the ratio?"
)
MATH_FALLBACK_ANSWER = (
    "With a fixed budget of 128 completions, prompts = budget / group size.\n\n"
    "- G=2: 128 / 2 = **64 prompts**\n"
    "- G=16: 128 / 16 = **8 prompts**\n"
    "- Ratio: 64 / 8 = **8:1**\n\n"
    "So G=2 covers eight times as many distinct prompts, while G=16 supplies more within-prompt samples."
)


def hosted_math(prompt: str) -> tuple[str, str]:
    if not HF_TOKEN:
        return (
            "⚠️ **DETERMINISTIC FALLBACK** — `HF_TOKEN` is not configured. The fixed defense example is shown below.",
            MATH_FALLBACK_ANSWER,
        )
    try:
        client = InferenceClient(provider="auto", api_key=HF_TOKEN)
        response = client.chat.completions.create(
            model=ROUTER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Solve the math problem carefully. Show compact calculations, then state the final answer.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.0,
        )
        answer = response.choices[0].message.content
        return f"✅ **LIVE HF ROUTER** — response from `{ROUTER_MODEL}`", answer
    except Exception as exc:
        return (
            f"⚠️ **DETERMINISTIC FALLBACK** — hosted call failed: `{type(exc).__name__}: {exc}`",
            MATH_FALLBACK_ANSWER,
        )


def run_offline_verification() -> tuple[str, str]:
    result = verify()
    badge = "✅ **OFFLINE PASS**" if result["status"] == "PASS" else "❌ **OFFLINE FAIL**"
    return badge + " — standard-library checks completed with no network request.", json.dumps(result, indent=2)


def snapshot_rows() -> list[list[Any]]:
    return [
        [row["label"], row["group_size"], row["run_id"], row["interpretation"]]
        for row in SNAPSHOT["claim_2_matched_budget"]
    ]


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
.hero {background: linear-gradient(120deg,#0b1f3a,#173f75); color:white; padding:24px 28px; border-radius:18px;}
.hero h1 {margin:0 0 8px 0; font-size:2rem;}
.mode-note {border-left:4px solid #2563eb; padding-left:14px;}
footer {display:none !important;}
"""


def fetch_live_results(framework: str, task: str):
    """Pull the latest canonical-experiment outputs (HF Hub / W&B / GCS) into the Space.

    HF Spaces hosts no GPU training; this surfaces results produced on the
    GPU-capable backends. Each source degrades gracefully if unreachable.
    """
    import json as _json

    try:
        from fetch_results import fetch_all
    except Exception:  # pragma: no cover - missing module in a stripped build
        return "⚠️ `fetch_results` unavailable in this Space build.", "{}"
    try:
        data = fetch_all(framework or "trl", task or "gsm8k")
        return (
            "✅ Fetched — any source showing an `error` field was unreachable.",
            _json.dumps(data, indent=2, default=str),
        )
    except Exception as e:  # pragma: no cover
        return f"⚠️ fetch failed: {e}", "{}"


with gr.Blocks(title="Tinker RL Defense Demo", css=CSS) as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>Tinker RL · Project Defense Evidence</h1>
          <div>Live hosted inference + frozen W&B evidence + offline verification</div>
        </div>
        """
    )
    gr.Markdown(
        "**Defense safety:** every inference response is labeled either `LIVE HF ROUTER` or "
        "`DETERMINISTIC FALLBACK`. Static evidence remains available even if providers are cold."
    )

    with gr.Tab("1 · Hosted tool call"):
        gr.Markdown(
            "### A hosted model chooses a tool; the Space executes it\n"
            "The Router model is named in the status. The project [tool-call LoRA]"
            "(https://huggingface.co/arvindcr4/tool-call-lora-qwen2.5-7b) is linked separately as an artifact."
        )
        tool_prompt = gr.Textbox(
            value="Use the evidence tool to compare the matched-budget W&B runs for G=2 and G=16 at seed 123.",
            label="Prompt",
            lines=2,
        )
        tool_button = gr.Button("Run hosted tool-call demo", variant="primary")
        tool_status = gr.Markdown()
        tool_json = gr.Code(label="Tool call and executed result", language="json")
        tool_explanation = gr.Markdown()
        tool_button.click(hosted_tool_call, tool_prompt, [tool_status, tool_json, tool_explanation])

    with gr.Tab("2 · Hosted math"):
        gr.Markdown(
            "### Ordinary math reasoning — not a tool-call-only demo\n"
            "This sends the question directly to the HF Router. The fixed fallback is only used when hosted inference is unavailable."
        )
        math_prompt = gr.Textbox(value=MATH_FALLBACK_QUESTION, label="Math problem", lines=3)
        math_button = gr.Button("Run hosted math demo", variant="primary")
        math_status = gr.Markdown()
        math_answer = gr.Markdown()
        math_button.click(hosted_math, math_prompt, [math_status, math_answer])

    with gr.Tab("3 · W&B evidence"):
        gr.Markdown(
            "### Frozen evidence snapshot\n"
            "These figures are bundled in the Space, so they do not depend on W&B loading during the defense. "
            "The run links remain available for drill-down.\n\n"
            "**Audit layers:** **983** account-level objects → **70+** curated runs with usable telemetry → "
            "**19** claim-critical gold rows."
        )
        with gr.Row():
            gr.Image(str(ROOT / "wandb_claim2.png"), label="Claim 2 · matched rollout budget", interactive=False)
            gr.Image(str(ROOT / "wandb_run_hygiene.png"), label="Run hygiene and evidence scale", interactive=False)
        gr.Dataframe(
            value=snapshot_rows(),
            headers=["Evidence row", "G", "W&B run", "Interpretation"],
            datatype=["str", "number", "str", "str"],
            interactive=False,
            label="Gold matched-budget rows",
        )
        gr.Markdown(
            "[Open G=2 run](https://wandb.ai/arvindcr4-pes-university/zvf-training/runs/pob7nd05) · "
            "[Open G=16 run](https://wandb.ai/arvindcr4-pes-university/zvf-training/runs/tiicy3km) · "
            "[Open full W&B project](https://wandb.ai/arvindcr4-pes-university/zvf-training)"
        )

    with gr.Tab("4 · Offline provenance"):
        gr.Markdown(
            "### Verify the fail-safe bundle locally\n"
            "The verifier uses only Python's standard library. It checks the 983-vs-70+ reconciliation, "
            "the 19-row gold subset, "
            "the G=2/G=16 evidence pair, HTTPS run links, unique run IDs, image presence, and SHA-256 digests."
        )
        verify_button = gr.Button("Run offline verification", variant="primary")
        verify_status = gr.Markdown()
        verify_json = gr.Code(language="json", label="Verification report")
        verify_button.click(run_offline_verification, outputs=[verify_status, verify_json])
        gr.File(
            value=[str(ROOT / "offline_verify.py"), str(ROOT / "evidence_snapshot.json")],
            label="Download executable verifier and frozen snapshot",
            interactive=False,
        )
        gr.Markdown(
            "**983 vs 70+:** 983 is the account-level object count; 70+ is the curated set of runs with usable "
            "telemetry. Separately, **19 gold rows** form the claim-critical evidence subset. These are different "
            "audit layers, so the counts are consistent.\n\n"
            "[Modal evidence](https://github.com/arvindcr4/tinker-rl-lab/blob/main/zvf-program/experiments-next/results/passk_modal_qwen3-8b_base_gsm8k_p200_n32_s42.json) · "
            "[Lightning evidence](https://github.com/arvindcr4/tinker-rl-lab/blob/main/zvf-program/experiments-next/results/passk_lightning_qwen3-8b_base_mbpp_p200_n32_s42.json) · "
            "[Colab notebook](https://github.com/arvindcr4/tinker-rl-lab/blob/main/platform_colab/ppo_reinforce_baselines_colab.ipynb)"
        )

    with gr.Tab("5 · Live results"):
        gr.Markdown(
            "### Live results from the canonical experiment\n"
            "Fetches the latest GSM8K GRPO outputs the GPU-capable backends produce — from "
            "HF Hub, W&B, and the GCS receipt bucket. HF Spaces hosts no training; this tab "
            "displays results computed elsewhere (see `fetch_results.py`)."
        )
        with gr.Row():
            lr_framework = gr.Textbox(value="trl", label="framework")
            lr_task = gr.Textbox(value="gsm8k", label="task")
        lr_button = gr.Button("Fetch live results", variant="primary")
        lr_status = gr.Markdown()
        lr_json = gr.Code(language="json", label="Fetched results (HF / W&B / GCS)")
        lr_button.click(fetch_live_results, [lr_framework, lr_task], [lr_status, lr_json])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=7860)
