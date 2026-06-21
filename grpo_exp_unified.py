import os, json, re, warnings, random, sys, argparse

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env"
)

import torch, tinker, tinker.types as T
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description="Unified GRPO Experiment")
parser.add_argument("--exp_name", type=str, default="A_baseline", help="Experiment name")
parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Base model")
parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
parser.add_argument("--group_size", type=int, default=8, help="Group size for GRPO")
parser.add_argument("--steps", type=int, default=30, help="Training steps")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--temp", type=float, default=0.8, help="Temperature for sampling")
parser.add_argument("--save_every", type=int, default=10, help="Save frequency")
args = parser.parse_args()

EXP_NAME = args.exp_name
MODEL = args.model
LORA_RANK = args.lora_rank
GROUP_SIZE = args.group_size
STEPS = args.steps
LR = args.lr
TEMP = args.temp
SAVE_EVERY = args.save_every

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
    '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
    "No prose. Only JSON."
)

TOOLS = [
    {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
    {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"city": "string", "units": "string"},
    },
    {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
    {"name": "get_time", "description": "Time in timezone", "parameters": {"timezone": "string"}},
    {
        "name": "set_reminder",
        "description": "Set a reminder",
        "parameters": {"task": "string", "time": "string"},
    },
]
TOOL_SCHEMA = json.dumps(TOOLS)

RAW = [
    ("What is 245 * 37?", "calculator", {"expression": "245 * 37"}),
    ("Calculate sqrt(144)", "calculator", {"expression": "sqrt(144)"}),
    ("15% of 980?", "calculator", {"expression": "0.15 * 980"}),
    ("Divide 1024 by 32", "calculator", {"expression": "1024 / 32"}),
    ("2 to the power of 10", "calculator", {"expression": "2 ** 10"}),
    ("Weather in Tokyo?", "get_weather", {"city": "Tokyo", "units": "metric"}),
    ("Is it raining in London?", "get_weather", {"city": "London", "units": "metric"}),
    ("Temperature in New York", "get_weather", {"city": "New York", "units": "imperial"}),
    ("How hot is Dubai right now?", "get_weather", {"city": "Dubai", "units": "metric"}),
    ("Search for GPT-5 news", "web_search", {"query": "GPT-5 news"}),
    ("Capital of Australia?", "web_search", {"query": "capital of Australia"}),
    ("Find Python asyncio tutorial", "web_search", {"query": "Python asyncio tutorial"}),
    ("What time is it in Singapore?", "get_time", {"timezone": "Asia/Singapore"}),
    ("Current time in Los Angeles?", "get_time", {"timezone": "America/Los_Angeles"}),
    ("Time in Berlin?", "get_time", {"timezone": "Europe/Berlin"}),
    ("Remind me to call mom at 6pm", "set_reminder", {"task": "call mom", "time": "6pm"}),
    (
        "Set a reminder for team meeting 10am",
        "set_reminder",
        {"task": "team meeting", "time": "10am"},
    ),
    ("Remind me to take medicine at 8pm", "set_reminder", {"task": "take medicine", "time": "8pm"}),
]

def make_prompt(query):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nAvailable tools:\n{TOOL_SCHEMA}\n\nUser: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

examples = [(make_prompt(q), t, a) for q, t, a in RAW] * 28
random.shuffle(examples)

def reward(response, tool_name, arguments):
    m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
    if not m:
        return 0.0
    try:
        p = json.loads(m.group())
    except json.JSONDecodeError:
        return 0.1
    score = 0.3
    if p.get("tool") == tool_name or p.get("name") == tool_name:
        score += 0.4
    pred_args = p.get("arguments", p.get("parameters", {}))
    if isinstance(pred_args, dict) and arguments:
        score += 0.3 * sum(1 for k in arguments if k in pred_args) / len(arguments)
    return min(score, 1.0)

from utils.tinker_grpo import run_grpo_training

run_grpo_training(
    exp_name=EXP_NAME,
    model_name=MODEL,
    rank=LORA_RANK,
    steps=STEPS,
    lr=LR,
    group_size=GROUP_SIZE,
    batch_size=4,
    save_every=SAVE_EVERY,
    examples=examples,
    reward_fn=reward,
    max_tokens=192,
    temperature=TEMP,
    top_p=0.95
)
