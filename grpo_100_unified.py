"""100-step GRPO on multiple datasets — optimized for speed"""

import os, json, re, warnings, random, argparse
from datasets import load_dataset

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env"
)
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["math", "synthetic", "xlam"], required=True)
args = parser.parse_args()

MODEL = "Qwen/Qwen3-8B"
SAVE_EVERY = 25

if args.dataset == "math":
    EXP = "math100"
    STEPS, GROUP, LR, TEMP = 100, 4, 5e-6, 0.9
    SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final answer inside \\boxed{}."
    MATH_PROBLEMS = [
        ("What is 17 * 23?", "391"),
        ("What is 256 / 16?", "16"),
        ("What is 2^8?", "256"),
        ("Solve: 3x + 7 = 22", "5"),
        ("What is sqrt(625)?", "25"),
        ("What is 15! / 14!?", "15"),
        ("What is the sum of the first 10 positive integers?", "55"),
        ("What is 7^3?", "343"),
        ("Solve: 2x - 5 = 13", "9"),
        ("What is 144 / 12?", "12"),
        ("What is the GCD of 48 and 36?", "12"),
        ("What is 3^4 + 4^3?", "145"),
    ]
    def make_prompt(q):
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    examples = [(make_prompt(q), a) for q, a in MATH_PROBLEMS] * 20
    random.shuffle(examples)
    def reward(response, answer):
        boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
        for b in boxed:
            if b.strip() == answer:
                return 1.0
        return 0.0

elif args.dataset == "synthetic":
    EXP = "synth100"
    STEPS, GROUP, LR, TEMP = 100, 4, 3e-5, 0.8
    SYSTEM_PROMPT = 'You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n{"tool": "<name>", "arguments": {<key>: <value>}}\nNo prose. Only JSON.'
    TOOLS = [
        {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
        {"name": "get_weather", "description": "Weather", "parameters": {"city": "string", "units": "string"}},
        {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
        {"name": "get_time", "description": "Time", "parameters": {"timezone": "string"}},
        {"name": "set_reminder", "description": "Reminder", "parameters": {"task": "string", "time": "string"}},
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
        ("Set a reminder for team meeting 10am", "set_reminder", {"task": "team meeting", "time": "10am"}),
    ]
    def mkp(q):
        return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nAvailable tools:\n{TOOL_SCHEMA}\n\nUser: {q}<|im_end|>\n<|im_start|>assistant\n"
    examples = [(mkp(q), t, a) for q, t, a in RAW] * 28
    random.shuffle(examples)
    def reward(response, tn, args):
        m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if not m: return 0.0
        try: p = json.loads(m.group())
        except: return 0.1
        s = 0.3
        if p.get("tool") == tn or p.get("name") == tn: s += 0.4
        pa = p.get("arguments", p.get("parameters", {}))
        if isinstance(pa, dict) and args: s += 0.3 * sum(1 for k in args if k in pa) / len(args)
        return min(s, 1.0)

elif args.dataset == "xlam":
    EXP = "xlam100"
    STEPS, GROUP, LR, TEMP = 100, 4, 3e-5, 0.8
    SYSTEM_PROMPT = 'You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n{"tool": "<name>", "arguments": {<key>: <value>}}\nNo prose. Only JSON.'
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    _examples = []
    for row in ds:
        try:
            query = row.get("query", row.get("instruction", ""))
            tools = json.loads(row.get("tools", "[]")) if isinstance(row.get("tools"), str) else row.get("tools", [])
            answers = json.loads(row.get("answers", "[]")) if isinstance(row.get("answers"), str) else row.get("answers", [])
            if not answers or not isinstance(answers, list): continue
            ans = answers[0]
            tn = ans.get("name", ans.get("tool", ""))
            a = ans.get("arguments", ans.get("parameters", {}))
            if isinstance(a, str): a = json.loads(a)
            schema = json.dumps(tools)
            prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nAvailable tools:\n{schema}\n\nUser: {query}<|im_end|>\n<|im_start|>assistant\n"
            _examples.append((prompt, tn, a))
            if len(_examples) > 5000: break
        except: pass
    examples = _examples
    random.shuffle(examples)
    def reward(response, tn, args):
        m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
        if not m: return 0.0
        try: p = json.loads(m.group())
        except: return 0.1
        s = 0.3
        if p.get("tool") == tn or p.get("name") == tn: s += 0.4
        pa = p.get("arguments", p.get("parameters", {}))
        if isinstance(pa, dict) and args: s += 0.3 * sum(1 for k in args if k in pa) / max(len(args), 1)
        return min(s, 1.0)

from utils.tinker_grpo import run_grpo_training

max_tokens = 512 if args.dataset == "math" else 192

run_grpo_training(
    exp_name=EXP,
    model_name=MODEL,
    rank=32,
    steps=STEPS,
    lr=LR,
    group_size=GROUP,
    batch_size=4,
    save_every=SAVE_EVERY,
    examples=examples,
    reward_fn=reward,
    max_tokens=max_tokens,
    temperature=TEMP,
    top_p=0.95
)
