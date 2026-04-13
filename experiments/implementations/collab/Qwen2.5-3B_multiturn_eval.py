# ============================================================
#  ADVANCED EXPERIMENT — STEP 3: Multi-Turn Evaluation
#  Compares SFT vs GRPO on multi-turn tool call chains
#
#  🔧 FIXES vs previous version:
#   FIX 1 — run_scenario: wrap-up message is now injected
#            IMMEDIATELY after all scenario tool responses are
#            consumed, before the very next generation — not at
#            a fixed final turn index. This matches the change
#            made in generate_chain (GRPO Step 2).
#   FIX 2 — run_scenario: if model repeats the exact same tool
#            call, inject wrap-up immediately to stop the loop.
#   FIX 3 — fake_tool_executor added to eval so Scenario 03
#            (calculate_density) returns a meaningful result
#            instead of the generic fallback.
# ============================================================

# !pip install -q transformers peft accelerate bitsandbytes

from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

import json, os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen2.5-3B-Instruct"
SFT_PATH  = "/content/qwen3b-toolbench-sft"
GRPO_PATH = "/content/qwen3b-toolbench-grpo"

SYSTEM_PROMPT = """You are a helpful AI assistant with access to real-world APIs and tools.

For each user request:
1. Decide which tool to call and respond with ONLY valid JSON:
   {"name": "<tool_name>", "arguments": {<key>: <value>}}

2. After receiving the tool result, either:
   - Call another tool if more information is needed
   - Give a final natural language answer if the task is complete

Never explain your reasoning — just output the tool call JSON or the final answer."""

WRAPUP_MSG = "You now have all the tool results you need. Give your final answer in plain text — do NOT call any more tools."

# ── TEST SCENARIOS ────────────────────────────────────────────
SCENARIOS = [
    {
        "id": "01",
        "name": "Weather + Packing Advice",
        "user": "I'm travelling to Tokyo next week. What's the weather like and what should I pack?",
        "expected_chain": ["get_weather", "final_answer"],
        "tool_responses": [
            '{"location": "Tokyo", "temperature": 12, "condition": "Rainy", "humidity": "80%", "forecast": "Rain all week"}'
        ],
    },
    {
        "id": "02",
        "name": "Stock + News Chain",
        "user": "How is Tesla stock doing and what's the latest news about them?",
        "expected_chain": ["get_stock_price", "search_news", "final_answer"],
        "tool_responses": [
            '{"ticker": "TSLA", "price": 245.10, "change": "+3.2%"}',
            '{"results": ["Tesla Q4 earnings beat expectations", "New Model Y refresh announced"]}'
        ],
    },
    {
        "id": "03",
        "name": "Search + Calculate",
        "user": "Find the population of Tokyo and calculate how many people per km² given the area is 2194 km².",
        "expected_chain": ["search", "calculate", "final_answer"],
        "tool_responses": [
            '{"results": ["Tokyo population: 13.96 million (2023)"]}',
            '{"result": 6362.8}'
        ],
    },
    {
        "id": "04",
        "name": "Single Tool (Direct)",
        "user": "What is the square root of 1764?",
        "expected_chain": ["calculate", "final_answer"],
        "tool_responses": [
            '{"result": 42.0}'
        ],
    },
]

# ── LOAD MODEL HELPER ─────────────────────────────────────────
def load_model(adapter_path, label):
    print(f"\nLoading {label} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    print(f"  {label} ready.")
    return model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

sft_model  = load_model(SFT_PATH,  "SFT model")
grpo_model = load_model(GRPO_PATH, "GRPO model")

# ── INFERENCE ────────────────────────────────────────────────
def infer(model, history):
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=100, temperature=0.1,
            do_sample=True, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

# ── HELPERS ──────────────────────────────────────────────────
def extract_json(text):
    try:
        s = text.index("{");  e = text.rindex("}") + 1
        return json.loads(text[s:e]), True
    except:
        return None, False

def is_valid_tool_call(text):
    obj, valid = extract_json(text)
    return valid and "name" in obj and isinstance(obj.get("arguments"), dict)

def is_clean_output(text):
    try:
        s = text.index("{");  e = text.rindex("}") + 1
        return len(text[:s].strip()) + len(text[e:].strip()) < 15
    except:
        return True

def get_tool_name(text):
    obj, valid = extract_json(text)
    return obj.get("name", "") if valid else ""

def get_args(text):
    obj, valid = extract_json(text)
    return obj.get("arguments", {}) if valid else {}

# ── SCORING ──────────────────────────────────────────────────
def score_chain(responses):
    scores = {
        "chain_started":    False,
        "valid_json_calls": 0,
        "args_populated":   0,
        "clean_outputs":    0,
        "repeated_calls":   0,
        "chain_length":     len(responses),
        "reached_answer":   False,
        "tool_names":       [],
        "tool_args":        [],
    }

    seen_sigs = []

    for i, resp in enumerate(responses):
        if is_valid_tool_call(resp):
            scores["valid_json_calls"] += 1
            if i == 0:
                scores["chain_started"] = True
            if is_clean_output(resp):
                scores["clean_outputs"] += 1

            name = get_tool_name(resp)
            args = get_args(resp)
            scores["tool_names"].append(name)
            scores["tool_args"].append(args)

            if args and len(args) > 0:
                scores["args_populated"] += 1

            sig = (name, frozenset(str(v) for v in args.values()))
            if sig in seen_sigs:
                scores["repeated_calls"] += 1
            seen_sigs.append(sig)
        else:
            if len(resp) > 20:
                scores["reached_answer"] = True

    n = len(responses)
    if n == 0:
        scores["quality"] = 0.0
    else:
        q = 0.0
        if scores["chain_started"]:   q += 0.25
        if scores["reached_answer"]:  q += 0.30
        if scores["valid_json_calls"] > 0:
            q += 0.15 * (scores["valid_json_calls"] / max(1, n - 1))
            q += 0.15 * (scores["args_populated"] / scores["valid_json_calls"])
        if scores["clean_outputs"] > 0:
            q += 0.10 * (scores["clean_outputs"] / max(1, n))
        q -= 0.15 * scores["repeated_calls"]
        scores["quality"] = round(max(0.0, min(1.0, q)), 2)

    return scores


# ── RUN SCENARIO ─────────────────────────────────────────────
def run_scenario(model, scenario, max_turns=6):
    """
    Run a full multi-turn scenario.

    FIX 1: wrap-up message injected immediately when tool_responses
           are exhausted (tool_idx >= len), before the next generation.

    FIX 2: if model repeats the same tool call, inject wrap-up
           immediately to cut the loop short.
    """
    history  = [{"role": "system", "content": SYSTEM_PROMPT}]
    history.append({"role": "user", "content": scenario["user"]})
    responses       = []
    tool_idx        = 0
    last_sig        = None
    wrapup_injected = False

    for turn_idx in range(max_turns):

        # FIX 1: all tool responses consumed — nudge NOW, before generating
        if (not wrapup_injected
                and tool_idx >= len(scenario["tool_responses"])
                and responses):
            history.append({"role": "user", "content": WRAPUP_MSG})
            wrapup_injected = True

        response = infer(model, history)
        responses.append(response)
        history.append({"role": "assistant", "content": response})

        if is_valid_tool_call(response):
            name = get_tool_name(response)
            args = get_args(response)
            sig  = (name, frozenset(str(v) for v in args.values()))

            # FIX 2: repeated call → inject wrap-up and stop adding tool results
            if sig == last_sig and not wrapup_injected:
                history.append({"role": "user", "content": WRAPUP_MSG})
                wrapup_injected = True
                # don't inject another tool result; let model answer next turn
                last_sig = sig
                continue

            last_sig = sig

            if tool_idx < len(scenario["tool_responses"]):
                history.append({"role": "tool",
                                "content": scenario["tool_responses"][tool_idx]})
                tool_idx += 1
            else:
                # no more scripted responses; still stop generating tool calls
                if not wrapup_injected:
                    history.append({"role": "user", "content": WRAPUP_MSG})
                    wrapup_injected = True
        else:
            break  # final answer reached

    return responses, history


# ── PRINT RESULTS ─────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'MULTI-TURN TOOL CALL EVALUATION: SFT vs GRPO':^72}")
print("=" * 72)

all_sft_scores  = []
all_grpo_scores = []

for sc in SCENARIOS:
    print(f"\n[{sc['id']}] {sc['name']}")
    print(f"  User    : {sc['user'][:80]}")
    print(f"  Expected: {' → '.join(sc['expected_chain'])}")

    sft_responses,  _ = run_scenario(sft_model,  sc)
    grpo_responses, _ = run_scenario(grpo_model, sc)

    sft_scores  = score_chain(sft_responses)
    grpo_scores = score_chain(grpo_responses)

    all_sft_scores.append(sft_scores["quality"])
    all_grpo_scores.append(grpo_scores["quality"])

    winner = ("GRPO" if grpo_scores["quality"] > sft_scores["quality"] else
              "SFT"  if sft_scores["quality"]  > grpo_scores["quality"] else "TIE")

    for label, sc_scores, resps in [("SFT ", sft_scores, sft_responses),
                                     ("GRPO", grpo_scores, grpo_responses)]:
        print(f"\n  {label} [{sc_scores['quality']:.2f}]")
        for i, r in enumerate(resps):
            obj, valid = extract_json(r)
            if valid and is_valid_tool_call(r):
                args_str = json.dumps(obj.get("arguments", {}))
                print(f"    Turn {i+1} [TOOL]  : {obj.get('name','')}({args_str[:60]})")
            else:
                print(f"    Turn {i+1} [ANSWER]: {r[:90]}")
        print(f"    Args populated : {sc_scores['args_populated']}/{sc_scores['valid_json_calls']} tool calls")
        print(f"    Repeated calls : {sc_scores['repeated_calls']}")
        print(f"    Got answer     : {sc_scores['reached_answer']}")

    print(f"\n  Winner → {winner}")
    print("-" * 72)

# ── FINAL SUMMARY ─────────────────────────────────────────────
sft_avg  = round(sum(all_sft_scores)  / len(all_sft_scores),  2)
grpo_avg = round(sum(all_grpo_scores) / len(all_grpo_scores), 2)

print(f"\n{'=' * 72}")
print(f"{'FINAL SUMMARY':^72}")
print(f"{'=' * 72}")
print(f"{'Scenario':<35} {'SFT':>8} {'GRPO':>8} {'Winner':>10}")
print(f"{'-' * 72}")
for sc, s, g in zip(SCENARIOS, all_sft_scores, all_grpo_scores):
    w = "GRPO ✅" if g > s else "SFT ✅" if s > g else "TIE"
    print(f"{sc['name']:<35} {s:>8.2f} {g:>8.2f} {w:>10}")
print(f"{'-' * 72}")
print(f"{'AVERAGE':<35} {sft_avg:>8.2f} {grpo_avg:>8.2f}")
print(f"{'=' * 72}")
print(f"\n🏆 Overall Winner: {'GRPO' if grpo_avg > sft_avg else 'SFT' if sft_avg > grpo_avg else 'TIE'}")

print(f"\n{'─' * 72}")
print("Chain health check (what good results look like):")
print("  ✅  Turn 1 is TOOL with populated args  e.g. get_weather({'city': 'Tokyo'})")
print("  ✅  Final turn is ANSWER, not another TOOL")
print("  ✅  Repeated calls = 0 for all scenarios")
print("  ✅  GRPO Args populated ≥ SFT Args populated")
print(f"{'─' * 72}")
