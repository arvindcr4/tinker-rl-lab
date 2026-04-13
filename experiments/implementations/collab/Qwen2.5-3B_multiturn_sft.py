# ============================================================
#  ADVANCED EXPERIMENT — Multi-Turn Tool Calling
#  Model  : Qwen2.5-3B-Instruct + QLoRA
#  Dataset: tuandunghcmut/toolbench-v1 (real APIs, multi-turn)
#  Method : SFT on multi-turn tool call chains
#
#  🔧 FIXES vs previous version:
#   FIX 1 — Parser now REQUIRES at least one tool-result turn
#            so the model always sees [tool_call → result → answer]
#   FIX 2 — Tool calls with empty arguments {} are skipped;
#            the model only trains on calls with real args
#   FIX 3 — Synthetic "bridge" examples are appended to guarantee
#            the model sees complete tool_call→result→answer chains
#            even if ToolBench parsing yields few clean ones
#   FIX 4 — finish_match checked BEFORE action_match so the final
#            answer is never silently dropped
#  ⚡ Runtime-safe: 3B model, 200 samples, 512 seq, 1 epoch
# ============================================================

# !pip install -q transformers datasets peft trl accelerate bitsandbytes

from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

import json, os, re, torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ── CONFIG ───────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR  = "/content/qwen3b-toolbench-sft"
NUM_SAMPLES = 200
MAX_SEQ_LEN = 512
LORA_RANK   = 8
LORA_ALPHA  = 16
EPOCHS      = 1
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LR          = 2e-4

# ── SYSTEM PROMPT ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful AI assistant with access to real-world APIs and tools.

For each user request:
1. Decide which tool to call and respond with ONLY valid JSON:
   {"name": "<tool_name>", "arguments": {<key>: <value>}}

2. After receiving the tool result, either:
   - Call another tool if more information is needed
   - Give a final natural language answer if the task is complete

Never explain your reasoning — just output the tool call JSON or the final answer."""

# ── LOAD TOOLBENCH DATASET ───────────────────────────────────
print("Loading ToolBench dataset ...")
raw = load_dataset("tuandunghcmut/toolbench-v1", split="train")
print(f"Total ToolBench examples: {len(raw)}")

# ── TOKENIZER (needed for apply_chat_template) ────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── PARSE TOOLBENCH FORMAT ───────────────────────────────────
def parse_toolbench_example(row):
    """
    Convert one ToolBench row into a fully-formed multi-turn chat string.

    Requirements for a row to be kept (all three must pass):
      ✅  At least one tool call with non-empty arguments
      ✅  At least one tool-result turn (observation/function/tool)
      ✅  A final natural-language answer (Finish action)

    These three together guarantee the model always trains on the
    complete pattern: tool_call → tool_result → final_answer.
    """
    try:
        convs = row.get("conversations", {})
        if not convs:
            return None

        if isinstance(convs, list):
            roles  = [t.get("from", "")  for t in convs]
            values = [t.get("value", "") for t in convs]
        elif isinstance(convs, dict):
            roles  = convs.get("from",  [])
            values = convs.get("value", [])
        else:
            return None

        if len(roles) < 4 or len(roles) != len(values):   # FIX: need ≥4 turns
            return None

        messages          = [{"role": "system", "content": SYSTEM_PROMPT}]
        has_tool_call     = False   # saw at least one non-empty tool call
        has_tool_result   = False   # saw at least one tool result turn
        has_final_answer  = False   # saw a Finish / final answer turn

        for role, value in zip(roles, values):
            value = (value or "").strip()
            if not value:
                continue

            if role == "system":
                continue

            elif role == "user":
                if "This is not the first time" in value:
                    continue
                messages.append({"role": "user", "content": value[:500]})

            elif role == "assistant":
                # ── FIX 4: check Finish BEFORE generic Action ──
                finish_match = re.search(
                    r'Action:\s*Finish\s*\nAction Input:\s*(\{.*?\})',
                    value, re.DOTALL
                )
                action_match = re.search(
                    r'Action:\s*(\w+)\s*\nAction Input:\s*(\{.*?\})',
                    value, re.DOTALL
                ) if not finish_match else None

                if finish_match:
                    try:
                        obj   = json.loads(finish_match.group(1))
                        final = obj.get("final_answer", "")
                        if final and len(final.strip()) > 10:
                            messages.append({"role": "assistant",
                                             "content": final.strip()})
                            has_final_answer = True
                    except:
                        pass

                elif action_match:
                    try:
                        func_name = action_match.group(1).strip()
                        func_args = json.loads(action_match.group(2))
                        # ── FIX 2: reject empty-argument tool calls ──
                        if not func_args:
                            continue
                        tool_call = json.dumps({
                            "name": func_name,
                            "arguments": func_args
                        })
                        messages.append({"role": "assistant",
                                         "content": tool_call})
                        has_tool_call = True
                    except:
                        pass

            elif role in ("function", "tool", "observation"):
                # ── FIX 1 contribution: record that we saw a result ──
                messages.append({"role": "tool", "content": value[:400]})
                has_tool_result = True

        # ── FIX 1: require the full pattern ──────────────────
        if not (has_tool_call and has_tool_result and has_final_answer):
            return None

        user_turns = sum(1 for m in messages if m["role"] == "user")
        if user_turns == 0:
            return None

        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        return None

# ── PARSE DATASET ────────────────────────────────────────────
print("Parsing ToolBench examples (strict: tool_call+result+answer required) ...")
texts, skipped = [], 0
for row in raw:
    if len(texts) >= NUM_SAMPLES:
        break
    t = parse_toolbench_example(row)
    if t and len(t) > 150:
        texts.append(t)
    else:
        skipped += 1

print(f"Parsed  : {len(texts)} complete chain examples")
print(f"Skipped : {skipped} (missing tool call / result / final answer)")

# ── FIX 3: SYNTHETIC BRIDGE EXAMPLES ─────────────────────────
# If ToolBench parsing is sparse, inject hand-crafted examples that
# explicitly demonstrate tool_call → tool_result → final_answer.
# These are minimal but complete; they teach the "when to stop" pattern.
SYNTHETIC = [
    # ── Single-tool chain ──────────────────────────────────────
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "What is the weather in Paris today?"},
            {"role": "assistant", "content": '{"name": "get_weather", "arguments": {"location": "Paris"}}'},
            {"role": "tool",      "content": '{"location": "Paris", "temperature": 15, "condition": "Cloudy", "humidity": "72%"}'},
            {"role": "assistant", "content": "The weather in Paris today is 15°C and cloudy with 72% humidity. You may want to bring a light jacket."},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "What is the square root of 144?"},
            {"role": "assistant", "content": '{"name": "calculator", "arguments": {"expression": "144**0.5"}}'},
            {"role": "tool",      "content": '{"expression": "144**0.5", "result": 12.0}'},
            {"role": "assistant", "content": "The square root of 144 is 12."},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Search for the latest news about SpaceX."},
            {"role": "assistant", "content": '{"name": "search_news", "arguments": {"query": "SpaceX latest news"}}'},
            {"role": "tool",      "content": '{"results": ["SpaceX Starship completes orbital test", "SpaceX wins NASA contract", "Elon Musk discusses Mars mission timeline"]}'},
            {"role": "assistant", "content": "Here are the latest SpaceX news: Starship completed an orbital test, SpaceX won a new NASA contract, and Elon Musk discussed the Mars mission timeline."},
        ]
    },
    # ── Two-tool chain ─────────────────────────────────────────
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "How is Apple stock doing and what's the latest news about them?"},
            {"role": "assistant", "content": '{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}'},
            {"role": "tool",      "content": '{"ticker": "AAPL", "price": 189.45, "change": "-0.8%", "market": "NASDAQ"}'},
            {"role": "assistant", "content": '{"name": "search_news", "arguments": {"query": "Apple Inc latest news"}}'},
            {"role": "tool",      "content": '{"results": ["Apple Vision Pro sales disappoint analysts", "iPhone 16 pre-orders strong", "Apple Services revenue up 14%"]}'},
            {"role": "assistant", "content": "Apple (AAPL) is currently at $189.45, down 0.8%. Recent news: Vision Pro sales have disappointed analysts, though iPhone 16 pre-orders are strong and Services revenue is up 14%."},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "What's the weather in Tokyo and what are the top tourist spots there?"},
            {"role": "assistant", "content": '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'},
            {"role": "tool",      "content": '{"location": "Tokyo", "temperature": 18, "condition": "Clear", "humidity": "55%"}'},
            {"role": "assistant", "content": '{"name": "search_attractions", "arguments": {"city": "Tokyo"}}'},
            {"role": "tool",      "content": '{"results": ["Senso-ji Temple", "Shibuya Crossing", "Tokyo Skytree", "Tsukiji Market"]}'},
            {"role": "assistant", "content": "Tokyo has clear skies at 18°C — great weather for sightseeing! Top attractions include Senso-ji Temple, Shibuya Crossing, Tokyo Skytree, and Tsukiji Market."},
        ]
    },
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Find the population of New York and calculate the population density given area 783 km²."},
            {"role": "assistant", "content": '{"name": "search", "arguments": {"query": "New York City population 2024"}}'},
            {"role": "tool",      "content": '{"results": ["New York City population: 8.3 million (2024)"]}'},
            {"role": "assistant", "content": '{"name": "calculator", "arguments": {"expression": "8300000 / 783"}}'},
            {"role": "tool",      "content": '{"expression": "8300000 / 783", "result": 10601.5}'},
            {"role": "assistant", "content": "New York City has a population of approximately 8.3 million. With an area of 783 km², the population density is about 10,602 people per km²."},
        ]
    },
    # ── Three-tool chain ───────────────────────────────────────
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Get Tesla stock price, search for their latest news, and tell me the stock-to-news sentiment."},
            {"role": "assistant", "content": '{"name": "get_stock_price", "arguments": {"ticker": "TSLA"}}'},
            {"role": "tool",      "content": '{"ticker": "TSLA", "price": 245.10, "change": "+3.2%"}'},
            {"role": "assistant", "content": '{"name": "search_news", "arguments": {"query": "Tesla news today"}}'},
            {"role": "tool",      "content": '{"results": ["Tesla Q4 earnings beat expectations", "New Model Y refresh announced", "Tesla expands Supercharger network"]}'},
            {"role": "assistant", "content": "Tesla (TSLA) is up 3.2% at $245.10. The news is positive: Q4 earnings beat expectations, a new Model Y refresh was announced, and the Supercharger network is expanding. The sentiment aligns well with the price increase."},
        ]
    },
    # ── Edge cases: model should answer directly without tools ──
    {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": "Translate 'Good morning' to French."},
            {"role": "assistant", "content": '{"name": "translate", "arguments": {"text": "Good morning", "target_language": "French"}}'},
            {"role": "tool",      "content": '{"original": "Good morning", "translated": "Bonjour", "language": "French"}'},
            {"role": "assistant", "content": "'Good morning' in French is 'Bonjour'."},
        ]
    },
]

print(f"\nAdding {len(SYNTHETIC)} synthetic chain examples ...")
for ex in SYNTHETIC:
    t = tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False
    )
    texts.append(t)

print(f"Total training examples: {len(texts)}")
print(f"  From ToolBench : {len(texts) - len(SYNTHETIC)}")
print(f"  Synthetic      : {len(SYNTHETIC)}")

# Show what a complete chain looks like
print(f"\n── Sample complete chain (first 600 chars) ──")
print(texts[0][:600] if texts else "No examples parsed!")
print("...")

multi_turn = sum(1 for t in texts if "<|im_start|>tool" in t)
print(f"\nExamples with tool turns: {multi_turn}/{len(texts)}")

if len(texts) < 10:
    raise RuntimeError(
        "Too few examples parsed from ToolBench. "
        "Check the dataset schema. At minimum the synthetic examples should be present."
    )

dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.1, seed=42)
print(f"\nTrain: {len(dataset['train'])}  |  Eval: {len(dataset['test'])}")

# ── MODEL ────────────────────────────────────────────────────
print("\nLoading Qwen2.5-3B model ...")
has_gpu = torch.cuda.is_available()
print(f"GPU: {torch.cuda.get_device_name(0) if has_gpu else 'CPU only'}")
if not has_gpu:
    raise RuntimeError("No GPU! Runtime → Change runtime type → T4 GPU")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.config.use_cache = False
print("Model loaded.")

# ── LORA ─────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# ── TRAIN ────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
    optim="paged_adamw_8bit",
    seed=42,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    args=sft_config,
)

print("\n🚀 Starting multi-turn SFT training ...")
print(f"   Model  : Qwen2.5-3B")
print(f"   Dataset: {len(dataset['train'])} examples (ToolBench + synthetic)")
print(f"   Each example has: tool_call → tool_result → final_answer\n")
trainer.train()

# ── SAVE ─────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Adapter saved to {OUTPUT_DIR}")
print(f"   Files: {os.listdir(OUTPUT_DIR)}")

# ── MULTI-TURN INFERENCE TEST ─────────────────────────────────
print("\n── Multi-turn inference test ──")

def run_multiturn(conversation_history):
    model.eval()
    prompt = tokenizer.apply_chat_template(
        conversation_history, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

print("\n[Simulating multi-turn tool call chain]")
print("=" * 60)

history = [{"role": "system", "content": SYSTEM_PROMPT}]
user_q  = "What's the weather in Tokyo and what are the top tourist attractions there?"
history.append({"role": "user", "content": user_q})

print(f"User   : {user_q}")
r1 = run_multiturn(history)
print(f"Model  : {r1}")

history.append({"role": "assistant", "content": r1})
history.append({"role": "tool", "content": '{"location": "Tokyo", "temperature": 18, "condition": "Clear", "humidity": "55%"}'})
r2 = run_multiturn(history)
print(f"\nTool   : weather result injected")
print(f"Model  : {r2}")

history.append({"role": "assistant", "content": r2})
history.append({"role": "tool", "content": '{"results": ["Senso-ji Temple", "Shibuya Crossing", "Tokyo Skytree", "Tsukiji Market"]}'})
r3 = run_multiturn(history)
print(f"\nTool   : attractions result injected")
print(f"Model  : {r3}")
print("=" * 60)
print("\n✅ Step 1 complete! Run Step 2 (GRPO) next.")
