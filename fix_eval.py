import re
import os

def fix_xlam(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Data loading patch
    content = content.replace(
        "examples = examples[:2000]  # cap at 2000 for speed\nprint(f\"[{EXP_NAME}] Parsed {len(examples)} usable examples\")",
        "train_examples = examples[:2000]  # cap at 2000 for speed\ntest_examples = examples[2000:2500]\nprint(f\"[{EXP_NAME}] Parsed {len(train_examples)} train examples, {len(test_examples)} test examples\")"
    )

    # Batch sampling patch
    content = content.replace(
        "batch = random.sample(examples, 4)",
        "batch = random.sample(train_examples, 4)"
    )

    # Eval block
    eval_block = """
    # ── Held-out Evaluation ──────────────────────────────────────────────────
    print(f"\\n[{EXP_NAME}] Evaluating on {len(test_examples)} held-out test examples...")
    test_rewards = []
    for i in range(0, len(test_examples), 4):
        batch = test_examples[i:i+4]
        for prompt_text, tn, args in batch:
            pid = tok.encode(prompt_text, add_special_tokens=False)
            if len(pid) > 2048:
                pid = pid[:2048]
            sp = T.SamplingParams(max_tokens=256, temperature=0.1, top_p=0.95)
            try:
                resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp).result()
                text = tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
                test_rewards.append(reward(text, tn, args))
            except Exception:
                continue

    avg_test = sum(test_rewards) / len(test_rewards) if test_rewards else 0.0
    print(f"[{EXP_NAME}] Held-out Test Reward: {avg_test:.3f}")
"""

    content = content.replace("    print(f\"[{EXP_NAME}] Sampler: {final.path}\")", "    print(f\"[{EXP_NAME}] Sampler: {final.path}\")\n" + eval_block)

    with open(filename, 'w') as f:
        f.write(content)


fix_xlam('grpo_exp_d_xlam.py')
if os.path.exists('experiments/tinker-runs/scripts/grpo_exp_d_xlam.py'):
    fix_xlam('experiments/tinker-runs/scripts/grpo_exp_d_xlam.py')


def fix_100_xlam(filename):
    if not os.path.exists(filename): return
    with open(filename, 'r') as f:
        content = f.read()

    # Data loading
    content = content.replace(
        "examples = examples[:3000]\nprint(f\"[{EXP}] {len(examples)} examples\")",
        "train_examples = examples[:3000]\ntest_examples = examples[3000:3500]\nprint(f\"[{EXP}] {len(train_examples)} train examples, {len(test_examples)} test examples\")"
    )

    # Sampling
    content = content.replace(
        "batch = random.sample(examples, 2)",
        "batch = random.sample(train_examples, 2)"
    )

    eval_block = """
    # ── Held-out Evaluation ──────────────────────────────────────────────────
    print(f"\\n[{EXP}] Evaluating on {len(test_examples)} held-out test examples...")
    test_rewards = []
    for i in range(0, len(test_examples), 4):
        batch = test_examples[i:i+4]
        for prompt_text, tn, args in batch:
            pid = tok.encode(prompt_text, add_special_tokens=False)
            if len(pid) > 1536:
                pid = pid[:1536]
            sp = T.SamplingParams(max_tokens=128, temperature=0.1, top_p=0.95)
            try:
                resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp).result()
                text = tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
                test_rewards.append(reward(text, tn, args))
            except Exception:
                continue

    avg_test = sum(test_rewards) / len(test_rewards) if test_rewards else 0.0
    print(f"[{EXP}] Held-out Test Reward: {avg_test:.3f}")
"""
    content = content.replace("    print(f\"\\n[{EXP}] DONE | last10={avg10:.3f} | run={tc.model_id} | path={f.path}\")", "    print(f\"\\n[{EXP}] DONE | last10={avg10:.3f} | run={tc.model_id} | path={f.path}\")\n" + eval_block)
    with open(filename, 'w') as f:
        f.write(content)

fix_100_xlam('grpo_100_xlam.py')
fix_100_xlam('experiments/tinker-runs/scripts/grpo_100_xlam.py')


def fix_tooluse(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Data loading
    content = content.replace(
        "examples = [(make_prompt(q), t, a) for q, t, a in RAW] * 28\nrandom.shuffle(examples)\nprint(f\"Dataset: {len(examples)} examples, {len(set(t for _, t, _ in RAW))} tools\")",
        """RAW_TEST = [
    ("3 to the power of 4", "calculator", {"expression": "3 ** 4"}),
    ("What is the weather in Paris?", "get_weather", {"city": "Paris", "units": "metric"}),
    ("Search for Python 3.12 release notes", "web_search", {"query": "Python 3.12 release notes"}),
    ("Current time in Tokyo", "get_time", {"timezone": "Asia/Tokyo"}),
    ("Remind me to buy groceries tomorrow", "set_reminder", {"task": "buy groceries", "time": "tomorrow"}),
]

examples = [(make_prompt(q), t, a) for q, t, a in RAW] * 28
random.shuffle(examples)
test_examples = [(make_prompt(q), t, a) for q, t, a in RAW_TEST]
print(f"Dataset: {len(examples)} train examples, {len(test_examples)} test examples")"""
    )

    eval_block = """
    # ── Held-out Evaluation ──────────────────────────────────────────────────
    print(f"\\nEvaluating on {len(test_examples)} held-out test examples...")
    test_rewards = []
    for prompt_text, tn, args in test_examples:
        pid = tok.encode(prompt_text, add_special_tokens=False)
        sp = T.SamplingParams(max_tokens=192, temperature=0.1, top_p=0.95)
        try:
            resp = sc.sample(T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp).result()
            text = tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
            test_rewards.append(reward(text, tn, args))
        except Exception:
            continue

    avg_test = sum(test_rewards) / len(test_rewards) if test_rewards else 0.0
    print(f"Held-out Test Reward: {avg_test:.3f}")
"""

    content = content.replace("    print(\"Done.\")", "    print(\"Done.\")\n" + eval_block)

    with open(filename, 'w') as f:
        f.write(content)

fix_tooluse('grpo_tooluse_tinker.py')

