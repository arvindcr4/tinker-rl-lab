# Evaluation Protocol Documentation

This document details the evaluation protocols, dataset splits, reward parsers, and claim status for every task in the TinkerRL Lab study.

## Quick Reference Table

| Task | Dataset/Split | Training Use | Held-Out Eval | N | Reward/Parser | Decoding | Claim Status |
|------|--------------|-------------|---------------|---|---------------|----------|--------------|
| **GSM8K Rollout** | train split / sampled batches | ✅ Yes | ❌ No | varies | boxed/final-answer parser | T=0.8–1.0 | Training dynamics only |
| **GSM8K Held-Out** | test slice (seed 0) | ❌ No | ✅ Yes | 200 | same parser | greedy T=0 | Capability check |
| **MATH-500** | test split as reward env | ✅ Yes (probe) | ❌ No | varies | boxed partial + final | varies | Proxy/probe only |
| **HumanEval-style** | test prompts / restricted sandbox | ✅ Yes (probe) | ❌ No | varies | Python sandbox (builtins removed) | varies | Proxy/probe only |
| **Tool-Use** | custom/schema tasks | ✅ Yes | ❌ No | varies | JSON/schema reward | varies | Schema compliance only |
| **Arithmetic** | generated | ✅ Yes | ❌ No | varies | binary correctness | T=0.8 | Training dynamics |
| **BrowserGym** | MiniWoB | ✅ Yes (smoke) | ❌ No | 1 | browser task success | greedy | Integration smoke only |

---

## Detailed Protocols

### GSM8K

**Dataset:** GSM8K (`train`/`test` split from OpenAI release)

**Training Protocol:**
- Prompts sampled from `train` split
- Reward: Binary correctness via boxed-answer or final numerical answer parser
- Group size: G ∈ {4, 8, 16, 32}
- Decoding: Temperature 0.8–1.0
- Reported metric: `last10_avg` (mean reward over last 10 steps)

**Held-Out Protocol:**
- Evaluation on `test` split, N = 200 examples
- Fixed seed (seed=0) for reproducibility
- Decoding: Greedy (T=0)
- Metric: Accuracy (% correct)
- **NOT** the same as training reward

**Claim Status:**
- ❌ Training reward ≠ held-out accuracy
- ✅ Held-out is the only valid capability claim
- Paper reports: base 82.0% → GRPO 83.3% (p = 0.26, not significant)

---

### MATH-500

**Dataset:** MATH (`test` split used as reward-environment probe)

**Protocol:**
- Prompt pool loaded from `test` split
- Used **inside** training loop as reward signal
- NOT held-out from training
- Reward: Partial credit for `\boxed{}` format + final answer match

**Claim Status:**
- ❌ NOT a held-out generalization test
- ✅ Reward-environment probe for gradient availability
- Paper treats as: "useful evidence about reward dynamics"

---

### HumanEval-style

**Dataset:** HumanEval (`test` prompts)

**Protocol:**
- Prompt pool from HumanEval `test` split
- Execution in restricted Python sandbox
- **Built-ins removed:** `len`, `range`, `sum`, `print`, etc.
- This causes false negatives (valid programs rejected)

**Claim Status:**
- ❌ NOT proof of code generation capability
- ❌ NOT comparable to standard HumanEval benchmarks
- ✅ Evidence about reward-environment behavior
- Paper states: "sandbox removes normal Python built-ins"

---

### Tool-Use

**Datasets:** Custom function-calling tasks, ToolBench, xlam-60k

**Reward Components:**
1. +0.3: Valid JSON output
2. +0.4: Correct tool name
3. +0.3: All argument keys present

**What is NOT scored:**
- ❌ Argument values correctness
- ❌ Tool execution success
- ❌ Environment feedback
- ❌ Task completion

**Claim Status:**
- ❌ NOT proof of tool execution competence
- ✅ Evidence of schema compliance learning
- Paper states: "reward scores JSON well-formedness, tool-name selection, and argument-key presence without executing tools"

---

### Arithmetic RL

**Dataset:** Generated arithmetic problems

**Protocol:**
- Two-digit addition: 199 possible answers
- Random baseline: ~0.5% accuracy
- Oracle: 100% accuracy
- Binary correctness reward

**Claim Status:**
- ✅ Stack validation (pipeline works)
- ❌ Not evidence of LLM reasoning
- ✅ Evidence for GRPO mechanics

---

### BrowserGym MiniWoB

**Protocol:**
- Smoke test only (N=1 run, 1 step)
- Real Chromium/Playwright integration
- Reward: Browser task success

**Result:** 0.0 reward obtained

**Claim Status:**
- ❌ NOT proof of browser competence
- ✅ Evidence of pipeline integration
- Paper states: "only establishes pipeline routing"

---

## Prompt Templates

### GSM8K
```
Solve this math problem step by step. End with #### followed by the numerical answer.
```

### Tool-Use
```
You have access to the following tools:
{tool_schema_json}

User query: {query}
```

### HumanEval
```
{docstring_from_problem}
```

### Arithmetic
```
Calculate: {problem}
Answer:
```

---

## Decoding Parameters

| Task | Temperature | Top-p | Max Tokens | Notes |
|------|-------------|-------|------------|-------|
| GSM8K (train) | 0.8–1.0 | 0.95 | 512 | Sampling for diversity |
| GSM8K (held-out) | 0.0 | 1.0 | 512 | Greedy for evaluation |
| MATH | 0.7–1.0 | 0.95 | 1024 | Varies by run |
| HumanEval | 0.8 | 0.95 | 256 | Code generation |
| Tool-Use | 0.7 | 0.95 | 256 | Structured output |
| Arithmetic | 0.8 | 0.95 | 64 | Short generation |

---

## Key Terminology

| Term | Meaning |
|------|---------|
| **Training reward** | Reward obtained during RL training on sampled prompts |
| **Held-out accuracy** | Accuracy on examples NOT used in training |
| **Reward-environment probe** | Using test-split prompts as training signal (not held-out) |
| **Schema compliance** | Correct JSON structure/tool format (not semantic correctness) |
| **Proxy reward** | Reward that proxies the true goal (e.g., format for capability) |

---

## What This Means for Claims

### Valid Claims
1. ✅ "Training reward improved on GSM8K" (training dynamics)
2. ✅ "ZVF dropped during training" (diagnostic behavior)
3. ✅ "Held-out GSM8K was 82.0%→83.3%" (valid capability comparison)
4. ✅ "Tool-use schema compliance improved" (schema learning)

### Invalid Claims (Not Made in Paper)
1. ❌ "GRPO improves GSM8K capability" (held-out was not significant)
2. ❌ "GRPO learns tool use" (only schema compliance shown)
3. ❌ "HumanEval improved to X%" (test split was probe, not held-out)
4. ❌ "Model learned to use tools correctly" (no execution verified)

---

## Verification Commands

To verify the protocol matches the paper:

```bash
# Verify GSM8K held-out files exist and contain correct splits
python3 scripts/verify_claims_offline.py --claim gsm8k_heldout_nonsignificant

# Verify tool-use reward scope
python3 scripts/verify_claims_offline.py --claim tool_proxy_scope

# Verify proxy harness usage
python3 scripts/verify_claims_offline.py --claim proxy_harnesses
```

---

## Updates and Version

This document reflects the protocol as described in the paper's:
- Abstract (evaluation hierarchy)
- Section 2 (evidence types)
- Section 4 (experimental setup)
- Section Limitations (proxy rewards)

For questions, see `REVIEWER_VERIFICATION.md`.
