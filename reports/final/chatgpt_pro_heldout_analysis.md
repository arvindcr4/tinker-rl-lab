# ChatGPT Pro (Extended Pro) — Held-Out Results Critical Analysis
**Date:** 2026-04-04
**Session:** chatgpt-pro2

---

## Summary

**Verdict: 83.3% is plausible, but NOT strong evidence that GRPO caused the performance.**

The train/test inversion is explainable because the two numbers measure very different things:
- Training reward: online sampled completion reward under high-temperature exploration (T≈0.9)
- Test accuracy: greedy argmax decoding on final checkpoint

## Key Points

### 1. Validity Check
- 83.3% on GSM8K is plausible for Qwen3-8B (official report: 89.84% for Qwen3-8B-Base in 4-shot CoT)
- Community reproductions show 82.49% / 85.06% for Qwen3-8B-Base depending on extraction mode
- **Conclusion: the absolute test number does not by itself show GRPO helped much**

### 2. Why 30.5% Training Reward and 83.3% Test Accuracy Can Coexist
- **Decode mismatch**: Training uses T=0.8-1.0 sampling; test uses greedy. Model can have correct mode while spraying bad paths under sampling.
- **Trajectory averaging**: 30.5% averaged over whole training trajectory, not final checkpoint. Final sampled reward could be ~40%.
- **Hotter-than-recommended decoding**: Qwen3 recommends T=0.6-0.7; training used 0.8-1.0.
- **Math**: If greedy correct on 83.3% and sampled correct 36.6% of the time → 0.833 × 0.366 ≈ 0.305

### 3. Methodological Concerns
- **Answer extraction mismatch**: Training parser vs eval parser may use different formats (#### vs \boxed{})
- **Template/thinking-mode mismatch**: enable_thinking, reasoning tags, history handling
- **Checkpoint-loading failure**: If LoRA adapter failed to load, base model already scores ~83%
- **Contamination**: GSM8K is public; some models drop up to 8 points on GSM1k
- **Small sample**: 200-example test slice has limited power

### 4. Recommended Paper Interpretation
> "The online GRPO reward is not directly comparable to final held-out accuracy because it is measured under high-temperature sampled rollouts during training. Under our current budget, the final checkpoint attains high GSM8K accuracy, but attribution of this performance to GRPO requires matched baseline and parser/template controls."

- Do NOT call 30.5% "training-set accuracy" → call it "mean online sampled reward during RL"
- Include a base-model control under the exact same evaluation harness

### 5. Definitive Disambiguating Experiment
Run **crossed-evaluation audit** on same prompts with 3 model states × 2 decoding regimes × 2 parsers:
- Models: Base checkpoint, GRPO LoRA checkpoint, GRPO checkpoint with adapters disabled
- Decoding: Greedy (test setup), Sampled (T=0.9, group 8, matching training)
- Parsers: Training reward parser, Evaluation parser
- Compute: greedy pass@1, sampled per-completion accuracy, sampled pass@8, parser disagreement rate

**Interpretation guide:**
| Result | Meaning |
|--------|---------|
| Base ≈ GRPO on greedy+sampled | GRPO adds little; Qwen3-8B was already strong |
| GRPO > base on greedy, sampled stays low | GRPO helps, but training metric misleading |
| Training parser << eval parser | Extraction bug / format mismatch |
| Adapter-enabled ≈ adapter-disabled | Checkpoint not loaded |
| All matched, GRPO still better | Result is probably real |
