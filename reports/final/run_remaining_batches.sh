#!/bin/bash
# Resume batches 2-10, skip failures, alternate sessions
OUTPUT_DIR="/home/arvind/tinker-rl-lab/reports/final/chatgpt_responses"
mkdir -p "$OUTPUT_DIR"

send_batch() {
    local SESSION=$1
    local BATCH=$2
    local QUESTIONS=$3

    echo "[Batch $BATCH] Starting on $SESSION..."

    # Navigate to new chat
    BROWSE_SESSION=$SESSION bb browse open "https://chatgpt.com" 2>&1 > /dev/null
    sleep 3

    # Type via eval
    ESCAPED=$(echo "$QUESTIONS" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
    BROWSE_SESSION=$SESSION bb browse eval "
    var ta = document.querySelector('textarea, [contenteditable=\"true\"]');
    if (ta) { ta.focus(); document.execCommand('selectAll'); document.execCommand('insertText', false, $ESCAPED); ta.dispatchEvent(new Event('input', {bubbles:true})); 'ok' } else { 'fail' }
    " 2>&1 > /dev/null
    sleep 2

    # Find and click send
    SEND_REF=$(BROWSE_SESSION=$SESSION bb browse snapshot 2>&1 | python3 -c "
import json,sys,re
data=json.loads(sys.stdin.read())
m=re.search(r'\[(\d+-\d+)\] button: Send prompt', data.get('tree',''))
print(m.group(1) if m else 'FAIL')
")

    if [ "$SEND_REF" = "FAIL" ]; then
        echo "[Batch $BATCH] ERROR: No send button"
        return 1
    fi

    BROWSE_SESSION=$SESSION bb browse click "$SEND_REF" 2>&1 > /dev/null
    echo "[Batch $BATCH] Sent, waiting..."

    # Poll for response (max 3 min for instant mode)
    for i in $(seq 1 18); do
        sleep 10
        RESULT=$(BROWSE_SESSION=$SESSION bb browse eval "
        var s=document.querySelector('button[aria-label=\"Stop streaming\"]');
        var t=document.querySelectorAll('[data-message-author-role=\"assistant\"]');
        JSON.stringify({gen:!!s, len: t.length>0 ? t[t.length-1].innerText.length : 0})
        " 2>&1)

        # Parse result
        GEN=$(echo "$RESULT" | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); d=json.loads(r.get('result','{}')); print(d.get('gen',True))" 2>/dev/null || echo "True")
        LEN=$(echo "$RESULT" | python3 -c "import json,sys; r=json.loads(sys.stdin.read()); d=json.loads(r.get('result','{}')); print(d.get('len',0))" 2>/dev/null || echo "0")

        if [ "$GEN" = "False" ] && [ "${LEN:-0}" -gt 200 ] 2>/dev/null; then
            echo "[Batch $BATCH] Done! ($LEN chars)"
            BROWSE_SESSION=$SESSION bb browse eval "
            var t=document.querySelectorAll('[data-message-author-role=\"assistant\"]');
            t[t.length-1].innerText
            " 2>&1 | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['result'])" > "$OUTPUT_DIR/batch_${BATCH}.txt" 2>/dev/null
            return 0
        fi
        echo "[Batch $BATCH] ... (${i}x10s, ${LEN:-0} chars)"
    done
    echo "[Batch $BATCH] TIMEOUT - skipping"
    return 0  # Don't fail, just skip
}

# Batch questions
B2="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q6: Training reward 30.5% reported as single number - was it measured across 5 seeds? Is the 30.5% vs 83.3% discrepancy potentially seed-specific?
Q7: 3B-4B capacity threshold from few model sizes with no parametric test - should we fit sigmoid or threshold regression with CIs on breakpoint?
Q8: MoE routing volatility 2.43x - what denominator metric and is this statistically significant under bootstrap test?
Q9: Code improvement 32% to 40% on 50 items = 4 problems. Fisher exact test p-value? Is this statistically significant at p<0.05?
Q10: Cross-seed stability with 5 seeds SD=2.2% - have we compared to published GRPO baselines to show stability is meaningfully lower?"

B3="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q11: ToolBench, APIBench, ToolAlpaca study tool-calling with held-out evals - why not cited? How does 0% to 92% compare?
Q12: DeepSeekMath and DAPO extend GRPO with clip-higher and overlong penalties for math - how to position GSM8K findings against these variants?
Q13: SFT+GRPO complementarity echoes ReST, STaR, iterative DPO - how to differentiate from warm-starting RL literature?
Q14: Concurrent work STILL and Tiny-GRPO target sub-3B models - do they report similar capacity thresholds? Strengthen or weaken novelty?
Q15: MoE routing volatility should be contextualized against Switch Transformer, GLaM, Mixtral instability literature - is 2.43x surprising?"

B4="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q16: Agentic LLM fine-tuning in title but never defined - provide operational definition distinguishing agentic from non-agentic in GRPO context.
Q17: Novel vs confirmatory classification criteria are implicit - should we add Methods subsection stating evidence threshold?
Q18: Are figure captions self-contained per NeurIPS standards? Can reader interpret results without main text?
Q19: Paper conflates synthetic-real gap (data distribution) with train-test reward discrepancy (metric alignment) - how to separate?
Q20: Group sizes 4-16 and LoRA ranks 8-64 mentioned in passing - should we add configuration table mapping each result to exact hyperparameters?"

B5="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q21: 3B-4B threshold interpolated from only 2 model sizes (Qwen2.5-3B and Qwen3.5-4B) - reframe as hypothesis requiring validation?
Q22: With confirmatory findings labeled as such, contribution reduces to 3 novel claims - should abstract foreground these three specifically?
Q23: MoE routing volatility is purely observational - should we propose mechanism (reward gradient interfering with load balancing loss)?
Q24: Cross-seed stability under multi-seed eval is standard practice not a finding - reframe unless we show GRPO more stable than alternatives?
Q25: Would unified mechanistic narrative (GRPO succeeds when reward is dense and verifiable) strengthen framing vs treating each task separately?"

B6="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q26: All models are Qwen/Llama - do capacity threshold and MoE volatility generalize to Mistral, Phi-3, Gemma, or are they Qwen-specific?
Q27: At what step count do improvements saturate? Would 200-500 steps change capacity threshold or close training/test reward gap?
Q28: Would 92% tool-call accuracy transfer to OpenAI/Anthropic/Google function-calling formats, or is model learning schema-specific patterns?
Q29: Paper studies mostly single-turn tool calling - how does capacity threshold change for multi-turn agentic trajectories?
Q30: GSM8K may be saturated for 8B models - do GRPO dynamics hold on harder benchmarks like MATH-500 or AIME?"

B7="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q31: No ablation on group size (4 vs 8 vs 16) - since group size controls GRPO advantage estimator variance, is this the most critical ablation missing?
Q32: LoRA rank-speed tradeoff reported but accuracy vs rank curve missing - add ablation at 8,16,32,64 to find if monotonic or has knee?
Q33: SFT+GRPO complementarity needs matched-compute control where GRPO alone runs 2x steps - otherwise SFT may just be better initialization for any RL.
Q34: To isolate GRPO from reward design, include SFT baseline on GRPO-generated positive rollouts - separates policy optimization from data curriculum.
Q35: Tool calling reward function not ablated - is 0% to 92% driven by reward structure (format, correctness, partial credit) rather than GRPO?"

B8="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q36: Training reward 30.5% while test 83.3% suggests reward misspecification - formal analysis of why optimizing training reward does not maximize test accuracy?
Q37: Two-phase learning mirrors curriculum learning theory - grounding in theory would predict phase transition as function of capacity and reward sparsity?
Q38: GRPO advantage normalizes within group - theoretical argument for why group sizes 4-16 appropriate for observed reward scales?
Q39: Capacity threshold implies phase transition in expressivity - theoretical basis (scaling laws, NTK, circuit interpretability) for 3B-4B boundary?
Q40: MoE volatility from load balancing loss conflicting with policy gradient - formalize as optimization interference, propose mitigation?"

B9="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q41: No PPO/REINFORCE baselines is rejection risk - best rebuttal for why GRPO is appropriate primary comparison without running additional experiments?
Q42: 50 training steps questioned as meaningful RL - prepare compute-normalized comparison showing 50 GRPO steps matches more SFT steps?
Q43: Custom tool-calling eval lacks standardization - should we adopt BFCL/ToolBench-eval for subset, or is releasing eval code sufficient?
Q44: Qwen3 has existing RLHF - reviewers may argue GRPO unwinding alignment not clean RL. Address by reporting base (non-instruct) results?
Q45: 17 runs insufficient for breadth of claims - prepare table mapping each claim to specific supporting runs?"

B10="NeurIPS paper on GRPO for agentic LLM fine-tuning. Answer these 5 concisely:
Q46: Provide practitioner decision flowchart (model size, task type, data) mapping to recommended GRPO configuration - making findings actionable.
Q47: What exact repository structure, license, and data restrictions needed for full reproducibility?
Q48: LoRA rank-speed tradeoff cost implications - quantify wall-clock time and GPU-hours at each rank for practitioners.
Q49: Cloud GPU variability introduces noise - what hardware config and framework versions to report for reproducibility?
Q50: Sub-3B GRPO failure as negative result - how to quantify with confidence so it can be cited as engineering guideline?"

echo "=== Resuming batches 2-10 ==="
for BATCH_NUM in 2 3 4 5 6 7 8 9 10; do
    if [ $((BATCH_NUM % 2)) -eq 0 ]; then
        SESSION="chatgpt-pro2"
    else
        SESSION="chatgpt-pro"
    fi

    BATCH_VAR="B${BATCH_NUM}"
    QUESTIONS="${!BATCH_VAR}"

    send_batch "$SESSION" "$BATCH_NUM" "$QUESTIONS"
    echo ""
done

echo "=== All batches attempted ==="
ls -la "$OUTPUT_DIR/batch_"*.txt 2>/dev/null
echo "Total files: $(ls "$OUTPUT_DIR/batch_"*.txt 2>/dev/null | wc -l)"
