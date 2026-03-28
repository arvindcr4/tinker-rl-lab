# AI Scientist Integration for GRPO Research

This directory contains a complete [Sakana AI Scientist](https://github.com/SakanaAI/AI-Scientist) template for your GRPO math reasoning research. AI Scientist will autonomously generate ideas, run experiments, write papers, and self-review them.

## What AI Scientist Will Do

Given this template, AI Scientist will:

1. **Generate ideas** — novel variations on GRPO training (reward shaping, curriculum, hyperparameters, etc.)
2. **Modify `experiment.py`** — implement each idea as code changes
3. **Run experiments** — train on GSM8K with 3 seeds, collect metrics
4. **Write a paper** — full LaTeX paper with results, citations, and analysis
5. **Self-review** — NeurIPS-style review with Accept/Reject decision
6. **Optionally improve** — revise based on review feedback

Cost: ~$15 per generated paper (Claude Sonnet), ~$5 with DeepSeek.

## Setup

### 1. Clone AI Scientist

```bash
cd ~
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist
conda create -n ai-scientist python=3.11 -y
conda activate ai-scientist
pip install -r requirements.txt
```

### 2. Install GPU dependencies

```bash
# For local GPU (A10/A100/T4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets trl peft accelerate bitsandbytes
```

### 3. Set API keys

```bash
# At least one LLM provider (Claude recommended for best results)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI required for the review stage (hardcoded in AI Scientist)
export OPENAI_API_KEY="sk-..."

# Optional: better literature search
export S2_API_KEY="..."  # Semantic Scholar
```

### 4. Install LaTeX

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# Or minimal (faster install)
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-bibtex-extra biber
```

### 5. Copy template into AI Scientist

```bash
# From tinker-rl-lab root
cp -r ai-scientist-template ~/AI-Scientist/templates/grpo_gsm8k
```

### 6. Generate real baseline (recommended)

The provided `run_0/final_info.json` has placeholder values. For best results, run the actual baseline:

```bash
cd ~/AI-Scientist/templates/grpo_gsm8k
python experiment.py --out_dir run_0
python plot.py
```

This takes ~30-90 min on a single GPU depending on hardware.

### 7. Copy LaTeX style files

```bash
# Copy from an existing AI Scientist template
cp ~/AI-Scientist/templates/nanoGPT/latex/iclr2024_conference.sty ~/AI-Scientist/templates/grpo_gsm8k/latex/
cp ~/AI-Scientist/templates/nanoGPT/latex/iclr2024_conference.bst ~/AI-Scientist/templates/grpo_gsm8k/latex/
cp ~/AI-Scientist/templates/nanoGPT/latex/fancyhdr.sty ~/AI-Scientist/templates/grpo_gsm8k/latex/
```

### 8. Launch

```bash
cd ~/AI-Scientist

# Generate 2 ideas with Claude (good starting point)
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment grpo_gsm8k \
    --num-ideas 2

# Or with DeepSeek (cheaper)
python launch_scientist.py \
    --model "deepseek-chat" \
    --experiment grpo_gsm8k \
    --num-ideas 5

# With improvement loop (paper gets revised after review)
python launch_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment grpo_gsm8k \
    --num-ideas 2 \
    --improvement
```

## What Ideas Will It Generate?

Based on the seed ideas and task description, expect ideas like:

- **Reward shaping**: partial credit for format compliance, step-level rewards
- **Curriculum learning**: easy-first scheduling based on problem difficulty
- **GRPO variants**: different group sizes, KL penalties, advantage normalization
- **Prompt engineering**: chain-of-thought templates, few-shot examples in prompts
- **Learning rate schedules**: warmup, cosine decay, cyclical LR
- **Data augmentation**: rephrasing problems, augmenting with similar problems

## How to Use Results for Your Thesis/Paper

### Incorporating AI Scientist findings into your existing paper

1. **Check `results/grpo_gsm8k/` for generated papers** — each idea gets its own folder with:
   - `{idea_name}.pdf` — the generated paper
   - `review.txt` — automated review with scores
   - `experiment.py` — the modified code (shows exactly what was changed)
   - `final_info.json` — quantitative results

2. **Mine for ideas, not papers** — the generated papers are drafts. The value is in:
   - Novel experiment configurations that improve your metric
   - Ablation results you can cite or reproduce at scale on Tinker
   - Literature connections you may have missed (via auto-citations)

3. **Scale winners on Tinker** — when AI Scientist finds a configuration that improves accuracy:
   ```bash
   # Adapt the winning experiment.py changes back to your Tinker script
   # e.g., if partial credit reward worked, add it to grpo_gsm8k_base.py
   ```

4. **Mandatory disclosure** — AI Scientist's license requires disclosing AI use in any resulting manuscript. Add a note like:
   > "Preliminary experiment configurations were explored using the AI Scientist framework (Lu et al., 2024). All results reported in this paper were independently reproduced and validated by the authors."

### For the capstone thesis

The thesis can include a section on "AI-assisted research methodology" describing how AI Scientist was used for hyperparameter exploration and idea generation, with the human researchers validating and scaling the best findings.

## Running on Cloud GPUs

### vast.ai (cheapest)

```bash
# In your vast.ai instance
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist && pip install -r requirements.txt
pip install torch trl peft transformers datasets accelerate bitsandbytes
cp -r /path/to/ai-scientist-template templates/grpo_gsm8k
python launch_scientist.py --model "deepseek-chat" --experiment grpo_gsm8k --num-ideas 5
```

### Google Colab

Use the template with a Colab notebook:
```python
!git clone https://github.com/SakanaAI/AI-Scientist.git
%cd AI-Scientist
!pip install -r requirements.txt
!pip install trl peft accelerate bitsandbytes
# Copy template files, set API keys, then launch
```

## Template File Reference

| File | Purpose | Modify? |
|------|---------|---------|
| `experiment.py` | GRPO training script (AI Scientist modifies this) | Yes — AI Scientist's core target |
| `plot.py` | Visualization (AI Scientist updates plots) | Yes — updated per experiment |
| `prompt.json` | System prompt + domain description | Rarely — tune if ideas are off-target |
| `seed_ideas.json` | Example ideas (few-shot) | Rarely — add domain-specific ideas |
| `latex/template.tex` | Paper template with embedded references | No — AI Scientist fills in content |
| `run_0/final_info.json` | Baseline results | Once — regenerate with real baseline |

## Open-Ended Evolution Mode

AI Scientist also supports an **open-ended evolution** mode where it builds on previous results continuously. This is ideal for a sustained research campaign:

```bash
cd ~/AI-Scientist
python experimental/launch_oe_scientist.py \
    --model "claude-3-5-sonnet-20241022" \
    --experiment grpo_gsm8k \
    --num-ideas 10 \
    --num-rounds 5
```

This generates ideas → runs experiments → uses results to generate _better_ ideas → repeat. After 5 rounds, you'll have a chain of progressively refined experiments.
