import glob, re, os

# Update ai-scientist
for f in glob.glob('ai-scientist-v2-integration/ai_scientist/ideas/*.py'):
    with open(f, 'r') as file:
        content = file.read()
    
    # Change NUM_SEEDS = 3 to 5
    content = re.sub(r'NUM_SEEDS\s*=\s*3', 'NUM_SEEDS = 5', content)
    # Change STEPS = 30 to 200
    content = re.sub(r'STEPS\s*=\s*30', 'STEPS = 200', content)
    
    with open(f, 'w') as file:
        file.write(content)

# ai-scientist-template/experiment.py
with open('ai-scientist-template/experiment.py', 'r') as file:
    content = file.read()
content = re.sub(r'NUM_SEEDS\s*=\s*3', 'NUM_SEEDS = 5', content)
content = re.sub(r'NUM_STEPS\s*=\s*30', 'NUM_STEPS = 200', content)
with open('ai-scientist-template/experiment.py', 'w') as file:
    file.write(content)

# Update experiments/tinker-runs scripts
# These are identical to the root scripts but stored inside experiments/tinker-runs/scripts/
# We can just apply the same refactoring script we wrote earlier.
from scratch.refactor import process_file
for f in glob.glob('experiments/tinker-runs/scripts/grpo_exp_*.py'):
    process_file(f)

# Update retry_llama33_70b_seeds.py and run_llama33_70b_seeds.py
for f in glob.glob('experiments/tinker-runs/scripts/*llama33*.py'):
    with open(f, 'r') as file:
        content = file.read()
    content = re.sub(r'STEPS\s*=\s*30', 'STEPS = 200', content)
    with open(f, 'w') as file:
        file.write(content)
        
# Update wave6_ablations.py
with open('experiments/tinker-runs/wave6_ablations.py', 'r') as file:
    content = file.read()
content = re.sub(r'STEPS\s*=\s*30', 'STEPS = 200', content)
content = re.sub(r'steps=30', 'steps=200', content)
with open('experiments/tinker-runs/wave6_ablations.py', 'w') as file:
    file.write(content)

# TODO: Address remaining limitations from adversarial_review.md
# - ZVF Metric Fragility (Saturates at 1.0 for format-gated tasks): No fix applicable in this script. Needs to be addressed by using ERF.
# - The Closed-Source Confound (Tinker API is a black box): No fix applicable.
# - Failure to Prove Generalization (Gains on GSM8K/HumanEval not statistically significant): No fix applicable here, might require evaluating on a larger hold-out set.

print("Updates applied")
