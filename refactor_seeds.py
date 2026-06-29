import glob, re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Change STEPS
    content = re.sub(r'STEPS\s*=\s*(30|50)', 'STEPS = 200', content)
    
    # If file is grpo_gsm8k_base.py, change default steps
    if 'gsm8k_base' in filepath:
        content = re.sub(r'add_argument\("--steps", type=int, default=50\)', 'add_argument("--steps", type=int, default=200)', content)
        # Maybe that's all for gsm8k_base, it's parameterized.
        with open(filepath, 'w') as f:
            f.write(content)
        return

    # Check if already has seed loop
    if 'for seed in range(5):' in content or 'for seed in' in content:
        with open(filepath, 'w') as f:
            f.write(content)
        return

    # For the experiment scripts, we need to wrap the main part in a seed loop
    # The main part starts around `print(f"[{EXP_NAME}] Connecting...")` or `print("Connecting to Tinker...")`
    
    parts = re.split(r'(print\([f]?"\[?.*EXP_NAME\]?.*Connecting.*\nsvc = tinker\.ServiceClient)', content)
    if len(parts) < 2:
        parts = re.split(r'(print\("Connecting to Tinker..."\)\nsvc = tinker\.ServiceClient)', content)
    
    if len(parts) >= 3:
        header = parts[0]
        # ensure SEEDS = 5 is added
        header = re.sub(r'(STEPS = 200)', r'NUM_SEEDS = 5\n\1', header)
        
        main_code = parts[1] + parts[2]
        
        # indent main_code
        indented_main = '\n'.join('    ' + line if line.strip() else line for line in main_code.split('\n'))
        
        # fix the ckpt paths so they don't overwrite
        # state_{step+1} -> state_seed{seed}_{step+1}
        indented_main = re.sub(r'name=f"state_\{step \+ 1\}"', r'name=f"state_seed{seed}_{step + 1}"', indented_main)
        indented_main = re.sub(r'name=f"step_\{step \+ 1\}"', r'name=f"step_seed{seed}_{step + 1}"', indented_main)
        indented_main = re.sub(r'name="step_0"', r'name=f"seed{seed}_step_0"', indented_main)
        indented_main = re.sub(r'name="final"', r'name=f"seed{seed}_final"', indented_main)
        
        loop_code = f"""
for seed in range(NUM_SEEDS):
    print(f"\\n{'='*50}\\nRunning seed {{seed}} ({{seed+1}}/{{NUM_SEEDS}})\\n{'='*50}")
    random.seed(seed)
    torch.manual_seed(seed)
    
{indented_main}
"""
        new_content = header + loop_code
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
    else:
        print(f"Could not split {filepath}")

for f in ['grpo_exp_a_baseline.py', 'grpo_exp_b_high_lr.py', 'grpo_exp_c_low_temp.py', 'grpo_exp_d_xlam.py', 'grpo_tooluse_tinker.py', 'grpo_gsm8k_base.py']:
    process_file(f)

