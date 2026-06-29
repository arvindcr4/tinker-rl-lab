import os
import re

PATCH_LINES = [
    "try:",
    "    import wandb, torch",
    "    if not getattr(wandb, '_vram_patched', False):",
    "        _old_log = wandb.log",
    "        def _vram_log(data, *args, **kwargs):",
    "            if torch.cuda.is_available():",
    "                data['system/vram_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)",
    "                data['system/vram_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)",
    "                torch.cuda.reset_peak_memory_stats()",
    "            _old_log(data, *args, **kwargs)",
    "        wandb.log = _vram_log",
    "        wandb._vram_patched = True",
    "except ImportError:",
    "    pass"
]

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if "wandb._vram_patched" in content:
        return False
        
    lines = content.split('\n')
    new_lines = []
    patched = False
    
    for line in lines:
        # Find the first wandb.init or wandb.login or wandb.log
        if not patched and re.search(r'\bwandb\.(init|login|log)\(', line):
            indent = line[:len(line) - len(line.lstrip())]
            for p_line in PATCH_LINES:
                new_lines.append(indent + p_line)
            patched = True
        # Also patch before GRPOTrainer or SFTTrainer if wandb is used via report_to
        elif not patched and re.search(r'\b(GRPOTrainer|SFTTrainer|PPOTrainer|Trainer)\(', line):
            indent = line[:len(line) - len(line.lstrip())]
            for p_line in PATCH_LINES:
                new_lines.append(indent + p_line)
            patched = True
            
        new_lines.append(line)
        
    if patched:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Patched {filepath}")
        return True
    return False

for root, dirs, files in os.walk('.'):
    if '.git' in root or '.system_generated' in root:
        continue
    for f in files:
        if f.endswith('.py'):
            process_file(os.path.join(root, f))

