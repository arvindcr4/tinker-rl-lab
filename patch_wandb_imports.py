import os
import re

PATCH_LINES = [
    "    try:",
    "        import torch, wandb",
    "        if not getattr(wandb, '_vram_patched', False):",
    "            _old_log = wandb.log",
    "            def _vram_log(data, *args, **kwargs):",
    "                if torch.cuda.is_available():",
    "                    data['system/vram_peak_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)",
    "                    data['system/vram_reserved_gb'] = torch.cuda.max_memory_reserved() / (1024**3)",
    "                    torch.cuda.reset_peak_memory_stats()",
    "                _old_log(data, *args, **kwargs)",
    "            wandb.log = _vram_log",
    "            wandb._vram_patched = True",
    "    except ImportError:",
    "        pass"
]

def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if any("wandb._vram_patched" in l for l in lines):
        return False
        
    new_lines = []
    patched = False
    
    for line in lines:
        new_lines.append(line)
        # Check if line imports wandb
        if re.search(r'^(?:\s*)import .*wandb', line) or re.search(r'^(?:\s*)from wandb import', line):
            indent = line[:len(line) - len(line.lstrip())]
            for p_line in PATCH_LINES:
                new_lines.append(indent + p_line[4:] + "\n")
            patched = True
            
    if patched:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        print(f"Patched {filepath}")
        return True
    return False

for root, dirs, files in os.walk('.'):
    if '.git' in root or '.system_generated' in root:
        continue
    for f in files:
        if f.endswith('.py') and f != "patch_wandb_imports.py":
            process_file(os.path.join(root, f))
