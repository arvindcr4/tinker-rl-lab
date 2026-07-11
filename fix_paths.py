import os
import re

# 1. Fix dead skill cross-links
skill_dir = ".claude/skills"
dead_links_removed = 0
for root, _, files in os.walk(skill_dir):
    if "SKILL.md" in files:
        filepath = os.path.join(root, "SKILL.md")
        with open(filepath, "r") as f:
            content = f.read()
        
        # Find all markdown links to sibling skills
        # e.g. [deep-research](../deep-research/)
        # We also need to handle the comma and space before/after if any
        def replace_link(match):
            name = match.group(1)
            target = match.group(2)
            # Check if target skill exists
            target_path = os.path.join(skill_dir, target.strip('/'))
            if not os.path.exists(target_path):
                global dead_links_removed
                dead_links_removed += 1
                return "" # Remove the link
            return match.group(0) # Keep the link
        
        # Regex to match [name](../target/)
        # We want to remove it entirely, maybe cleaning up commas
        # A simpler way: just remove the `[name](../target/)` substring if target is dead.
        # Then clean up `,,` or `, ,` and trailing/leading commas on the line.
        
        new_content = content
        links = re.findall(r'\[([^\]]+)\]\(\.\./([^\)]+)\)', content)
        for name, target in links:
            target_path = os.path.join(skill_dir, target.strip('/'))
            if not os.path.exists(target_path):
                link_str = f"[{name}](../{target})"
                new_content = new_content.replace(link_str, "")
                dead_links_removed += 1
        
        # Clean up empty spaces and commas left behind
        lines = []
        for line in new_content.split('\n'):
            if line.startswith('- Upstream:') or line.startswith('- Downstream:') or line.startswith('- See also:'):
                prefix = line.split(':')[0] + ':'
                rest = line.split(':')[1]
                # split by comma, filter empty
                items = [x.strip() for x in rest.split(',') if x.strip()]
                if items:
                    lines.append(f"{prefix} {', '.join(items)}")
                else:
                    pass # skip the line if no links left
            else:
                lines.append(line)
                
        new_content = '\n'.join(lines)
        if new_content != content:
            with open(filepath, "w") as f:
                f.write(new_content)

print(f"Removed {dead_links_removed} dead skill references.")

# 2. Fix stale moved-path links
files_to_fix = [
    "README.md",
    "PROJECT_HISTORY.md",
    "ARTIFACT.md",
    "REPRODUCE.md",
    "platform_hybrid/sem 4 work/README.md",
    "platform_hybrid/sem 4 work/PROVENANCE.md",
    "platform_hybrid/paper/figures/v2/README.md",
    "platform_hybrid/sem 3 work/README.md",
]

replacements = [
    ("sem 3 work/", "platform_hybrid/sem 3 work/"),
    ("paper/main.tex", "platform_hybrid/paper/main.tex"),
    ("reports/final/", "platform_tinker/reports/final/"),
    ("scripts/smoke_test.sh", "platform_modal/scripts/smoke_test.sh"),
    ("scripts/regenerate_figures.py", "platform_modal/scripts/regenerate_figures.py"),
    ("sem 4 work/", "platform_hybrid/sem 4 work/"),
]

for filepath in files_to_fix:
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        
        new_content = content
        for old, new in replacements:
            # We want to replace occurrences, but be careful not to double replace.
            # E.g. if it already has platform_hybrid/sem 3 work/, we shouldn't make it platform_hybrid/platform_hybrid/sem 3 work/
            # A simple regex can do this: replace `old` if it's not preceded by `platform_hybrid/` or `platform_tinker/` or `platform_modal/` or `../`
            
            # Since Python regex doesn't support variable length negative lookbehind easily, we can just replace and then fix doubles.
            new_content = new_content.replace(old, new)
            new_content = new_content.replace(f"platform_hybrid/{new}", new)
            new_content = new_content.replace(f"platform_modal/{new}", new)
            new_content = new_content.replace(f"platform_tinker/{new}", new)
            new_content = new_content.replace(f"../{new}", f"../{old}") # revert if it was a relative link!
            
            # Wait, the prompt said: "Broken Semester 4 relative links → correct repository-root-relative targets."
            # If the file is in platform_hybrid/sem 4 work/, it shouldn't use `../sem 3 work/`, it should use `platform_hybrid/sem 3 work/` repository-root-relative.
            # But wait, how do I know it's repository-root-relative? Maybe the prompt meant that `../sem 3 work/` should become `platform_hybrid/sem 3 work/`?
            # Let's just do a naive replace and fix `../platform_hybrid/...`
            
        with open(filepath, "w") as f:
            f.write(new_content)

