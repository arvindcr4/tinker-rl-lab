import os

replacements = {
    "platform_hybrid/sem 4 work/PROVENANCE.md": [
        ("../paper/neurips_2026_variants/main_workshop.tex", "../../platform_hybrid/paper/neurips_2026_variants/main_workshop.tex"),
        ("../sem 3 work/submissions/neurips-main-track/", "../../platform_hybrid/sem 3 work/submissions/neurips-main-track/"),
        ("../sem%203%20work/submissions/neurips-main-track/", "../../platform_hybrid/sem%203%20work/submissions/neurips-main-track/"),
        ("../paper/sections/_shared_author.tex", "../../platform_hybrid/paper/sections/_shared_author.tex"),
        ("../experiments/results/findings_ledger.jsonl", "../../platform_hybrid/experiments/results/findings_ledger.jsonl"),
        ("../PROJECT_HISTORY.md", "../../PROJECT_HISTORY.md"),
        ("../sem 3 work/", "../../platform_hybrid/sem 3 work/"),
        ("../sem%203%20work/", "../../platform_hybrid/sem%203%20work/")
    ],
    "platform_hybrid/sem 4 work/README.md": [
        ("../reports/esa_phase1/", "../../platform_tinker/reports/esa_phase1/"),
        ("../sem 3 work/submissions/neurips-main-track/", "../../platform_hybrid/sem 3 work/submissions/neurips-main-track/"),
        ("../sem%203%20work/submissions/neurips-main-track/", "../../platform_hybrid/sem%203%20work/submissions/neurips-main-track/"),
        ("../REPRODUCE.md", "../../REPRODUCE.md"),
        ("../ARTIFACT.md", "../../ARTIFACT.md"),
        ("../experiments/experiment_summary.md", "../../platform_hybrid/experiments/experiment_summary.md"),
        ("../experiments/results/", "../../platform_hybrid/experiments/results/")
    ],
    "platform_hybrid/sem 3 work/README.md": [
        ("../sem 4 work/", "../../platform_hybrid/sem 4 work/"),
        ("../sem%204%20work/", "../../platform_hybrid/sem%204%20work/")
    ]
}

for filepath, reps in replacements.items():
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        
        for old, new in reps:
            # We want to replace occurrences, but only if they haven't been replaced yet
            # Since old starts with '../', we can just replace
            content = content.replace(old, new)
            
        with open(filepath, "w") as f:
            f.write(content)

