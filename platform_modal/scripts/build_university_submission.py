import os
import shutil
import zipfile
import subprocess
from pathlib import Path

# Directories
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "university_submission"
FINAL_ZIP = ROOT / "University_Defense_Submission.zip"

# Clean output dir
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

# 1. Package the Codebase cleanly using git archive
print("Packaging clean codebase...")
CODE_ZIP = OUTPUT_DIR / "source_code.zip"
subprocess.run(["git", "archive", "--format=zip", "-o", str(CODE_ZIP), "HEAD"], cwd=ROOT, check=True)

# 2. Gather Artifacts
ARTIFACTS = {
    "paper/main.pdf": "research_paper.pdf",
    "reports/final/capstone_final_report_submission/capstone_final_report.pdf": "capstone_report.pdf",
    "reports/final/grpo_agentic_llm_paper.pdf": "grpo_agentic_llm_paper.pdf",
    "reports/final/final_capstone_presentation.pptx": "defense_presentation.pptx",
    "README.md": "README.md"
}

for src_rel, dst_name in ARTIFACTS.items():
    src_path = ROOT / src_rel
    dst_path = OUTPUT_DIR / dst_name
    if src_path.exists():
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src_rel} -> {dst_name}")
    else:
        print(f"Warning: Could not find {src_rel}")

# 3. Create Final ZIP
print(f"Creating {FINAL_ZIP.name}...")
with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            filepath = Path(root) / file
            # Store in zip without the parent path getting too messy
            arcname = Path("University_Defense_Submission") / filepath.name
            zf.write(filepath, arcname)

print(f"\nDone! Final package created at: {FINAL_ZIP}")
