import os
import re
import glob

PAPER_DIR = "/home/claude/tinker-rl-lab/paper/sections"

def apply_regex(filename, pattern, repl):
    filepath = os.path.join(PAPER_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Skipping {filepath}, does not exist")
        return
    with open(filepath, "r") as f:
        content = f.read()
    
    new_content = re.sub(pattern, repl, content)
    
    if new_content != content:
        with open(filepath, "w") as f:
            f.write(new_content)
        print(f"Updated {filename}")
    
def main():
    # 1. Figure placeholders (remove them)
    # The pattern is typically \fbox{...[Figure placeholder...} or similar.
    # Let's remove the whole \fbox{...} line if it contains "Figure placeholder"
    for file in glob.glob(os.path.join(PAPER_DIR, "*.tex")):
        with open(file, "r") as f:
            content = f.read()
        
        # Regex to remove lines containing "Figure placeholder"
        new_content = re.sub(r'\\fbox{.*?Figure placeholder.*?}.*?\n', '', content, flags=re.DOTALL)
        # Also handle variants that might not have \fbox on the same line
        new_content = re.sub(r'^[^\n]*Figure placeholder[^\n]*\n', '', new_content, flags=re.MULTILINE)
        
        if new_content != content:
            with open(file, "w") as f:
                f.write(new_content)
            print(f"Removed placeholders from {os.path.basename(file)}")

    # 2. Cross references
    # In scaling_laws.tex
    apply_regex("scaling_laws.tex", r'\\secref\{sec:zvf\}', r'companion paper on ZVF')
    apply_regex("scaling_laws.tex", r'\\secref\{sec:zvf-cross-experiment\}', r'companion paper on cross-experiment ZVF')
    
    # In zvf_cross_experiment_diagnostic.tex
    apply_regex("zvf_cross_experiment_diagnostic.tex", r'Section~\\ref\{sec:variance-honesty\}', r'the companion paper on variance honesty')
    
    # In zvf_iter50.tex
    apply_regex("zvf_iter50.tex", r'\\S\\ref\{sec:group-size-iter31\}', r'the companion paper on group size')
    
    # In appendix_zvf_formalization.tex
    apply_regex("appendix_zvf_formalization.tex", r'Section~\\ref\{sec:extended-related-work\}', r'the extended related work companion document')
    
    # 3. P3 Gradient Contradiction
    apply_regex("p3_intro.tex", 
                r'an explicit validation, against measured gradients', 
                r'an explicit validation, via observable signatures in logged diagnostics')
    apply_regex("p3_intro.tex",
                r'against measured GRPO gradients and',
                r'against logged GRPO diagnostic signatures and')
                
    # 4. Viva slides leakage
    apply_regex("_shared_methods.tex",
                r'The viva slides use the second roster;',
                r'Supplementary presentation materials use the second roster;')
                
    # 5. Lab Notebook Vocabulary (Simple sweep)
    # This is broad, but we can catch obvious ones. 
    for file in glob.glob(os.path.join(PAPER_DIR, "p[5678]*.tex")):
        with open(file, "r") as f:
            content = f.read()
            
        new_content = content.replace("this iter", "this iteration")
        new_content = new_content.replace("mint recommendation", "primary recommendation")
        new_content = new_content.replace("worktree", "repository")
        new_content = new_content.replace("PASS/FAIL", "Success/Failure")
        
        if new_content != content:
            with open(file, "w") as f:
                f.write(new_content)
            print(f"Cleaned vocab in {os.path.basename(file)}")

    # 6. P8 Abstract Framing
    apply_regex("p8_abstract.tex",
                r'The current reproducible quick artifacts\s*\([^)]+\)\s*are negative',
                r'The empirical results are negative')
    apply_regex("p8_abstract.tex",
                r'We therefore do\s*not use the older \$0.975\$/\$0.948\$ internal single-run record as a headline\s*result\. The tree keeps',
                r'The gradient-boosted tree keeps')
    apply_regex("p8_abstract.tex",
                r'by our internal estimate',
                r'by our estimates')

if __name__ == "__main__":
    main()
