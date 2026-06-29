import re
import sys
from pathlib import Path

# Provide standard file context lazily to avoid redundant I/O on import
class AuditContext:
    def __init__(self):
        self.ROOT = Path(__file__).resolve().parent.parent
        self.FINAL_DIR = self.ROOT / "reports" / "final"
        
        self._cache = {}
        
    def read_text(self, filepath):
        if filepath not in self._cache:
            try:
                self._cache[filepath] = Path(filepath).read_text()
            except FileNotFoundError:
                self._cache[filepath] = ""
        return self._cache[filepath]
        
    @property
    def anon(self): return self.read_text(self.FINAL_DIR / "grpo_agentic_llm_paper_anonymous.tex").lower()
    @property
    def anon_tex(self): return self.anon
    @property
    def capstone(self): return self.read_text(self.FINAL_DIR / "capstone_final_report.md").lower()
    @property
    def checklist(self): return self.read_text(self.FINAL_DIR / "SUBMISSION_CHECKLIST.md").lower()
    @property
    def ideas(self): return self.read_text(self.ROOT / "autoresearch.ideas.md").lower()
    @property
    def main_tex(self): return self.read_text(self.FINAL_DIR / "grpo_agentic_llm_paper.tex").lower()
    @property
    def md(self): return self.read_text(self.ROOT / "autoresearch.md").lower()
    @property
    def paper(self): return self.main_tex
    @property
    def readme(self): return self.read_text(self.FINAL_DIR / "README.md").lower()
    @property
    def export_script(self): return self.read_text(self.FINAL_DIR / "prepare_blind_review_package.py").lower()
    @property
    def script(self): return self.read_text(self.FINAL_DIR / "evaluate_gsm8k_test.py").lower()
    @property
    def submission(self): return self.read_text(self.FINAL_DIR / "SUBMISSION_README.md").lower()
    @property
    def supp(self): return self.read_text(self.FINAL_DIR / "supplementary_appendix.tex").lower()
    @property
    def text(self): return self.read_text(self.FINAL_DIR / "capstone_final_report.tex").lower()
    
    @property
    def files(self):
        return {
            "main_tex": self.main_tex,
            "anon_tex": self.anon_tex
        }

def run_audit(metric_name, get_issues_fn):
    ctx = AuditContext()
    issues = get_issues_fn(ctx)
    print(f"METRIC {metric_name}={len(issues)}")
    if issues:
        print("\\n".join(str(i) for i in issues))
