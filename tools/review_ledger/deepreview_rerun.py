"""Re-run Google Deep Review on the papers changed by the 2026-10-18 portfolio pass.

  python tools/review_ledger/deepreview_rerun.py build   [paper ...]   # rebuild PDFs from current sources
  python tools/review_ledger/deepreview_rerun.py submit  [paper ...]   # upload + start reviews (no wait)
  python tools/review_ledger/deepreview_rerun.py poll                  # fetch finished reviews

PDFs, the job checkpoint (batch.json) and review outputs go to reports/deepreview/rerun_2026-10-18/.
Pass --byok to `submit` to bill the reviews to $GEMINI_API_KEY.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/deepreview/rerun_2026-10-18"
BATCH = OUT / "batch.json"
CLIENT = ROOT / "tools/deepreview.py"
OUTS = "outputs/PES_Conference_Submissions_2026-09-25"

# Papers whose sources this pass changed. ICICC, the P1 note, both P6 papers and P8 are
# owned by the Amp session and their requested fixes are not applied yet, so they are excluded.
SOURCES = {
    "P1_scaling": "platform_hybrid/paper/paper_P1_scaling.tex",
    "P2_zvf": "platform_hybrid/paper/paper_P2_zvf.tex",
    "P2_zvf_falsification_note": "platform_hybrid/paper/paper_P2_zvf_falsification_note.tex",
    "P3_group_size": "platform_hybrid/paper/paper_P3_group_size.tex",
    "P4_length_bias": "platform_hybrid/paper/paper_P4_length_bias.tex",
    "P5_minreport": "platform_hybrid/paper/paper_P5_minreport.tex",
    "P7_zvf_controller": "platform_hybrid/paper/paper_P7_zvf_controller.tex",
    "P9_dnb": "platform_hybrid/paper/neurips_2026_variants/paper_P9_dnb.tex",
    "P10_zvf_theory": "zvf-program/theory/paper_P10_zvf_theory.tex",
    "P11_reproducibility_audit": "zvf-program/audit/paper_P11_reproducibility_audit.tex",
    "P12_signal_starvation": "platform_hybrid/paper/unified_signal_starvation/paper_P12_signal_starvation.tex",
    "Flagship_Main": "zvf-program/flagship/paper/main.tex",
    "Flagship_TMLR": "zvf-program/flagship/paper/tmlr_submission/main.tex",
    "Next_Submission": "zvf-program/next-submission/paper/main.tex",
    "Spectral_Entropy": "zvf-program/flagship/paper/spectral_entropy_paper.tex",
    "ACL_ARR_Oct2026_camera": f"{OUTS}/ACL_ARR_Oct2026/paper_camera.tex",
    "RAIT2027_camera": f"{OUTS}/RAIT2027/paper_camera.tex",
    "PES_IEEE_Attribution": "papers/pes_ieee_attribution/ArvindCR_GRPO_Attribution_IEEE.tex",
    "C1_Research_Addendum": "reports/public_revision_2026-10-03/addendum/C1_Research_Addendum_2026-10-03_PUBLIC.tex",
    "Thesis_2026_10_11_ESA_Corrected": "reports/project_finish_2026-10-11/thesis/build_thesis.py",
    "Sem3_Group6_Original_Report": "platform_tinker/reports/combined/base_final.tex",
    "Sem3_NeurIPS_Main_Track": "platform_hybrid/paper/paper.tex",
    "Sem4_P8_Fraud": "platform_hybrid/paper/archive/absorbed/P08_fraud/paper_P8_fraud.tex",
}

PROMPT = (
    "This manuscript was revised in response to an earlier Deep Review. Unsourced tables and "
    "claims were removed rather than patched, and statistics were recomputed with seeds as the "
    "unit. Review the current text on its merits; flag any remaining claim that the reported "
    "evidence does not support."
)


def load():
    return json.loads(BATCH.read_text()) if BATCH.exists() else {}


def save(state):
    BATCH.write_text(json.dumps(state, indent=2) + "\n")


def build(paper):
    src = ROOT / SOURCES[paper]
    dest = OUT / "pdfs" / f"{paper}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log = OUT / "logs" / f"{paper}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == ".py":
        r = subprocess.run(
            [sys.executable, src.name], cwd=src.parent, capture_output=True, text=True
        )
        m = re.search(r"OK -> (\S+)", r.stdout)
        built = Path(m.group(1)) if m else None
    else:
        tmp = OUT / "build" / paper
        tmp.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["tectonic", "-X", "compile", "--outdir", str(tmp), src.name],
            cwd=src.parent,
            capture_output=True,
            text=True,
        )
        built = tmp / f"{src.stem}.pdf"
    log.write_text(r.stdout + r.stderr)
    undefined = len(re.findall(r"undefined", r.stdout + r.stderr))
    if r.returncode != 0 or not built or not built.exists():
        return {"build": "FAILED", "log": str(log.relative_to(ROOT))}
    shutil.copyfile(built, dest)
    return {
        "build": "ok",
        "undefined_warnings": undefined,
        "path": str(dest.relative_to(ROOT)),
        "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
    }


def submit(paper, entry, byok):
    cmd = [
        sys.executable,
        str(CLIENT),
        "review",
        entry["path"],
        "--no-wait",
        "--display-name",
        f"{paper}_rerun_2026-10-18",
        "--prompt",
        PROMPT,
    ]
    if byok:
        cmd.append("--byok")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        entry.update(state="SUBMIT_FAILED", error=(r.stderr or r.stdout)[-800:])
    else:
        entry.update(state="RUNNING", review=r.stdout.strip().splitlines()[-1])


def poll(paper, entry):
    out = OUT / paper
    r = subprocess.run(
        [sys.executable, str(CLIENT), "get", entry["review"], "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    reviews = list(out.rglob("review.json")) if out.exists() else []
    if reviews:
        state = json.loads(reviews[0].read_text()).get("state", "UNKNOWN")
        entry.update(state=state, output_dir=str(reviews[0].parent.relative_to(ROOT)))
    elif r.returncode != 0:
        entry["poll_error"] = (r.stderr or r.stdout)[-400:]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cmd, papers = args[0], args[1:] or list(SOURCES)
    state = load()
    for paper in papers:
        entry = state.setdefault(paper, {})
        if cmd == "build":
            entry.update(build(paper))
        elif (
            cmd == "submit"
            and entry.get("build") == "ok"
            and entry.get("state") not in ("RUNNING", "SUCCEEDED")
        ):
            submit(paper, entry, "--byok" in sys.argv)
        elif cmd == "poll" and entry.get("state") == "RUNNING":
            poll(paper, entry)
        save(state)
        print(
            paper,
            {
                k: v
                for k, v in entry.items()
                if k in ("build", "state", "error", "undefined_warnings")
            },
        )


if __name__ == "__main__":
    main()
