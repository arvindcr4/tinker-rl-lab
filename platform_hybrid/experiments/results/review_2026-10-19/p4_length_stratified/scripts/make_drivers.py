"""Derive length-stratified copies of the two P4 Modal drivers (DeepReview #245).

Identical training/eval code; only adds per-rollout length x correctness logging,
a per-call MAX_NEW, separate volume/W&B namespaces, no HF push, and a JSON dump
of every run. Usage: python make_drivers.py <repo_root>
"""

import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
SRC = ROOT / "platform_hybrid/experiments/modal"
OUT = pathlib.Path(__file__).parent

LENSTRAT = """
        _L = mask.sum(1).float().cpu().numpy()
        _c = np.asarray(rewards) > 0.5
        lenstrat = {
            "len_correct_mean": float(_L[_c].mean()) if _c.any() else None,
            "len_incorrect_mean": float(_L[~_c].mean()) if (~_c).any() else None,
            "n_correct": int(_c.sum()),
            "n_incorrect": int((~_c).sum()),
            "frac_at_cap": float((_L >= MAX_NEW).mean()),
            "frac_at_cap_correct": float((_L[_c] >= MAX_NEW).mean()) if _c.any() else None,
            "frac_at_cap_incorrect": float((_L[~_c] >= MAX_NEW).mean()) if (~_c).any() else None,
            "lengths": _L.astype(int).tolist(),
            "rewards": _c.astype(int).tolist(),
        }"""

ENTRY = '''@app.local_entrypoint()
def main(max_new: int = 0, out: str = ""):
    mn = max_new or MAX_NEW
    jobs = [(al, s, mn) for al in ALGOS for s in SEEDS]
    print(f"{len(jobs)} runs ({ALGOS} x {SEEDS}), MAX_NEW={mn}")
    results = [r for r in run_arm.starmap(jobs, return_exceptions=True) if isinstance(r, dict)]
    with open(out, "w") as f:
        json.dump({"max_new": mn, "model": MODEL, "runs": results}, f)
    print("saved", out, len(results), "runs")
'''

SPECS = {
    "modal_drgrpo_vs_grpo.py": ("drgrpo", "tinkerrl-drgrpo-vs-grpo", "step_log[-1]", 'if "HF_TOKEN" in os.environ:'),
    "modal_drgrpo_gsm8k_cot.py": ("drgrpo_gsm8k", "tinkerrl-drgrpo-gsm8k-cot", "metrics", 'if os.environ.get("HF_TOKEN"):'),
}


def sub(s, old, new):
    n = s.count(old)
    if n != 1:
        sys.exit(f"{old!r}: {n} matches")
    return s.replace(old, new)


for name, (voldir, proj, logged, hf) in SPECS.items():
    s = (SRC / name).read_text()
    s = sub(s, f'modal.App("{proj}")', f'modal.App("{proj}-lenstrat")')
    s = sub(
        s,
        "def run_arm(algo: str, seed: int) -> dict:\n",
        "def run_arm(algo: str, seed: int, max_new: int) -> dict:\n    MAX_NEW = max_new\n",
    )
    s = sub(s, "        comp_len = float(mask.sum(1).mean().item())", "        comp_len = float(mask.sum(1).mean().item())" + LENSTRAT)
    s = sub(s, '"mean_comp_len": comp_len,', '"mean_comp_len": comp_len,\n                **lenstrat,')
    s = sub(s, f"wandb.log({logged})", f"wandb.log({{k: v for k, v in {logged}.items() if not isinstance(v, list)}})")
    s = s.replace(f'project="{proj}"', 'project="tinkerrl-drgrpo-lenstrat"')
    s = s.replace(f'"WANDB_PROJECT": "{proj}"', '"WANDB_PROJECT": "tinkerrl-drgrpo-lenstrat"')
    s = sub(s, 'config={"algo": algo, "seed": seed, "model": MODEL}', 'config={"algo": algo, "seed": seed, "model": MODEL, "max_new": MAX_NEW}')
    s = sub(s, '        "model": MODEL,\n', '        "model": MODEL,\n        "max_new": MAX_NEW,\n')
    s = s.replace(f"{{RESULTS_DIR}}/{voldir}", f"{{RESULTS_DIR}}/{voldir}_lenstrat_cap{{MAX_NEW}}")
    s = sub(s, hf, "if False:  # review rerun: no HF push")
    s = s[: s.index("@app.local_entrypoint()")] + ENTRY
    (OUT / name.replace(".py", "_lenstrat.py")).write_text(s)
    print("wrote", name)
