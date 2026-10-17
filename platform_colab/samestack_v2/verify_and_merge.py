"""Verify staged Colab same-stack v2 runs, then copy them into the canonical results directory.

Checks per staged run: full (non-smoke) config identical to the Modal runs apart from platform, 1,319 graded
items, completions present, and the execution environment recorded. As a hardware-drift control, the base
model's per-item greedy answers on Colab (L4) are compared with the Modal (A10G) base answers, which are
identical across all 27 Modal runs.

Usage: python3 platform_colab/samestack_v2/verify_and_merge.py [--merge]
Writes reports/project_finish_2026-10-11/validation/samestack_v2/colab_runs_check.json
"""

import gzip
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s_colab"
CANON = ROOT / "platform_hybrid/experiments/results/samestack_gsm8k_v2s"
OUT = ROOT / "reports/project_finish_2026-10-11/validation/samestack_v2/colab_runs_check.json"
CONFIG_KEYS = ("model", "n_steps", "k_epochs", "lr", "max_new", "n_gen", "n_eval")


def load(path):
    return json.loads(path.read_text().replace("NaN", "null"))


def main():
    ref = load(CANON / "grpo_g8_s42.json")
    ref_txt = json.loads(gzip.open(CANON / "grpo_g8_s42_texts.json.gz", "rt").read())
    report, ok_all = {"reference": "grpo_g8_s42 (Modal A10G)", "runs": {}}, True
    for path in sorted(STAGE.glob("*_s*.json")):
        r, name = load(path), path.stem
        txt_path = STAGE / f"{name}_texts.json.gz"
        txt = json.loads(gzip.open(txt_path, "rt").read()) if txt_path.exists() else None
        same_pre = sum(a == b for a, b in zip(r["pre_correct"], ref["pre_correct"]))
        same_text = sum(a == b for a, b in zip(txt["pre"], ref_txt["pre"])) if txt else None
        checks = {
            "not_smoke": r.get("smoke") is False,
            "config_matches_modal": all(r.get(k) == ref.get(k) for k in CONFIG_KEYS),
            "items_graded_1319": len(r["post_correct"]) == len(r["pre_correct"]) == 1319,
            "completions_present": txt is not None and len(txt["post"]) == 1319,
            "environment_recorded": "execution_environment" in r,
            "not_already_canonical": not (CANON / path.name).exists() or load(CANON / path.name) == r,
        }
        ok = all(checks.values())
        ok_all &= ok
        report["runs"][name] = {
            "checks": checks, "ok": ok, "pre_acc": r["heldout_pre_acc"], "post_acc": r["heldout_post_acc"],
            "base_items_same_verdict_as_modal": same_pre, "base_texts_identical_to_modal": same_text,
            "base_acc_modal": ref["heldout_pre_acc"], "gpu": r.get("execution_environment", {}).get("gpu"),
        }
        print(f"{'OK ' if ok else 'BAD'} {name}: post={r['heldout_post_acc']:.4f} pre={r['heldout_pre_acc']:.4f} "
              f"(Modal base {ref['heldout_pre_acc']:.4f}); base verdicts same as Modal on {same_pre}/1319, "
              f"identical base texts {same_text}/1319" + ("" if ok else f"  {checks}"))
    OUT.write_text(json.dumps(report, indent=1) + "\n")
    if "--merge" in sys.argv:
        if not ok_all:
            sys.exit("not merging: a staged run failed its checks")
        for path in sorted(STAGE.glob(f"*_s*")):
            shutil.copy2(path, CANON / path.name)
        print(f"merged {len(list(STAGE.glob('*_s*.json')))} runs into {CANON.relative_to(ROOT)}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
