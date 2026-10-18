"""Re-run the P5 Section 'Reference Verifier' demonstrations against the CURRENT code.

1. minreport.py verify --diagnostic on the two repository provenance records
   (campaign = multi-seed, sweep = single-seed).
2. sign.py: fresh Ed25519 identity in a temp dir, sign a copy of the campaign
   record, verify (expect PASS), forge git_sha, verify again (expect FAIL).

Outputs -> platform_hybrid/experiments/results/review_2026-10-18/P5_minreport/verifier_demo.json
"""
import json


import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROV = ROOT / "platform_hybrid/registry/provenance"
OUT = ROOT / "platform_hybrid/experiments/results/review_2026-10-18/P5_minreport"


def run(args):
    p = subprocess.run([sys.executable, *args], cwd=PROV, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("campaign", "sweep"):
            out = tmp / f"{name}.audit.json"
            code, text = run(["minreport.py", "verify", f"{name}.provenance.json", "--diagnostic", "--out", str(out)])
            rep = json.load(open(out))
            res[name] = {
                "exit": code,
                "accepted": rep["accepted"],
                "grade": rep["grade"],
                "score_pct": rep["score_pct"],
                "seed_check": [c for c in rep["checks"] if "seed" in c["msg"]],
                "n_pass": sum(c["level"] == "PASS" for c in rep["checks"]),
                "n_checks": len(rep["checks"]),
            }
        # sign.py's CLI `sign` refuses records that fail the evidence policy (both
        # repo records do), so exercise the signing primitive directly.
        sys.path.insert(0, str(PROV))
        import sign  # noqa: E402
        keys = tmp / "keys"
        sign.keygen(key_dir=keys)
        pub = keys / "provenance_ed25519.pub"
        record = json.load(open(PROV / "campaign.provenance.json"))
        env = sign.sign_record(record, keys)
        res["sign_cli_refuses_unready_record"] = run(
            ["sign.py", "sign", str(PROV / "campaign.provenance.json"), "--key-dir", str(keys)])[0] != 0
        res["verify_pristine"] = sign.verify_record(record, env, pub)
        forged = json.loads(json.dumps(record))
        forged["provenance"]["git_sha"] = "0" * 40
        assert forged != record
        res["verify_forged_git_sha"] = sign.verify_record(forged, env, pub)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "verifier_demo.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
