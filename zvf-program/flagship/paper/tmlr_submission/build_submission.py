#!/usr/bin/env python3
"""Build the anonymous TMLR paper and deterministic review supplement."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


TMLR_DIR = Path(__file__).resolve().parent
PAPER_DIR = TMLR_DIR.parent
REPO_ROOT = PAPER_DIR.parents[2]
CANONICAL_TEX = PAPER_DIR / "main.tex"
CANONICAL_BIB = PAPER_DIR / "flagship.bib"
CANONICAL_BUNDLE = PAPER_DIR / "review_bundle.zip"
OUTPUT_ZIP = TMLR_DIR / "anonymous_supplement.zip"
OUTPUT_DIGEST = TMLR_DIR / "ANONYMOUS_SUPPLEMENT.sha256"
ZIP_TIME = (2026, 8, 2, 0, 0, 0)
EXECUTED_OBJECTIVE_SHA256 = (
    "980a56a1651299a5adbe7a0927c13b12d42d9d7e1a36205500a24d5eeba9b61b"
)
IDENTITY_MARKERS = (
    b"arvind",
    b"pes university",
    b"gmail.com",
    b"/users/",
    b"arvindcr4",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def make_tmlr_source() -> str:
    source = CANONICAL_TEX.read_text(encoding="utf-8")
    source = source.replace(
        "\\usepackage[preprint,nonatbib]{neurips_2026}\n"
        "\\usepackage[numbers,sort&compress]{natbib}",
        "\\usepackage{tmlr}",
    )
    source = source.replace(
        "\\author{Arvind C R\\\\\nPES University\\\\\n"
        "\\texttt{arvindcr4@gmail.com}}",
        "\\author{Anonymous Authors}",
    )
    source = source.replace("\\path{verify_claims.py}", "\\path{verify_anonymous_claims.py}")
    source = source.replace("\\path{CLAIM_AUDIT.md}", "\\path{ANONYMOUS_CLAIM_LEDGER.md}")
    source = source.replace(
        "a content-addressed review bundle",
        "an anonymous review supplement",
    )
    source = source.replace("review bundle contains", "anonymous supplement contains")
    source = source.replace("are in the review bundle", "are in the anonymous supplement")
    start = source.index("\\section*{Reproducibility Statement}")
    end = source.index("\\bibliographystyle{plainnat}", start)
    statement = r"""\section*{Reproducibility Statement}

The anonymous supplement contains the complete 600-record numerical projection
used in the r4-2 tables, both S1 receipt projections, the unchanged S1 source and
tests, and the exact executed objective snapshot.  Remote account identifiers,
author metadata, and machine-local paths are omitted for double-blind review;
SHA-256 digests of the unredacted source objects remain as provenance anchors.
After extraction, run
\begin{verbatim}
python3 verify_anonymous_claims.py
\end{verbatim}
to verify the internal manifest, formula checks, S1 invariants, all six campaign
records, and the failed 69/100 mechanism gate.  The verifier performs no network
access and does not regenerate training, gradients, predictions, or private
corpora.  Exact coverage is documented in
\path{ANONYMOUS_CLAIM_LEDGER.md} and
\path{ANONYMITY_AND_PROVENANCE.md}.

"""
    source = source[:start] + statement + source[end:]
    source = source.replace("\\bibliographystyle{plainnat}", "\\bibliographystyle{tmlr}")
    assert "neurips_2026" not in source
    assert "\\usepackage{tmlr}" in source
    scan_identity({"main.tex": source.encode("utf-8")})
    return source


def write_paper_sources() -> None:
    (TMLR_DIR / "main.tex").write_text(make_tmlr_source(), encoding="utf-8")
    (TMLR_DIR / "flagship.bib").write_bytes(CANONICAL_BIB.read_bytes())


def anonymous_source_path(value: str) -> str:
    if "site-packages/" in value:
        return "<isolated-environment>/site-packages/" + value.split("site-packages/", 1)[1]
    if value.startswith("/"):
        return "<isolated-environment>/" + value.rsplit("/", 1)[-1]
    return value


def project_s1_receipt(stack: str) -> dict[str, Any]:
    path = REPO_ROOT / f"zvf-program/flagship/s1/results/{stack}_receipt.json"
    projected = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    projected["original_receipt_sha256"] = sha256(path)
    for key, value in list(projected["provenance"].items()):
        if key.endswith("_source") and isinstance(value, str):
            projected["provenance"][key] = anonymous_source_path(value)
    return projected


def project_unit(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    full = record["full_record"]
    return {
        "schema_version": "anonymous-r4-2-unit-v1",
        "status": record["status"],
        "unit": record["unit"],
        "unit_fingerprint": record["unit_fingerprint"],
        "corpus_fingerprint": record["corpus_fingerprint"],
        "original_acceptance_sha256": sha256(path),
        "runtime_versions": full["runtime_versions"],
        "gradient_receipts": full["manifest"]["gradient_receipts"],
        "evaluations": full["evaluations"],
        "token_flop_ledger": full["token_flop_ledger"],
    }


def project_design() -> dict[str, Any]:
    campaign = REPO_ROOT / "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2"
    prereg_path = REPO_ROOT / "zvf-program/flagship/pilot_preregistration.json"
    manifest_path = campaign / "launch_manifest.json"
    state_path = campaign / "supervisor_state.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    job_statuses = {key: value["status"] for key, value in sorted(state["jobs"].items())}
    assert Counter(job_statuses.values()) == Counter({
        "accepted": 10,
        "descoped_contract_infeasible": 14,
        "failed_infrastructure": 2,
        "failed_validation": 1,
        "pending_quota_reset": 4,
    })
    return {
        "schema_version": "anonymous-r4-2-design-v1",
        "job_count": manifest["job_count"],
        "job_statuses": job_statuses,
        "model_id": prereg["runtime"]["model"]["id"],
        "max_completion_length": prereg["runtime"]["execution_contract"]["max_completion_length"],
        "filtered_positive_control_cv_threshold": 0.35,
        "observed_filtered_positive_control_cv": 0.0,
        "mechanism_gate": {
            "required_equivalent_steps": 95,
            "total_steps": 100,
            "nonzero_cosine_minimum": 0.999,
            "nonzero_relative_l2_maximum": 0.01,
            "joint_zero_counts_as_equivalent": True,
        },
        "executed_objective_sha256": EXECUTED_OBJECTIVE_SHA256,
        "original_preregistration_sha256": sha256(prereg_path),
        "original_launch_manifest_sha256": sha256(manifest_path),
        "original_supervisor_state_sha256": sha256(state_path),
    }


def collect_payloads() -> dict[str, bytes]:
    required = (
        "main.tex", "main.pdf", "flagship.bib", "tmlr.sty", "tmlr.bst",
        "README.md", "ANONYMITY_AND_PROVENANCE.md",
        "ANONYMOUS_CLAIM_LEDGER.md", "verify_anonymous_claims.py",
    )
    payloads: dict[str, bytes] = {}
    for name in required:
        path = TMLR_DIR / name
        if not path.is_file():
            raise SystemExit(f"missing submission file: {path}")
        target = f"paper/{name}" if name in {"main.tex", "main.pdf", "flagship.bib", "tmlr.sty", "tmlr.bst"} else name
        payloads[target] = path.read_bytes()

    s1_root = REPO_ROOT / "zvf-program/flagship/s1"
    for path in sorted(s1_root.rglob("*")):
        if not path.is_file() or "results" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name == ".DS_Store":
            continue
        relative = path.relative_to(s1_root).as_posix()
        payloads[f"evidence/s1/source/{relative}"] = path.read_bytes()
    for stack in ("trl", "verl"):
        payloads[f"evidence/s1/{stack}_receipt.anonymous.json"] = json_bytes(project_s1_receipt(stack))

    campaign_acceptance = (
        REPO_ROOT / "zvf-program/flagship/pilot/launch-v2-corpus-resume-r4-2/acceptance"
    )
    acceptance_digests: dict[str, str] = {}
    for path in sorted(campaign_acceptance.glob("fpilot__*.json")):
        projected = project_unit(path)
        unit_id = projected["unit"]["id"]
        acceptance_digests[unit_id] = projected["original_acceptance_sha256"]
        payloads[f"evidence/r4_2/units/{unit_id}.json"] = json_bytes(projected)
    if len(acceptance_digests) != 6:
        raise SystemExit(f"expected six scientific acceptance records, found {len(acceptance_digests)}")

    design = project_design()
    payloads["evidence/r4_2/design_and_disposition.json"] = json_bytes(design)
    objective = REPO_ROOT / "zvf-program/flagship/pilot/provenance/r4-2-objective.py"
    if sha256(objective) != EXECUTED_OBJECTIVE_SHA256:
        raise SystemExit("executed objective snapshot hash mismatch")
    payloads["evidence/r4_2/source/r4-2-objective.py"] = objective.read_bytes()

    anchors = {
        "schema_version": "anonymous-provenance-anchors-v1",
        "full_review_bundle_sha256": sha256(CANONICAL_BUNDLE),
        "canonical_main_source_sha256": sha256(CANONICAL_TEX),
        "executed_objective_sha256": EXECUTED_OBJECTIVE_SHA256,
        "acceptance_digests": acceptance_digests,
        "s1_receipt_digests": {
            stack: sha256(REPO_ROOT / f"zvf-program/flagship/s1/results/{stack}_receipt.json")
            for stack in ("trl", "verl")
        },
    }
    payloads["evidence/provenance_anchors.json"] = json_bytes(anchors)
    scan_identity(payloads)
    return payloads


def scan_identity(payloads: dict[str, bytes]) -> None:
    failures = []
    for name, payload in payloads.items():
        lowered = payload.lower()
        for marker in IDENTITY_MARKERS:
            if marker in lowered:
                failures.append(f"{name}: {marker.decode('ascii')}")
    if failures:
        raise SystemExit("identity scan failed:\n" + "\n".join(failures))


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def build_supplement() -> None:
    payloads = collect_payloads()
    manifest = "".join(
        f"{sha256_bytes(payloads[name])}  {name}\n" for name in sorted(payloads)
    ).encode("utf-8")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(payloads):
            archive.writestr(zip_info(name), payloads[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        archive.writestr(zip_info("MANIFEST.sha256"), manifest, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = sha256(OUTPUT_ZIP)
    OUTPUT_DIGEST.write_text(f"{digest}  {OUTPUT_ZIP.name}\n", encoding="utf-8")
    print(f"{OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size} bytes)")
    print(f"sha256 {digest}")
    print(f"payload files {len(payloads)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-only", action="store_true")
    args = parser.parse_args()
    write_paper_sources()
    print(f"wrote anonymous TMLR source: {TMLR_DIR / 'main.tex'}")
    if not args.paper_only:
        build_supplement()


if __name__ == "__main__":
    main()
