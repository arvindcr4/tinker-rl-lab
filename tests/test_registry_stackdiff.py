"""Unit tests for the GRPO-Registry badge + stackdiff (platform_hybrid/registry/query.py)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "platform_hybrid" / "registry"
_spec = importlib.util.spec_from_file_location("registry_query", REG / "query.py")
q = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q)


def cell(value, status="code", source="test.py:L1"):
    return {"value": value, "status": status, "source": source}


def stack(**rc_overrides):
    """A fully specified synthetic stack: every FIELDS entry known."""
    rc = {f: cell(f"v-{f}") for f in q.FIELDS}
    rc["seeds"] = cell([1, 2, 3])
    rc["decoding.temperature"] = cell(0.8)
    for k, v in rc_overrides.items():
        rc[k.replace("__", ".")] = v
    return {
        "record_type": "stack", "id": "synthetic", "label_claimed": "grpo",
        "framework": {"name": "x", "version": None, "openness": "open"},
        "model": None, "task": None, "seeds": None,
        "provenance": {"source_artifacts": []},
        "variant_deltas_applied": [],
        "min_report": {it: {"a": None} for it in q.ITEMS},
        "run_config": rc,
    }


def entry(eid):
    return json.loads((REG / "entries" / f"{eid}.json").read_text())


# ------------------------------------------------------------------- badge
def test_badge_counts_reported_leaves():
    rec = stack()
    definitions = json.loads((REG / "schema.json").read_text())["$defs"]["min_report"]["properties"]
    expected = {}
    for item, definition in definitions.items():
        keys = sorted(set(definition["properties"]) - {"source", "description"})
        rec["min_report"][item] = {key: (1 if i % 2 == 0 else None) for i, key in enumerate(keys)}
        expected[item] = sum(value is not None for value in rec["min_report"][item].values()) / len(keys)
    total, scores = q.badge(rec)
    assert scores == expected and total == round(100 * sum(expected.values()) / 7)


def test_badge_false_and_zero_count_as_reported():
    assert q.item_score({"performed": False, "parser_robustness_probe": False}, "decontamination") == 1
    assert q.item_score({"reference_policy":False,"kl_beta":0.0,"kl_estimator":"none"}, "reference_kl") == 1


def test_spec_coverage_separates_unknown():
    rec = stack()
    assert q.spec_coverage(rec) == 1.0
    rec["run_config"]["lora.rank"] = cell(None, "unknown", "not logged")
    assert q.spec_coverage(rec) == pytest.approx(1 - 1 / len(q.FIELDS))


# --------------------------------------------------------------- stackdiff
def test_identical_entries_are_r0_full_coverage():
    r = q.stackdiff(stack(), stack())
    assert r["level"] == 0
    assert r["differs"] == []
    assert r["unknown"] == []
    assert r["coverage"] == 1.0
    assert not r["underspecified"]


def test_checkpoint_only_diff_is_r5():
    r = q.stackdiff(stack(), stack(checkpoint__id=cell("Qwen/Qwen3-8B")))
    assert r["level"] == 5
    assert [d[0] for d in r["differs"]] == ["checkpoint.id"]


def test_reward_only_diff_is_r5():
    r = q.stackdiff(stack(), stack(reward__id=cell("boxed-only")))
    assert r["level"] == 5
    assert [d[0] for d in r["differs"]] == ["reward.id"]


def test_reward_hash_only_diff_is_r2():
    r = q.stackdiff(stack(), stack(reward__fn_sha256=cell("ff" * 32)))
    assert r["level"] == 2


@pytest.mark.parametrize(("field", "expected"), [
    ("loss.clip", 4), ("kl.beta", 4), ("decoding.max_tokens", 3), ("lora.rank", 3),
    ("trainer.name", 2), ("seeds", 1),
])
def test_severity_ladder(field, expected):
    r = q.stackdiff(stack(), stack(**{field.replace(".", "__"): cell("other")}))
    assert r["level"] == expected == q.FIELDS[field]


def test_missing_field_is_unknown_not_a_difference():
    b = stack(checkpoint__id=cell(None, "unknown", "not logged"))
    r = q.stackdiff(stack(), b)
    assert r["level"] == 0
    assert r["differs"] == []
    assert ("checkpoint.id", 5, "b") in r["unknown"]
    assert r["underspecified"]
    assert r["hidden_max"] == 5
    assert r["coverage"] == pytest.approx(1 - 1 / len(q.FIELDS))


def test_missing_vs_differing_fields_are_separated():
    b = stack(lora__rank=cell(None, "unknown", "managed"),
              decoding__temperature=cell(1.0))
    r = q.stackdiff(stack(), b)
    assert [d[0] for d in r["differs"]] == ["decoding.temperature"]
    assert [u[0] for u in r["unknown"]] == ["lora.rank"]
    assert r["level"] == 3


def test_revision_unknown_alone_flags_unverified_identity():
    unk = cell(None, "unknown", "unpinned")
    r = q.stackdiff(stack(checkpoint__revision=unk), stack(checkpoint__revision=unk))
    assert r["level"] == 0 and r["underspecified"] and not r["comparable"]
    assert not r["matching_requirements"]["satisfied"]
    assert ("checkpoint.revision", 5, "both") in r["unknown"]


def test_float_equality_is_tolerant_and_bool_is_not_number():
    r = q.stackdiff(stack(optimizer__lr=cell(1e-5)), stack(optimizer__lr=cell(0.00001)))
    assert r["level"] == 0
    r = q.stackdiff(stack(kl__reference_policy=cell(False)), stack(kl__reference_policy=cell(0.0)))
    assert r["level"] == 4


def test_delta_rows_declared_both_sides_compare_at_r4():
    a, b = stack(), stack()
    a["variant_deltas_applied"] = [{"delta_id": "delta_dapo", "component": "clip_higher", "status": "implemented"}]
    b["variant_deltas_applied"] = [{"delta_id": "delta_dapo", "component": "clip_higher", "status": "absent"}]
    assert q.stackdiff(a, b)["level"] == 4


def test_delta_row_unknown_or_one_sided_is_not_a_difference():
    a, b = stack(), stack()
    a["variant_deltas_applied"] = [{"delta_id": "delta_dapo", "component": "kl_removed", "status": "implemented"}]
    r = q.stackdiff(a, b)
    assert r["level"] == 0 and r["undeclared"] == [("delta.delta_dapo:kl_removed", "b")]
    assert r["coverage"] < 1.0 and not r["comparable"]
    b["variant_deltas_applied"] = [{"delta_id": "delta_dapo", "component": "kl_removed", "status": "unknown"}]
    r = q.stackdiff(a, b)
    assert r["level"] == 0
    assert ("delta.delta_dapo:kl_removed", 4, "b") in r["unknown"]


def test_legacy_entry_without_run_config_maps_null_to_unknown():
    rec = entry("zvf130_grpo")
    assert "run_config" not in rec
    v = q.field_view(rec)
    assert v["checkpoint.id"]["status"] == "registry"
    assert v["reward.id"]["status"] == "unknown"


# ------------------------------------------------- real exhibit pairs
def test_exhibit_a_pair_reports_checkpoint_reward_template():
    r = q.stackdiff(entry("tinker_grpo_qwen3-8b_gsm8k"), entry("trl_grpo_qwen3-8b_gsm8k"))
    diffs = {d[0]: (d[2], d[3]) for d in r["differs"]}
    assert r["level"] == 5
    assert diffs["checkpoint.id"] == ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B")
    assert diffs["checkpoint.variant"] == ("base", "instruct")
    for f in ("reward.id", "prompt.template_sha256", "prompt.chat_template",
              "batch.prompts_per_step", "data.subset", "optimizer.betas"):
        assert f in diffs, f
    # TRL loss/KL/decoding follow unpinned library defaults -> unknown, not "differs"
    unknown = {u[0] for u in r["unknown"]}
    assert {"kl.beta", "decoding.temperature", "loss.clip"} <= unknown
    assert "kl.beta" not in diffs


def test_dapo_pair_reports_task_model_and_absent_dapo_components():
    r = q.stackdiff(entry("colab-open_dapo_e3"), entry("tinker_dapo_qwen3.5-4b_gsm8k"))
    diffs = {d[0]: (d[2], d[3]) for d in r["differs"]}
    assert r["level"] == 5
    assert r["same_label"]
    for f in ("checkpoint.id", "data.dataset", "eval.metric", "batch.samples_per_prompt",
              "train.steps", "reward.id"):
        assert f in diffs, f
    assert diffs["loss.clip"] == ("ratio[0.8,1.28]", "advantage[-2,3]")
    assert diffs["delta.delta_dapo:clip_higher"] == ("implemented", "absent")
    assert r["coverage"] > 0.7 and not r["comparable"]
    assert r["underspecified"]
    assert not r["matching_requirements"]["satisfied"]


def test_tinker_dapo_entry_is_corrected():
    e = entry("tinker_dapo_qwen3.5-4b_gsm8k")
    lf = e["min_report"]["loss_form"]
    assert lf["clip_eps_low"] is None
    assert lf["clip_eps_high"] is None
    assert lf["sampling_dynamic_filter"] is False
    assert lf["reward_shaping_type"] == "none"
    assert e["run_config"]["loss.clip"]["value"] == "advantage[-2,3]"
    assert "cell_runner.py@b0ac85bfb:L" in e["run_config"]["loss.clip"]["source"]
    statuses = {d["component"]: d["status"] for d in e["variant_deltas_applied"]}
    assert statuses["clip_higher"] == statuses["dynamic_sampling"] == "absent"


def test_all_pairs_distribution_is_not_degenerate():
    res = q.all_pairs()
    n = len(q.load("stack"))
    assert len(res) == n * (n - 1) // 2
    levels = {r["level"] for r in res.values()}
    assert len(levels) >= 3  # the old ladder put 366/378 pairs at R5


def test_generated_run_config_cells_are_wellformed():
    for p in (REG / "entries").glob("*.json"):
        rc = json.loads(p.read_text()).get("run_config")
        for k, c in (rc or {}).items():
            assert k in q.FIELDS, (p.name, k)
            assert c["status"] in q.KNOWN_STATUSES + ("unknown",)
            assert c["source"], (p.name, k)
            if c["status"] == "unknown":
                assert c["value"] is None


# ---------------------------------------------------------- CLI / generator
def test_cli_runs_from_repo_root():
    out = subprocess.run([sys.executable, "platform_hybrid/registry/query.py", "stackdiff",
                          "colab-open_dapo_e3", "tinker_dapo_qwen3.5-4b_gsm8k"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    assert "verdict: R5" in out
    assert "coverage:" in out


@pytest.mark.skipif(shutil.which("git") is None, reason="generator reads pinned git blobs")
def test_historical_entries_require_explicit_versioned_migration():
    rc = subprocess.run([sys.executable, str(REG / "generate_entries.py"), "--check"],
                        cwd=REPO, capture_output=True, text=True)
    # Historical artifacts are intentionally immutable during review fixes.
    # Freshness must expose the old generator version rather than silently bless it.
    assert rc.returncode == 1 and "stale:" in rc.stdout, rc.stdout + rc.stderr


def test_generated_entries_are_stable_after_regeneration(tmp_path, monkeypatch):
    # Historical evidence is not rewritten as part of a code correction. Test
    # idempotence and the real --check gate on a disposable reviewed projection.
    sys.path.insert(0, str(REG))
    import generate_entries as generator
    specs = {**generator.fc_family(), **generator.h2h_family(), **generator.e3_family()}
    for eid, spec in specs.items():
        new = generator.build(eid, spec, entry(eid))
        assert generator.build(eid, spec, new) == new
        (tmp_path / f"{eid}.json").write_text(json.dumps(new))
    monkeypatch.setattr(generator, "ENTRIES", tmp_path)
    assert generator.main(["--check"]) == 0
    broken = tmp_path / f"{next(iter(specs))}.json"
    rec = json.loads(broken.read_text()); rec["notes"] = "stale"
    broken.write_text(json.dumps(rec))
    assert generator.main(["--check"]) == 1


def test_identifier_annotations_and_aliases_do_not_count_as_differences():
    a = stack(checkpoint__id=cell("Qwen/Qwen3-8B-base (shared harness)"), data__dataset=cell("gsm8k"))
    b = stack(checkpoint__id=cell("Qwen/Qwen3-8B-Base"), data__dataset=cell("openai/gsm8k:main"))
    assert q.stackdiff(a, b)["level"] == 0


def test_exit_code_is_zero_by_default_and_level_with_flag():
    args = [sys.executable, "platform_hybrid/registry/query.py", "stackdiff",
            "tinker_grpo_qwen3-8b_gsm8k", "trl_grpo_qwen3-8b_gsm8k"]
    assert subprocess.run(args, cwd=REPO, capture_output=True).returncode == 0
    assert subprocess.run(args + ["--exit-level"], cwd=REPO, capture_output=True).returncode == 5
