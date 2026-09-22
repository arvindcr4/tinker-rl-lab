from __future__ import annotations

import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

from .e9_mle_bench_streaming import (
    LaunchGateError,
    build_run_receipt,
    build_streaming_plan,
    extract_python_code,
    identify_known_deterministic_repair,
    preapply_replay_runtime_repair,
    repair_known_submission_alignment,
    repair_known_runtime_error,
    validate_bridge_gate,
    validate_replay_provenance,
    validate_submission_regrade_provenance,
)


class E9StreamingPlanTests(unittest.TestCase):
    def freesound_replay_artifacts(self) -> tuple[str, dict[str, object]]:
        repo_root = Path(__file__).resolve().parents[2]
        run_root = (
            repo_root / "outputs/e9_mle_bench/modal_streaming/"
            "freesound-audio-tagging-2019-41fd84bf84d5"
        )
        solution = (run_root / "solution.py").read_text(encoding="utf-8").rstrip()
        receipt = json.loads((run_root / "receipt.json").read_text(encoding="utf-8"))
        return solution, receipt

    def smartphone_replay_artifacts(self) -> tuple[str, dict[str, object]]:
        repo_root = Path(__file__).resolve().parents[2]
        run_root = (
            repo_root / "outputs/e9_mle_bench/modal_streaming/"
            "smartphone-decimeter-2022-3de801aee652"
        )
        solution = (run_root / "solution.py").read_text(encoding="utf-8").rstrip()
        receipt = json.loads((run_root / "receipt.json").read_text(encoding="utf-8"))
        return solution, receipt

    def replay_receipt(self) -> tuple[str, dict[str, object]]:
        solution_sha256 = hashlib.sha256(b"import pathlib\n").hexdigest()
        receipt: dict[str, object] = {
            "competition_id": "spooky-author-identification",
            "run_id": "spooky-author-identification-source",
            "hf_commit": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8",
            "wandb_url": "https://wandb.ai/team/project/runs/source",
            "artifacts": {"solution_sha256": solution_sha256},
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return solution_sha256, receipt

    def test_validates_archived_solution_replay_without_live_bridge(self) -> None:
        solution_sha256, source_receipt = self.replay_receipt()

        provenance = validate_replay_provenance(
            competition_id="spooky-author-identification",
            solution_sha256=solution_sha256,
            source_receipt=source_receipt,
        )

        self.assertEqual(provenance["status"], "ARCHIVED_SAMPLING_RECEIPT_REPLAY")
        self.assertEqual(provenance["source_solution_sha256"], solution_sha256)

    def test_replay_rejects_solution_hash_drift(self) -> None:
        _, source_receipt = self.replay_receipt()

        with self.assertRaisesRegex(LaunchGateError, "solution hash"):
            validate_replay_provenance(
                competition_id="spooky-author-identification",
                solution_sha256="0" * 64,
                source_receipt=source_receipt,
            )

    def test_replay_rejects_tampered_source_receipt(self) -> None:
        solution_sha256, source_receipt = self.replay_receipt()
        source_receipt["run_id"] = "tampered"

        with self.assertRaisesRegex(LaunchGateError, "receipt hash"):
            validate_replay_provenance(
                competition_id="spooky-author-identification",
                solution_sha256=solution_sha256,
                source_receipt=source_receipt,
            )

    def test_validates_saved_submission_regrade_provenance(self) -> None:
        submission_sha256 = hashlib.sha256(b"id,after\n0_0,x\n").hexdigest()
        receipt: dict[str, object] = {
            "competition_id": "text-normalization-challenge-russian-language",
            "run_id": "russian-source",
            "hf_commit": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8",
            "wandb_url": "https://wandb.ai/team/project/runs/source",
            "artifacts": {"submission_sha256": submission_sha256},
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        provenance = validate_submission_regrade_provenance(
            competition_id="text-normalization-challenge-russian-language",
            submission_sha256=submission_sha256,
            source_receipt=receipt,
        )

        self.assertEqual(provenance["status"], "SAVED_SUBMISSION_NATIVE_REGRADE")
        self.assertEqual(provenance["source_submission_sha256"], submission_sha256)

    def test_regrade_rejects_submission_hash_drift(self) -> None:
        submission_sha256 = hashlib.sha256(b"id,after\n0_0,x\n").hexdigest()
        receipt: dict[str, object] = {
            "competition_id": "text-normalization-challenge-russian-language",
            "run_id": "russian-source",
            "hf_commit": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8",
            "wandb_url": "https://wandb.ai/team/project/runs/source",
            "artifacts": {"submission_sha256": submission_sha256},
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        with self.assertRaisesRegex(LaunchGateError, "submission hash"):
            validate_submission_regrade_provenance(
                competition_id="text-normalization-challenge-russian-language",
                submission_sha256="0" * 64,
                source_receipt=receipt,
            )

    def test_smallest_pending_competition_runs_first_with_single_concurrency(self) -> None:
        plan = build_streaming_plan(
            competition_sizes_bytes={
                "large": 9_000,
                "already-done": 500,
                "small": 1_000,
            },
            completed={"already-done"},
            maximum_incremental_usd=50.0,
        )

        self.assertEqual(
            [item["competition_id"] for item in plan["queue"]],
            ["small", "large"],
        )
        self.assertEqual(plan["maximum_concurrency"], 1)
        self.assertEqual(plan["maximum_incremental_usd"], 50.0)

    def test_user_cap_above_authorized_fifty_dollars_fails_closed(self) -> None:
        with self.assertRaisesRegex(LaunchGateError, "authorized cap"):
            build_streaming_plan(
                competition_sizes_bytes={"small": 1_000},
                completed=set(),
                maximum_incremental_usd=50.01,
            )

    def test_bridge_gate_requires_checkpoint_wandb_and_remaining_budget(self) -> None:
        remaining = validate_bridge_gate(
            {
                "status": "READY",
                "model": "pavlov-qwen36-tinker",
                "hf_commit": "64444133c55d88c3f1bf0df8a2f5d7ac646125c8",
                "wandb_url": "https://wandb.ai/team/project/runs/abc",
                "budget": {
                    "maximum_usd": "55.91445263",
                    "charged_usd": 5.5,
                    "reserved_usd": 0.25,
                },
            },
            authorized_total_usd=55.91445263,
        )

        self.assertEqual(remaining, 50.16445263)

    def test_pilot_receipt_never_claims_full_suite_score(self) -> None:
        receipt = build_run_receipt(
            competition_id="spooky-author-identification",
            native_grade={"score": 0.42, "above_median": True},
            bridge_health={"hf_commit": "commit", "wandb_url": "https://wandb.ai/run"},
        )

        self.assertEqual(receipt["competition_score"], 0.42)
        self.assertIsNone(receipt["score"])
        self.assertFalse(receipt["is_full_suite_score"])

    def test_invalid_native_grade_is_preserved_without_claiming_a_score(self) -> None:
        receipt = build_run_receipt(
            competition_id="detecting-insults-in-social-commentary",
            native_grade={
                "score": None,
                "valid_submission": False,
                "grading_error": "missing columns",
            },
            bridge_health={"hf_commit": "commit", "wandb_url": "https://wandb.ai/run"},
        )

        self.assertEqual(receipt["status"], "NATIVE_SINGLE_COMPETITION_INVALID")
        self.assertIsNone(receipt["competition_score"])
        self.assertIsNone(receipt["score"])
        self.assertIn("no competition", receipt["claim_boundary"])

    def test_agent_execution_failure_is_distinct_from_native_rejection(self) -> None:
        receipt = build_run_receipt(
            competition_id="random-acts-of-pizza",
            native_grade={
                "score": None,
                "valid_submission": False,
                "agent_execution_failed": True,
            },
            bridge_health={"hf_commit": "commit", "wandb_url": "https://wandb.ai/run"},
        )

        self.assertEqual(receipt["status"], "AGENT_EXECUTION_FAILED")
        self.assertIsNone(receipt["competition_score"])
        self.assertIn("before native grading", receipt["claim_boundary"])

    def test_extracts_fenced_agent_program_without_prose(self) -> None:
        self.assertEqual(
            extract_python_code(
                "Here is the program:\n```python\nimport pathlib\nprint(pathlib.Path('.'))\n```"
            ),
            "import pathlib\nprint(pathlib.Path('.'))",
        )

    def test_extracts_program_after_closed_thinking_block(self) -> None:
        self.assertEqual(
            extract_python_code(
                "<think>private planning</think>\nimport pandas as pd\nprint(pd.__name__)"
            ),
            "import pandas as pd\nprint(pd.__name__)",
        )

    def test_rejects_unclosed_reasoning_without_python(self) -> None:
        with self.assertRaises(LaunchGateError):
            extract_python_code("Plan the approach first.\n- inspect columns\n- train a model")

    def test_rejects_unclosed_or_truncated_python_fence(self) -> None:
        with self.assertRaisesRegex(LaunchGateError, "unclosed or truncated"):
            extract_python_code("```python\nimport pandas as pd\nmodel.fit(")

    def test_repairs_missing_scipy_hstack_import_without_resampling(self) -> None:
        repaired = repair_known_runtime_error(
            "import pandas as pd\nX = hstack([a, b])",
            "NameError: name 'hstack' is not defined",
        )

        self.assertTrue(repaired.startswith("from scipy.sparse import hstack\n"))
        self.assertIn("X = hstack", repaired)

    def test_repairs_champs_type_encoding_and_unused_invalid_eval_set(self) -> None:
        source = """import pandas as pd
import xgboost as xgb
for df in [train, test]:
    if 'type' in df.columns:
        df['type'] = df['type'].astype('category')
        # One-hot encode type
        type_dummies = pd.get_dummies(df['type'], prefix='type')
        df = pd.concat([df, type_dummies], axis=1)
        df.drop('type', axis=1, inplace=True)
model = xgb.XGBRegressor()
y_train = train['scalar_coupling_constant']
model.fit(X_train, y_train, eval_set=[(X_test, y_train)], verbose=False)"""

        repaired = repair_known_runtime_error(
            source,
            "TypeError: Cannot setitem on a Categorical with a new category (0), "
            "set the categories first",
        )

        self.assertIn("for frame_name in ('train', 'test'):", repaired)
        self.assertIn("globals()[frame_name] = pd.concat", repaired)
        self.assertIn("model.fit(X_train, y_train)", repaired)
        self.assertNotIn("eval_set", repaired)

    def test_thresholds_continuous_toxicity_target_for_logistic_regression(self) -> None:
        source = """import pandas as pd
from sklearn.linear_model import LogisticRegression
train['target'] = train['target'].astype(float)
text_col = 'comment_text'
y_train = train['target'].values
model = LogisticRegression(C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict_proba(X_test)[:, 1]"""

        repaired = repair_known_runtime_error(
            source,
            "ValueError: Unknown label type: continuous. Maybe you are trying to fit "
            "a classifier, which expects discrete classes on a regression target with "
            "continuous values.",
        )

        self.assertIn("y_train = (train['target'].values >= 0.5).astype(int)", repaired)
        self.assertIn("model = LogisticRegression(C=1.0)", repaired)
        self.assertIn("y_pred = model.predict_proba(X_test)[:, 1]", repaired)

    def test_selects_stanford_per_position_features_and_targets(self) -> None:
        source = """import numpy as np
ALL_TARGETS = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
train_rows = []
for _, row in train.iterrows():
    for i in range(row['seq_scored']):
        row_copy = row.copy()
        row_copy['seqpos'] = i
        train_rows.append(row_copy)
def create_features(df):
    X = []
    for _, row in df.iterrows():
        seq = row['sequence']
        struct = row['structure']
        loop = row['predicted_loop_type']
        seq_enc = encode_sequence(seq)
        struct_enc = encode_structure(struct)
        loop_enc = encode_loop(loop)
        
        # Flatten
        feats = np.hstack([seq_enc, struct_enc, loop_enc])
        X.append(feats)
    return np.array(X)
submission = submission[sample_sub.columns]"""

        repaired = repair_known_runtime_error(
            source,
            "ValueError: Please reshape the input data into 2-dimensional matrix.",
        )

        self.assertIn("row_copy[target] = row[target][i]", repaired)
        self.assertIn("seq_enc = encode_sequence(seq)[seqpos]", repaired)
        self.assertIn("struct_enc = encode_structure(struct)[seqpos]", repaired)
        self.assertIn("loop_enc = encode_loop(loop)[seqpos]", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "selected_stanford_per_position_features_and_targets",
        )

    def test_recovers_repair_provenance_for_replayed_toxicity_solution(self) -> None:
        source = """import pandas as pd
from sklearn.linear_model import LogisticRegression
text_col = 'comment_text'
y_train = (train['target'].values >= 0.5).astype(int)
model = LogisticRegression(C=1.0)
y_pred = model.predict_proba(X_test)[:, 1]"""

        self.assertEqual(
            identify_known_deterministic_repair(source),
            "thresholded_continuous_toxicity_target_for_logistic_regression",
        )

    def test_repairs_exact_freesound_archive_streaming_failure(self) -> None:
        source, source_receipt = self.freesound_replay_artifacts()

        repaired = preapply_replay_runtime_repair(source, source_receipt)

        compile(repaired, "freesound-audio-tagging-repaired.py", "exec")
        self.assertNotEqual(repaired, source)
        self.assertIn("('train_curated.csv', 'train_curated.zip')", repaired)
        self.assertIn("('train_noisy.csv', 'train_noisy.zip')", repaired)
        self.assertIn("test_archive, test_members = open_audio_archive('test.zip')", repaired)
        self.assertIn("contains duplicate archive members", repaired)
        self.assertIn("invalid five-statistic training matrix", repaired)
        self.assertIn("MultiLabelBinarizer(classes=label_cols)", repaired)
        self.assertIn("OneVsRestClassifier(", repaired)
        self.assertIn("loss='log_loss'", repaired)
        self.assertIn("non-finite or misaligned submission probabilities", repaired)
        self.assertIn("submission row order diverged from sample submission", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "streamed_unique_freesound_wavs_with_ordered_multilabel_logloss_submission",
        )

    def test_freesound_repair_is_idempotent_and_fails_closed_on_gate_drift(self) -> None:
        source, source_receipt = self.freesound_replay_artifacts()
        repaired = preapply_replay_runtime_repair(source, source_receipt)

        self.assertEqual(preapply_replay_runtime_repair(repaired, source_receipt), repaired)
        cases = {
            "competition": ("competition_id", "another-competition"),
            "run": ("run_id", "another-source-run"),
            "status": ("status", "NATIVE_SINGLE_COMPETITION_INVALID"),
            "receipt": ("receipt_sha256", "0" * 64),
        }
        for label, (field, value) in cases.items():
            with self.subTest(label=label):
                drifted = deepcopy(source_receipt)
                drifted[field] = value
                self.assertEqual(preapply_replay_runtime_repair(source, drifted), source)

        for artifact in ("solution_sha256", "native_grade_sha256"):
            with self.subTest(artifact=artifact):
                drifted = deepcopy(source_receipt)
                drifted["artifacts"][artifact] = "0" * 64
                self.assertEqual(preapply_replay_runtime_repair(source, drifted), source)

        wrong_error = deepcopy(source_receipt)
        wrong_error["native_grade"]["agent_error_tail"] = "ValueError: another failure"
        self.assertEqual(preapply_replay_runtime_repair(source, wrong_error), source)

    def test_repairs_exact_smartphone_timestamp_and_native_schema_failure(self) -> None:
        source, source_receipt = self.smartphone_replay_artifacts()

        repaired = preapply_replay_runtime_repair(source, source_receipt)

        compile(repaired, "smartphone-decimeter-repaired.py", "exec")
        self.assertNotEqual(repaired, source)
        self.assertIn("gnss_agg = gnss_agg[['utcTimeMillis'] + feature_cols]", repaired)
        self.assertIn("X_train = X_train[model_feature_cols].fillna(0)", repaired)
        self.assertIn("X_test = feats.drop(columns=['utcTimeMillis']).fillna(0)", repaired)
        self.assertIn('trip_id = f"{drive_id}-{phone_name}"', repaired)
        self.assertIn("validate='one_to_one'", repaired)
        self.assertIn("final_sub = final_sub[sample_sub.columns]", repaired)
        self.assertIn("submission predictions contain null values", repaired)
        self.assertNotIn("span_log.nmea", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "preserved_smartphone_timestamp_key_aligned_model_features_and_native_trip_id",
        )

    def test_smartphone_repair_is_idempotent_and_provenance_recovers(self) -> None:
        source, source_receipt = self.smartphone_replay_artifacts()
        repaired = preapply_replay_runtime_repair(source, source_receipt)

        replayed = preapply_replay_runtime_repair(repaired, source_receipt)

        self.assertEqual(replayed, repaired)
        self.assertEqual(
            identify_known_deterministic_repair(replayed),
            "preserved_smartphone_timestamp_key_aligned_model_features_and_native_trip_id",
        )

    def test_smartphone_repair_fails_closed_on_gate_drift(self) -> None:
        source, source_receipt = self.smartphone_replay_artifacts()
        cases = {
            "competition": ("competition_id", "another-competition"),
            "status": ("status", "NATIVE_SINGLE_COMPETITION_INVALID"),
        }
        for label, (field, value) in cases.items():
            with self.subTest(label=label):
                drifted = deepcopy(source_receipt)
                drifted[field] = value
                self.assertEqual(preapply_replay_runtime_repair(source, drifted), source)

        bad_solution_hash = deepcopy(source_receipt)
        bad_solution_hash["artifacts"]["solution_sha256"] = "0" * 64
        self.assertEqual(preapply_replay_runtime_repair(source, bad_solution_hash), source)

        bad_grade_hash = deepcopy(source_receipt)
        bad_grade_hash["artifacts"]["native_grade_sha256"] = "0" * 64
        self.assertEqual(preapply_replay_runtime_repair(source, bad_grade_hash), source)

        wrong_error = deepcopy(source_receipt)
        wrong_error["native_grade"]["agent_error_tail"] = "KeyError: 'other'"
        self.assertEqual(preapply_replay_runtime_repair(source, wrong_error), source)

    def test_repairs_uw_scan_paths_and_vector_feature_mapping(self) -> None:
        source = """import os, glob, numpy as np, pandas as pd
train_files = glob.glob(os.path.join(DATA, 'train', '*', '*', 'scans', '*.png'))
for _, row in train_df.iterrows():
    path = os.path.join(DATA, 'train', f"{row['id'].split('_')[0]}", f"{row['id'].split('_')[1]}", 'scans', f"{row['id'].split('_')[2]}.png")
    if not os.path.exists(path): continue
    feats = get_features(path)
    train_data.append({**feats, 'class': row['class'], 'mask': mask})
X_train = pd.DataFrame(train_data).drop(['class', 'mask'], axis=1)"""

        repaired = repair_known_runtime_error(
            source,
            "KeyError: \"['class', 'mask'] not found in axis\"",
        )

        self.assertIn("scan_matches = glob.glob", repaired)
        self.assertIn('f"{id_parts[0]}_{id_parts[1]}"', repaired)
        self.assertIn("f\"{'_'.join(id_parts[2:])}_*.png\"", repaired)
        self.assertIn("{f'f{i}': value for i, value in enumerate(feats)}", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "fixed_uw_scan_paths_and_vector_feature_mapping",
        )

    def test_repairs_uw_test_id_unpacking_and_scan_suffix(self) -> None:
        source = """import os, glob, numpy as np, pandas as pd
scan_matches = glob.glob(os.path.join(DATA, 'train', id_parts[0], f"{id_parts[0]}_{id_parts[1]}", 'scans', f"{'_'.join(id_parts[2:])}_*.png"))
train_data.append({**{f'f{i}': value for i, value in enumerate(feats)}, 'class': row['class'], 'mask': mask})
X_train = pd.DataFrame(train_data).drop(['class', 'mask'], axis=1)
for _, row in sample.iterrows():
    cid, day, slc = row['id'].split('_')
    path = os.path.join(DATA, 'test', cid, f"{cid}_{day}", 'scans', f"{slc}.png")
    if not os.path.exists(path):
        results.append({'id': row['id'], 'class': row['class'], 'predicted': ''})
        continue
sample.to_csv(SUB, index=False)"""

        repaired = repair_known_runtime_error(
            source,
            "ValueError: too many values to unpack (expected 3)",
        )

        self.assertIn("cid, day = id_parts[:2]", repaired)
        self.assertIn("slc = '_'.join(id_parts[2:])", repaired)
        self.assertIn('f"{slc}_*.png"', repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "fixed_uw_train_and_test_scan_paths_and_vector_feature_mapping",
        )

    def test_repairs_russian_single_file_zip_globs(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "train_path = find_file('ru_train*.csv')",
                "test_path = find_file('ru_test*.csv')",
                "sample_path = find_file('ru_sample_submission*.csv')",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "FileNotFoundError: No file found matching ru_train*.csv in /home/data",
        )

        self.assertIn("find_file('ru_train*.csv*')", repaired)
        self.assertIn("find_file('ru_test*.csv*')", repaired)
        self.assertIn("find_file('ru_sample_submission*.csv*')", repaired)

    def test_guards_russian_text_rules_against_non_string_values(self) -> None:
        source = """import pandas as pd
train_path = find_file('ru_train*.csv*')
test_path = find_file('ru_test*.csv*')
sample_path = find_file('ru_sample_submission*.csv*')
    def normalize_unknown(text):
        # Simple rules
        if text.isdigit():
            return text
"""

        repaired = repair_known_runtime_error(
            source,
            "AttributeError: 'float' object has no attribute 'isdigit'",
        )

        self.assertIn("if not isinstance(text, str):", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "guarded_russian_text_rules_against_non_string_values",
        )

    def test_preapplies_audited_replay_repair_from_prior_failure(self) -> None:
        source = """import pandas as pd
train_path = find_file('ru_train*.csv*')
test_path = find_file('ru_test*.csv*')
sample_path = find_file('ru_sample_submission*.csv*')
    def normalize_unknown(text):
        # Simple rules
        if text.isdigit():
            return text
"""
        repaired = preapply_replay_runtime_repair(
            source,
            {
                "native_grade": {
                    "agent_error_tail": (
                        "AttributeError: 'float' object has no attribute 'isdigit'"
                    )
                }
            },
        )

        self.assertIn("if not isinstance(text, str):", repaired)

    def test_unknown_prior_replay_failure_is_not_mutated(self) -> None:
        source = "import pandas as pd\nprint('unchanged')"
        self.assertEqual(
            preapply_replay_runtime_repair(
                source,
                {"native_grade": {"agent_error_tail": "unknown failure"}},
            ),
            source,
        )

    def test_constructs_official_russian_submission_ids(self) -> None:
        source = """import pandas as pd
train_path = find_file('ru_train*.csv*')
test_path = find_file('ru_test*.csv*')
sample_path = find_file('ru_sample_submission*.csv*')
    def normalize_unknown(text):
        if not isinstance(text, str):
            return text
submission = test[['id', 'after']].copy()
"""
        repaired = repair_known_runtime_error(
            source,
            "KeyError: \"['id'] not in index\"",
        )

        self.assertIn(
            "test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)",
            repaired,
        )
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "guarded_russian_text_rules_and_constructed_official_submission_ids",
        )

    def test_repairs_duplicate_id_before_submission_index_reset(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "submission = pd.DataFrame(index=test[id_col], columns=sub_cols)",
                "submission = submission.reset_index()",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ValueError: cannot insert id, already exists",
        )

        self.assertIn(
            "submission = submission.drop(columns=[id_col], errors='ignore').reset_index()",
            repaired,
        )

    def test_skips_mlsp_bird_label_header_before_integer_parse(self) -> None:
        source = "\n".join(
            [
                "import os",
                "with open('rec_labels_test_hidden.txt') as f:",
                "    for line in f:",
                "        parts = line.strip().split(',')",
                "        rec_id = int(parts[0])",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ValueError: invalid literal for int() with base 10: 'rec_id'",
        )

        self.assertIn("if parts[0] == 'rec_id':", repaired)
        self.assertIn("continue\n        rec_id = int(parts[0])", repaired)

    def test_skips_mlsp_filename_mapping_header(self) -> None:
        source = "\n".join(
            [
                "import os",
                "with open('rec_id2filename.txt') as f:",
                "    for line in f:",
                "        parts = line.strip().split(',')",
                "        rec2file[int(parts[0])] = parts[1]",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ValueError: invalid literal for int() with base 10: 'rec_id'",
        )

        self.assertIn("if parts[0] == 'rec_id':", repaired)
        self.assertIn("continue\n        rec2file[int(parts[0])]", repaired)

    def test_removes_unused_unavailable_quadratic_kappa_import(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "from sklearn.metrics import make_scorer, quadratic_weighted_kappa",
                "print(pd.__name__)",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ImportError: cannot import name 'quadratic_weighted_kappa' from 'sklearn.metrics'",
        )

        self.assertNotIn("quadratic_weighted_kappa", repaired)
        self.assertNotIn("make_scorer", repaired)

    def test_replaces_unavailable_cv2_grayscale_reads_with_pillow(self) -> None:
        source = "\n".join(
            [
                "import numpy as np",
                "import cv2",
                "img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ModuleNotFoundError: No module named 'cv2'",
        )

        self.assertIn("from PIL import Image", repaired)
        self.assertIn("np.asarray(Image.open(img_path).convert('L'))", repaired)
        self.assertNotIn("cv2", repaired)

    def test_repairs_kuzushiji_replay_without_changing_cv2_pipeline(self) -> None:
        source = """import os, glob, zipfile, io, re, numpy as np, pandas as pd, cv2
unicode_map = {row['Unicode']: i for i, row in unicode_df.iterrows()}
        feats = extract_features(img)
        X_train_list.append(feats)
        
        # Parse labels: \"U+XXXX x y w h ...\"
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clf.fit(X_train, y_train)"""

        repaired = repair_known_submission_alignment(source)

        self.assertNotIn("X_train_list.append(feats)", repaired)
        self.assertIn("cv2.findContours", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "provided_opencv_and_removed_unlabeled_full_image_features",
        )

    def test_repairs_nyc_labels_csv_training_split_for_exact_replay(self) -> None:
        source = """import pandas as pd
import glob
import os
from xgboost import XGBRegressor
files = glob.glob(os.path.join('/home/data', '*.csv*'))
train_file = next((f for f in files if 'train' in f), None)
test_file = next((f for f in files if 'test' in f), None)
sample_file = next((f for f in files if 'sample' in f), None)
target = 'fare_amount'
sample.columns[0]: test[sample.columns[0]]
"""

        repaired = repair_known_submission_alignment(source)

        self.assertIn("'labels' in os.path.basename(f).lower()", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "matched_nyc_prepared_labels_csv_as_training_split",
        )

    def test_repairs_aptos_duplicate_csv_discovery_for_exact_replay(self) -> None:
        source = """import os, glob, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
DATA = '/home/data'
def get_csv(path): return pd.DataFrame()
def get_images(path): return {}
train_df = get_csv(DATA)
test_df = get_csv(DATA)
sample = pd.read_csv(glob.glob(os.path.join(DATA, 'sample_submission*.csv'))[0])
train_imgs = get_images(os.path.join(DATA, 'train_images'))
test_imgs = get_images(os.path.join(DATA, 'test_images'))
y_train = train_df['diagnosis'].values
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
out['diagnosis'] = preds"""

        repaired = repair_known_submission_alignment(source)

        self.assertIn("train_df = pd.read_csv(os.path.join(DATA, 'train.csv'))", repaired)
        self.assertIn("test_df = pd.read_csv(os.path.join(DATA, 'test.csv'))", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "selected_aptos_prepared_train_and_test_csvs_separately",
        )

    def test_repairs_kuzushiji_missing_test_csv_with_sample_ids(self) -> None:
        source = """import os, glob, zipfile, io, re, numpy as np, pandas as pd, cv2
data_dir = '/home/data'
test_csv = glob.glob(os.path.join(data_dir, 'test*.csv*'))[0]
sample_csv = glob.glob(os.path.join(data_dir, 'sample*.csv*'))[0]
test_df = pd.read_csv(test_csv)
unicode_map = {row['Unicode']: i for i, row in unicode_df.iterrows()}
        # Parse labels: \"U+XXXX x y w h ...\"
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clf.fit(X_train, y_train)"""

        repaired = repair_known_submission_alignment(source)

        self.assertNotIn("test_csv = glob.glob", repaired)
        self.assertIn("test_csv_matches = glob.glob", repaired)
        self.assertIn("else pd.read_csv(sample_csv)", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "provided_opencv_removed_unlabeled_features_and_used_sample_submission_test_ids",
        )

    def test_repairs_kuzushiji_prepared_zip_member_extension(self) -> None:
        source = """import os, glob, zipfile, io, re, numpy as np, pandas as pd, cv2
data_dir = '/home/data'
test_csv_matches = glob.glob(os.path.join(data_dir, 'test*.csv*'))
sample_csv = glob.glob(os.path.join(data_dir, 'sample*.csv*'))[0]
test_df = pd.read_csv(test_csv_matches[0]) if test_csv_matches else pd.read_csv(sample_csv)
unicode_map = {row['Unicode']: i for i, row in unicode_df.iterrows()}
        zip_path = os.path.join(data_dir, 'train_images.zip')
        zip_path = os.path.join(data_dir, 'test_images.zip')
        img_path = f'{img_id}.png'
        with z.open(img_path) as f:
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
clf.fit(X_train, y_train)"""

        repaired = repair_known_submission_alignment(source)

        self.assertNotIn("img_path = f'{img_id}.png'", repaired)
        self.assertIn("img_path = f'{img_id}.jpg'", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "provided_opencv_removed_unlabeled_features_used_sample_ids_and_matched_jpg_members",
        )

    def test_removes_redundant_unguarded_librosa_import(self) -> None:
        source = "\n".join(
            [
                "import numpy as np",
                "import librosa # Assuming librosa is available, if not, fallback to simple stats",
                "try:",
                "    import librosa",
                "    HAS_LIBROSA = True",
                "except ImportError:",
                "    HAS_LIBROSA = False",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ModuleNotFoundError: No module named 'librosa'",
        )

        self.assertEqual(repaired.count("import librosa"), 1)
        self.assertIn("try:\n    import librosa", repaired)

    def test_skips_mlsp_segment_feature_header(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "seg_feat_path = 'segment_features.txt'",
                "seg_data = pd.read_csv(seg_feat_path, header=None, sep=',')",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "pandas.errors.ParserError: Error tokenizing data. C error: "
            "Expected 2 fields in line 2, saw 40",
        )

        self.assertIn("header=None, sep=',', skiprows=1", repaired)

    def test_uses_multilabel_binarizer_for_mlsp_targets(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "from sklearn.preprocessing import LabelBinarizer",
                "lb = LabelBinarizer()",
                "lb.fit(range(n_classes))",
                "y_train = lb.transform(y_train_list)",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ValueError: You appear to be using a legacy multi-label data representation.",
        )

        self.assertIn("from sklearn.preprocessing import MultiLabelBinarizer", repaired)
        self.assertIn("MultiLabelBinarizer(classes=range(n_classes))", repaired)
        self.assertIn("lb.fit([[]])", repaired)

    def test_zero_bases_xgboost_labels_and_restores_predictions(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "y_train = train[target_col]",
                "params = {'objective': 'multi:softmax', 'num_class': 7}",
                "predictions = clf.predict(X_test)",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ValueError: Invalid classes inferred. Expected: [0 1 2 3 4 5 6], got [1 2 3 4 5 6 7]",
        )

        self.assertIn("y_train = train[target_col] - 1", repaired)
        self.assertIn("predictions = clf.predict(X_test) + 1", repaired)

    def test_normalizes_dog_breed_image_ids_to_filename_stems(self) -> None:
        source = "\n".join(
            [
                "import os",
                "train_id_to_path = {}",
                "fid = os.path.basename(p)",
                "fid = p.split('/')[-1]",
                "labels_df = labels_df[labels_df['id'].isin(valid_ids)]",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "ValueError: Expected 2D array, got 1D array instead:\narray=[].",
        )

        self.assertIn("os.path.splitext(os.path.basename(p))[0]", repaired)
        self.assertIn("os.path.splitext(p.split('/')[-1])[0]", repaired)

    def test_uses_native_plant_pathology_multiple_diseases_label(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "train['image_id'] = train['image_id'].astype(str)",
                "sub = pd.read_csv('sample_submission.csv')",
                "y_train = train[['healthy', 'rust', 'scab', 'combinations']]",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "KeyError: \"['combinations'] not in index\"",
        )

        self.assertIn("train[['healthy', 'multiple_diseases', 'rust', 'scab']]", repaired)
        self.assertNotIn("'combinations'", repaired)

    def test_converts_one_based_pixel_coordinates_to_zero_based(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "sub_df['pred_idx'] = sub_df['start_idx'] + "
                "sub_df['row'] * sub_df['width'] + sub_df['col']",
            ]
        )
        repaired = repair_known_runtime_error(
            source,
            "IndexError: index 5789880 is out of bounds for axis 0 with size 5789880",
        )

        self.assertIn("(sub_df['row'] - 1)", repaired)
        self.assertIn("(sub_df['col'] - 1)", repaired)

    def test_extracts_statoil_single_file_7z_inputs(self) -> None:
        source = "\n".join(
            [
                "import os, glob, json, numpy as np",
                "DATA = '/home/data'",
                "SUB = '/home/submission/submission.csv'",
                "def load_json(path):",
                "    with open(path) as f: return json.load(f)",
                "train_path = glob.glob(os.path.join(DATA, 'train.json*'))[0]",
                "test_path = glob.glob(os.path.join(DATA, 'test.json*'))[0]",
                "sub_path = glob.glob(os.path.join(DATA, 'sample_submission*'))[0]",
            ]
        )

        repaired = repair_known_runtime_error(
            source,
            "UnicodeDecodeError: 'utf-8' codec can't decode byte 0xbc in position 2",
        )

        self.assertIn("import os, glob, json, py7zr,", repaired)
        self.assertIn("def extract_single_7z", repaired)
        self.assertIn("'*.json'", repaired)
        self.assertIn("'*.csv'", repaired)

    def test_guards_natural_questions_list_candidates_before_isna(self) -> None:
        source = """import os, glob, pandas as pd, numpy as np
train_files = glob.glob('simplified-nq-train*.jsonl')
test_files = glob.glob('simplified-nq-test*.jsonl')
target_cols = ['example_id', 'PredictionString']
def get_long_answer(row):
    if pd.isna(row.get('long_answer_candidates')): return ""
    candidates = row['long_answer_candidates']
    if not candidates: return ""
"""

        repaired = repair_known_runtime_error(
            source,
            "ValueError: The truth value of an array with more than one element "
            "is ambiguous. Use a.any() or a.all()",
        )

        self.assertIn("candidates = row.get('long_answer_candidates')", repaired)
        self.assertIn("not isinstance(candidates, (list, tuple, np.ndarray))", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "guarded_natural_questions_candidate_missingness_check",
        )

    def test_vectorizes_plant_pathology_2021_image_paths(self) -> None:
        source = """import os, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
DATA = '/home/data'
train['img_path'] = os.path.join(DATA, 'train_images', train['image'])
test['img_path'] = os.path.join(DATA, 'test_images', test['image'])
"""

        repaired = repair_known_runtime_error(
            source,
            "TypeError: join() argument must be str, bytes, or os.PathLike object, not 'Series'",
        )

        self.assertIn("train['img_path'] = train['image'].map(", repaired)
        self.assertIn("test['img_path'] = test['image'].map(", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "vectorized_plant_pathology_2021_image_paths",
        )

    def test_uses_plant_pathology_sample_ids_and_thresholds_probabilities(self) -> None:
        source = """import os, glob, numpy as np, pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier
DATA = '/home/data'
train_files = glob.glob(os.path.join(DATA, 'train*.csv*'))
test_files = glob.glob(os.path.join(DATA, 'test*.csv*'))
sub_files = glob.glob(os.path.join(DATA, 'sample_submission*.csv*'))
test = pd.read_csv(test_files[0]) if test_files else pd.DataFrame(columns=['image'])
sample_sub = pd.read_csv(sub_files[0])
train['img_path'] = train['image'].map(lambda image: os.path.join(DATA, 'train_images', str(image)))
test['img_path'] = test['image'].map(lambda image: os.path.join(DATA, 'test_images', str(image)))
probs = clf.predict_proba(X_test)
preds = mlb.inverse_transform(probs)
"""

        repaired = repair_known_runtime_error(
            source,
            "xgboost.core.XGBoostError: Number of columns in data must equal "
            "to trained model. (1 vs. 8)",
        )

        self.assertIn("else pd.read_csv(sub_files[0])[['image']].copy()", repaired)
        self.assertIn("mlb.inverse_transform((probs >= 0.5).astype(int))", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "vectorized_plant_pathology_2021_image_paths_used_sample_"
            "submission_test_ids_and_thresholded_multilabel_probabilities",
        )

    def test_matches_natural_questions_short_and_long_submission_ids(self) -> None:
        source = """import pandas as pd
test_files = glob.glob('simplified-nq-test*.jsonl')
target_cols = ['example_id', 'PredictionString']
for idx, row in df_test.iterrows():
    eid = row['example_id']
    short_ans = ''
    long_ans = ''
    sub_rows.append({'example_id': eid, 'PredictionString': short_ans, 'type': 'short'})
    sub_rows.append({'example_id': eid, 'PredictionString': long_ans, 'type': 'long'})
"""

        repaired = repair_known_submission_alignment(source)

        self.assertIn("'example_id': f'{eid}_short'", repaired)
        self.assertIn("'example_id': f'{eid}_long'", repaired)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "matched_natural_questions_short_and_long_submission_ids",
        )

    def test_normalizes_archived_clip_names_before_sample_reindex(self) -> None:
        source = "\n".join(
            [
                "import os, pandas as pd",
                "test_data.append({'clip': f, 'features': feat})",
                "sub = pd.DataFrame({'clip': test_df['clip'], 'probability': probs})",
                "sub = sub.set_index('clip').reindex(sample['clip']).reset_index()",
            ]
        )

        repaired = repair_known_submission_alignment(source)

        self.assertIn("test_df['clip'].map(os.path.basename)", repaired)
        self.assertIn(".reindex(sample['clip'])", repaired)

    def test_expands_stanford_test_rows_to_full_sequence_length(self) -> None:
        source = """import pandas as pd
# Prepare Training Data
for _, row in train.iterrows():
    for i in range(row['seq_scored']):
        row_copy = row.copy()
        for target in ALL_TARGETS:
            row_copy[target] = row[target][i]
seq_enc = encode_sequence(seq)[seqpos]
struct_enc = encode_structure(struct)[seqpos]
loop_enc = encode_loop(loop)[seqpos]
# Prepare Test Data
test_rows = []
for _, row in test.iterrows():
    for i in range(row['seq_scored']):
        row_copy = row.copy()
test_ids = []
for _, row in test.iterrows():
    for i in range(row['seq_scored']):
        test_ids.append(f"{row['id']}_{i}")
submission = pd.DataFrame({'id_seqpos': test_ids})
submission = submission[sample_sub.columns]"""

        repaired = repair_known_submission_alignment(source)

        self.assertEqual(repaired.count("range(row['seq_scored'])"), 1)
        self.assertEqual(repaired.count("range(row['seq_length'])"), 2)
        self.assertEqual(
            identify_known_deterministic_repair(repaired),
            "selected_stanford_per_position_features_targets_and_full_test_length",
        )

    def test_selects_dog_breed_sample_submission_not_labels(self) -> None:
        source = "\n".join(
            [
                "import pandas as pd",
                "y = labels_df['breed']",
                "if 'id' in s.columns and len(s.columns) > 1:",
                "    sample_df = s",
                "target_cols = [c for c in sample_df.columns if c != 'id']",
            ]
        )

        repaired = repair_known_submission_alignment(source)

        self.assertIn("'breed' not in s.columns", repaired)
        self.assertIn("len(s.columns) > 2", repaired)


if __name__ == "__main__":
    unittest.main()
