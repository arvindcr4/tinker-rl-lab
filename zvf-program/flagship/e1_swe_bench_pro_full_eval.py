"""Run the pinned SWE-bench Pro evaluator with digest-pinned Modal images.

The upstream evaluator derives mutable Docker tags from instance IDs.  This
wrapper preserves the upstream evaluator and run scripts verbatim while
replacing only that URI lookup with the registry digests recorded before the
campaign starts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _load_image_map(path: Path, expected_instance_ids: list[str] | None = None) -> dict[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("image manifest root must be an object")
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("image manifest contains no images")
    result: dict[str, str] = {}
    for item in images:
        if not isinstance(item, dict):
            raise RuntimeError("image manifest record must be an object")
        instance_id = str(item.get("instance_id") or "")
        immutable_uri = str(item.get("immutable_uri") or "")
        digest = str(item.get("digest") or "")
        if (
            not instance_id
            or not immutable_uri
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise RuntimeError("image manifest contains an invalid record")
        if not immutable_uri.endswith("@" + digest):
            raise RuntimeError(f"image URI/digest mismatch for {instance_id}")
        if instance_id in result:
            raise RuntimeError(f"duplicate image record for {instance_id}")
        result[instance_id] = immutable_uri
    if expected_instance_ids is not None and (
        len(set(expected_instance_ids)) != len(expected_instance_ids)
        or set(result) != set(expected_instance_ids)
    ):
        raise RuntimeError("image manifest must match exact evaluation task inventory")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image_manifest_path", required=True)
    parser.add_argument("--evaluation_task_ids_path", required=True)
    wrapper_args, evaluator_args = parser.parse_known_args()

    root = Path(__file__).resolve().parents[2]
    evaluator_dir = root / "outputs/e1_swe_bench_pro/evaluator"
    sys.path.insert(0, str(evaluator_dir))
    import swe_bench_pro_eval as evaluator

    expected = json.loads(Path(wrapper_args.evaluation_task_ids_path).read_text())
    if (
        not isinstance(expected, list)
        or not expected
        or any(not isinstance(x, str) or not x for x in expected)
    ):
        raise RuntimeError("evaluation task inventory must be a nonempty string list")
    image_map = _load_image_map(Path(wrapper_args.image_manifest_path), expected)

    def immutable_image_uri(uid: str, _username: str, _repo: str = "") -> str:
        try:
            return image_map[uid]
        except KeyError as exc:
            raise RuntimeError(f"no digest-pinned image for {uid}") from exc

    evaluator.get_dockerhub_image_uri = immutable_image_uri
    sys.argv = [sys.argv[0], *evaluator_args]
    evaluator.main()


if __name__ == "__main__":
    main()
