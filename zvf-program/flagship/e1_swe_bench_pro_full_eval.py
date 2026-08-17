"""Run the pinned SWE-bench Pro evaluator with digest-pinned Modal images.

The upstream evaluator derives mutable Docker tags from instance IDs.  This
wrapper preserves the upstream evaluator and run scripts verbatim while
replacing only that URI lookup with the registry digests recorded before the
campaign starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_image_map(path: Path) -> dict[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, list) or not images:
        raise RuntimeError("image manifest contains no images")
    result: dict[str, str] = {}
    for item in images:
        instance_id = str(item.get("instance_id") or "")
        immutable_uri = str(item.get("immutable_uri") or "")
        digest = str(item.get("digest") or "")
        if not instance_id or not immutable_uri or not digest.startswith("sha256:"):
            raise RuntimeError("image manifest contains an invalid record")
        if not immutable_uri.endswith("@" + digest):
            raise RuntimeError(f"image URI/digest mismatch for {instance_id}")
        if instance_id in result:
            raise RuntimeError(f"duplicate image record for {instance_id}")
        result[instance_id] = immutable_uri
    return result


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image_manifest_path", required=True)
    wrapper_args, evaluator_args = parser.parse_known_args()

    root = Path(__file__).resolve().parents[2]
    evaluator_dir = root / "outputs/e1_swe_bench_pro/evaluator"
    sys.path.insert(0, str(evaluator_dir))
    import swe_bench_pro_eval as evaluator

    image_map = _load_image_map(Path(wrapper_args.image_manifest_path))

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
