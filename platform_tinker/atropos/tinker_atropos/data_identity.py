"""Content-addressed split manifests retained with environment checkpoints."""

import hashlib
import json


def split_manifest(train, evaluation, *, data_seed, label):
    def records(rows):
        return [
            {
                "id": row.get("task_id", row.get("env_id")),
                "content_sha256": hashlib.sha256(
                    json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            }
            for row in rows
        ]

    return {
        "version": "data-splits-v2",
        "label": label,
        "data_seed": data_seed,
        "train": records(train),
        "evaluation": records(evaluation),
    }
