"""Validated, byte-bound native GSM8K input artifacts (no provider execution)."""

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .artifacts import read_regular_bytes


SOURCE_ROWS = 500
_NUMERIC_LABEL = re.compile(r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numeric_label(value: object) -> str:
    if not isinstance(value, str) or not _NUMERIC_LABEL.fullmatch(value.strip()):
        raise ValueError("GSM8K label must be a complete finite numeric answer")
    label = value.strip().replace(",", "")
    try:
        finite = Decimal(label).is_finite()
    except InvalidOperation:
        finite = False
    if not finite:
        raise ValueError("GSM8K label must be a complete finite numeric answer")
    return label


def materialize_gsm8k(out_path: str, system_prompt: str) -> str:
    """Validate the complete fixed source before publishing a new JSONL file."""
    from datasets import load_dataset

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system prompt must be a nonempty string")
    dataset = load_dataset("openai/gsm8k", "main", split="train[:500]")
    records = []
    for index, row in enumerate(dataset, 1):
        if index > SOURCE_ROWS:
            raise ValueError(f"GSM8K source exceeds {SOURCE_ROWS} rows")
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise ValueError(f"GSM8K row {index} has no nonempty question")
        answer = row.get("answer")
        if not isinstance(answer, str) or "####" not in answer:
            raise ValueError(f"GSM8K row {index} has no final-answer delimiter")
        try:
            label = numeric_label(answer.rsplit("####", 1)[1])
        except ValueError as exc:
            raise ValueError(f"GSM8K row {index} has an invalid final numeric answer") from exc
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": row["question"] + " Give a numerical answer inside \\boxed{}.",
                    },
                ],
                "label": label,
            }
        )
    if len(records) != SOURCE_ROWS:
        raise ValueError(f"GSM8K source must contain {SOURCE_ROWS} rows, got {len(records)}")
    data = "".join(json.dumps(row) + "\n" for row in records)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The driver requires a fresh output directory. This also protects direct
    # calls from replacing an earlier artifact with a new dataset download.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(data)
    return str(output)


def _unique_json_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON fields are not supported")
        result[key] = value
    return result


def inspect_native_input(path: str) -> dict:
    """Validate and hash one snapshot of the JSONL chat-message source."""
    source = Path(path)
    data = read_regular_bytes(source)
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("native input must be UTF-8 JSONL") from exc
    if len(lines) != SOURCE_ROWS:
        raise ValueError(f"native input must contain {SOURCE_ROWS} rows, got {len(lines)}")
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line, object_pairs_hook=_unique_json_object)
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"native input row {index} is invalid JSON") from exc
        if not isinstance(row, dict) or set(row) != {"prompt", "label"}:
            raise ValueError(f"native input row {index} must contain prompt and label")
        prompt = row["prompt"]
        if (
            not isinstance(prompt, list)
            or len(prompt) != 2
            or any(
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
                for message in prompt
            )
            or [message["role"] for message in prompt] != ["system", "user"]
        ):
            raise ValueError(f"native input row {index} must contain system/user chat messages")
        try:
            normalized = numeric_label(row["label"])
        except ValueError as exc:
            raise ValueError(f"native input row {index} has an invalid label") from exc
        if row["label"] != normalized:
            raise ValueError(f"native input row {index} label must be normalized")
    return {
        "file": source.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(lines),
        "bytes": len(data),
        "format": "openrlhf-chat-messages-jsonl-v1",
        "source": "openai/gsm8k",
        "configuration": "main",
        "split": "train[:500]",
        "source_revision": None,
        "scope": "validated launch-input bytes before native tokenizer rendering; upstream revision is unpinned",
    }


def ensure_input_unchanged(path: str, receipt: dict) -> None:
    current = hashlib.sha256(read_regular_bytes(Path(path))).hexdigest()
    if current != receipt["sha256"]:
        raise RuntimeError("native input changed during execution; refusing a success receipt")
