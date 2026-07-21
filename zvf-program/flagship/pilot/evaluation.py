from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import sha256_file
from .remote_core import gsm8k_reward, math500_reward, prompt_messages


class EvaluationContractError(RuntimeError):
    """Held-out evidence is incomplete or cannot be independently recomputed."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row_fingerprint(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "row_sha256"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evidence_row(
    *,
    index: int,
    source_index: int,
    regime: str,
    question: str,
    answer: str,
    completion_text: str,
    generated_tokens: int,
) -> dict[str, Any]:
    if index < 0 or source_index < 0 or generated_tokens <= 0:
        raise EvaluationContractError("evaluation indices and generated tokens must be positive")
    reward = (
        gsm8k_reward(completion_text, answer)
        if regime == "balanced_equal_length"
        else math500_reward(completion_text, answer)
    )
    record = {
        "index": index,
        "source_index": source_index,
        "regime": regime,
        "question_sha256": _sha256_text(question),
        "answer": answer,
        "completion_text": completion_text,
        "completion_sha256": _sha256_text(completion_text),
        "generated_tokens": generated_tokens,
        "correct": int(reward),
    }
    record["row_sha256"] = _row_fingerprint(record)
    return record


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_evidence(
    path: Path,
    *,
    regime: str,
    questions: Sequence[str],
    answers: Sequence[str],
    source_indices: Sequence[int],
) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    heldout_n = len(questions)
    if not (len(answers) == len(source_indices) == heldout_n == len(lines)):
        raise EvaluationContractError("held-out evidence row count mismatch")
    rows = [json.loads(line) for line in lines]
    observed_hashes: set[str] = set()
    correct = 0
    generated_tokens = 0
    for index, (row, question, answer, source_index) in enumerate(
        zip(rows, questions, answers, source_indices, strict=True)
    ):
        if row.get("index") != index:
            raise EvaluationContractError(f"held-out index is not contiguous at row {index}")
        if row.get("source_index") != source_index or row.get("regime") != regime:
            raise EvaluationContractError(f"held-out identity mismatch at row {index}")
        if row.get("question_sha256") != _sha256_text(question) or row.get("answer") != answer:
            raise EvaluationContractError(f"held-out source mismatch at row {index}")
        completion = row.get("completion_text")
        if not isinstance(completion, str):
            raise EvaluationContractError(f"held-out completion is not text at row {index}")
        if row.get("completion_sha256") != _sha256_text(completion):
            raise EvaluationContractError(f"completion hash mismatch at row {index}")
        if row.get("row_sha256") != _row_fingerprint(row):
            raise EvaluationContractError(f"row hash mismatch at row {index}")
        if row["row_sha256"] in observed_hashes:
            raise EvaluationContractError(f"duplicate held-out row hash at row {index}")
        observed_hashes.add(row["row_sha256"])
        recomputed = (
            gsm8k_reward(completion, answer)
            if regime == "balanced_equal_length"
            else math500_reward(completion, answer)
        )
        if row.get("correct") != int(recomputed):
            raise EvaluationContractError(f"held-out correctness mismatch at row {index}")
        tokens = row.get("generated_tokens")
        if not isinstance(tokens, int) or tokens <= 0:
            raise EvaluationContractError(f"held-out token count invalid at row {index}")
        correct += int(recomputed)
        generated_tokens += tokens
    return {
        "heldout_n": heldout_n,
        "correct": correct,
        "accuracy": correct / heldout_n,
        "generated_tokens": generated_tokens,
        "unique_row_hashes": len(observed_hashes),
        "evidence_sha256": sha256_file(path),
    }


def _completion_tokens(sequence: Any, *, prompt_width: int, eos_token_id: int) -> list[int]:
    values = [int(token) for token in sequence[prompt_width:].tolist()]
    if eos_token_id in values:
        values = values[: values.index(eos_token_id) + 1]
    return values or [eos_token_id]


def evaluate_model(
    *,
    model: Any,
    tokenizer: Any,
    regime: str,
    questions: Sequence[str],
    answers: Sequence[str],
    source_indices: Sequence[int],
    output_path: Path,
    max_prompt_length: int,
    max_completion_length: int,
    batch_size: int = 8,
) -> dict[str, Any]:
    import torch

    if not (len(questions) == len(answers) == len(source_indices)) or not questions:
        raise EvaluationContractError("held-out source arrays are empty or misaligned")
    if batch_size <= 0:
        raise EvaluationContractError("evaluation batch size must be positive")
    rows: list[dict[str, Any]] = []
    started_at = time.monotonic()
    model.eval()
    device = next(model.parameters()).device
    for start in range(0, len(questions), batch_size):
        stop = min(start + batch_size, len(questions))
        rendered = [
            tokenizer.apply_chat_template(
                prompt_messages(regime, questions[index]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for index in range(start, stop)
        ]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        ).to(device)
        # Pinned to the deterministic math backend; see training.completion_logps.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_return_sequences=1,
                max_new_tokens=max_completion_length,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        prompt_width = encoded.input_ids.shape[1]
        for offset, sequence in enumerate(generated):
            index = start + offset
            tokens = _completion_tokens(
                sequence,
                prompt_width=prompt_width,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(tokens, skip_special_tokens=True)
            rows.append(
                evidence_row(
                    index=index,
                    source_index=int(source_indices[index]),
                    regime=regime,
                    question=questions[index],
                    answer=answers[index],
                    completion_text=completion,
                    generated_tokens=len(tokens),
                )
            )
    atomic_jsonl(output_path, rows)
    summary = validate_evidence(
        output_path,
        regime=regime,
        questions=questions,
        answers=answers,
        source_indices=source_indices,
    )
    return {**summary, "wall_clock_seconds": time.monotonic() - started_at}
