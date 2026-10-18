"""Local student-trajectory distillation, protocol student-trajectory-kl-v1.

Fresh student samples define the histories; the teacher only scores them. The
objective is token-mean forward KL(teacher || student), with stop-gradient through
sampling (a semi-gradient, not a policy-gradient estimate of state occupancy).
No optimizer, network, artifact-writing or model-loading side effects live here.
"""

from dataclasses import dataclass
import json

import torch

from platform_hybrid.experiments.research_contracts import distillation_loss

PROTOCOL_VERSION = "student-trajectory-kl-v1"


@dataclass(frozen=True)
class StudentTrajectory:
    """Exact token snapshot, independent of mutable generation buffers."""

    input_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    behavior_version: int
    sampling_config_json: str
    protocol_version: str = PROTOCOL_VERSION


def student_trajectory_loss(
    student,
    teacher,
    input_ids,
    attention_mask,
    *,
    max_new_tokens,
    eos_token_id,
    pad_token_id,
    behavior_version,
    temperature=2.0,
):
    """Generate and score fresh continuations under the same eval-mode student.

    A fresh, unwarped generation configuration uses unit sampling temperature,
    full vocabulary and one sequence. Each row is generated separately
    to avoid padding-dependent causal positions. EOS is scored once, padding never.
    Only local, single-device causal models are supported by this bounded contract.
    Returned records are provenance, never a reusable training replay buffer.
    """
    if student is teacher:
        raise ValueError("Teacher and student must be distinct models")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Aligned rank-two prompt IDs and attention masks required")
    if input_ids.shape[0] == 0 or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("Nonempty batch and positive generation budget required")
    if behavior_version < 0:
        raise ValueError("Nonnegative behavior version required")
    if eos_token_id is None or pad_token_id is None:
        raise ValueError("Explicit EOS and padding token IDs required")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Binary prompt attention masks required")
    from transformers import GenerationConfig

    # Never inherit a checkpoint's top-k, forced tokens or repetition penalties.
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        num_return_sequences=1,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        return_dict_in_generate=False,
        max_new_tokens=max_new_tokens,
    )
    # Transformers 5 fills unset fields from model.generation_config. Resolve
    # global defaults, then temporarily isolate the model's config as well.
    defaults = getattr(generation_config, "_get_default_generation_params", None)
    if defaults is not None:
        generation_config.update(**defaults(), defaults_only=True)
    missing = object()
    previous_generation_config = getattr(student, "generation_config", missing)
    rows, records, masks = [], [], []
    student_training, teacher_training = student.training, teacher.training
    try:
        student.generation_config = generation_config
        student.eval()
        teacher.eval()
        for ids, attention in zip(input_ids, attention_mask, strict=True):
            active = torch.where(attention.bool())[0]
            if active.numel() == 0 or active[-1] - active[0] + 1 != active.numel():
                raise ValueError("Each prompt must have a nonempty contiguous token span")
            prompt = ids[active].unsqueeze(0)
            with torch.no_grad():
                sampled = student.generate(
                    input_ids=prompt,
                    attention_mask=torch.ones_like(prompt),
                    generation_config=generation_config,
                )
            if sampled.ndim != 2 or sampled.shape[0] != 1:
                raise ValueError("Generation must return one token sequence per prompt")
            width = prompt.shape[1]
            if not torch.equal(sampled[:, :width], prompt):
                raise ValueError("Generation changed the exact prompt tokens")
            response = sampled[0, width:]
            if not 0 < response.numel() <= max_new_tokens:
                raise ValueError("Generation returned no continuation or exceeded its budget")
            eos = torch.where(response == eos_token_id)[0]
            if eos.numel():
                response = response[: int(eos[0]) + 1]
            row = torch.cat((prompt[0], response)).detach().clone()
            mask = torch.arange(row.numel(), device=row.device) >= width
            rows.append(row)
            masks.append(mask)
            records.append(
                StudentTrajectory(
                    tuple(row.tolist()),
                    tuple(mask.tolist()),
                    behavior_version,
                    json.dumps(generation_config.to_dict(), sort_keys=True),
                )
            )
        # Right padding preserves each unpadded prompt's original causal positions.
        ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_token_id)
        response_mask = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=False
        )
        attention = (
            torch.arange(ids.shape[1], device=ids.device)[None, :]
            < torch.tensor([row.numel() for row in rows], device=ids.device)[:, None]
        )
        outputs = student(input_ids=ids, attention_mask=attention)
        with torch.no_grad():
            targets = teacher(input_ids=ids, attention_mask=attention)
        loss, _, _ = distillation_loss(
            outputs.logits, targets.logits, ids, attention, response_mask, temperature, alpha=0.0
        )
        return loss, outputs, tuple(records)
    finally:
        if previous_generation_config is missing:
            if hasattr(student, "generation_config"):
                del student.generation_config
        else:
            student.generation_config = previous_generation_config
        student.train(student_training)
        teacher.train(teacher_training)
