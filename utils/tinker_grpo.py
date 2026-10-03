from __future__ import annotations

import random
from typing import Any, Callable, Dict, Sequence, Tuple

import torch, tinker, tinker.types as T
from transformers import AutoTokenizer


def _await_remote(future: Any, what: str) -> Any:
    """Block on a Tinker remote future, adding context on failure."""
    try:
        return future.result()
    except Exception as exc:
        raise RuntimeError(f"Tinker remote call failed during {what}: {exc}") from exc


def run_grpo_training(
    exp_name: str,
    model_name: str,
    rank: int,
    steps: int,
    lr: float,
    group_size: int,
    batch_size: int,
    save_every: int,
    examples: Sequence[Tuple[str, str] | Tuple[str, str, Any]],
    reward_fn: Callable[..., float],
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> Dict[str, Any]:
    print(f"[{exp_name}] Connecting to Tinker...")
    try:
        svc = tinker.ServiceClient(base_url=None)
        tc = svc.create_lora_training_client(base_model=model_name, rank=rank)
    except Exception as exc:
        raise RuntimeError(f"[{exp_name}] failed to connect to Tinker: {exc}") from exc
    try:
        # WARNING: trust_remote_code=True executes Python shipped in the HF repo for
        # model_name. Only pass model names from trusted, pinned sources.
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"[{exp_name}] failed to load tokenizer for {model_name}: {exc}"
        ) from exc
    w0 = _await_remote(tc.save_weights_for_sampler(name="s0"), "initial save_weights_for_sampler")
    sc = tc.create_sampling_client(model_path=w0.path)
    print(f"[{exp_name}] Run: {tc.model_id}")

    # Shared state for custom loss
    _adv: list[float] = []

    def loss_fn(data, lp):
        losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
        loss = torch.stack(losses).mean()
        return loss, {"loss": loss.item()}

    step_rewards = []
    zero_loss_steps = 0
    zero_reward_steps = 0

    for step in range(steps):
        batch = random.sample(examples, batch_size)
        all_data, all_advs, batch_r = [], [], []

        for item in batch:
            # item should be a tuple (prompt_text, expected_answer)
            prompt_text, ans = item[:2]

            pid = tok.encode(prompt_text, add_special_tokens=False)
            if len(pid) > 1024:
                pid = pid[:1024]
            sp = T.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            resp = _await_remote(
                sc.sample(T.ModelInput.from_ints(pid), num_samples=group_size, sampling_params=sp),
                f"sampling (step {step})",
            )

            rews = []
            for r in resp.sequences:
                text = tok.decode(list(r.tokens), skip_special_tokens=True)
                # Pass extra context to reward_fn if present (e.g. tool_name)
                if len(item) == 3:
                    r_val = reward_fn(text, ans, item[2])
                else:
                    r_val = reward_fn(text, ans)
                rews.append(r_val)

            mr = sum(rews) / len(rews)
            sr = (sum((r - mr) ** 2 for r in rews) / len(rews)) ** 0.5 + 1e-8
            advs = [(r - mr) / sr for r in rews]
            batch_r.extend(rews)

            for r, a in zip(resp.sequences, advs):
                rid = list(r.tokens)
                fid = pid + rid
                tid = fid[1:] + [0]
                all_data.append(
                    T.Datum(
                        model_input=T.ModelInput.from_ints(fid),
                        loss_fn_inputs={
                            "target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])
                        },
                    )
                )
                all_advs.append(a)

        if not all_data:
            continue

        _adv = all_advs
        result = _await_remote(
            tc.forward_backward_custom(data=all_data, loss_fn=loss_fn),
            f"forward_backward_custom (step {step})",
        )
        _await_remote(
            tc.optim_step(T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)),
            f"optim_step (step {step})",
        )

        avg = sum(batch_r) / len(batch_r)
        step_rewards.append(avg)
        loss_val = result.metrics.get("loss", 0)
        if abs(loss_val) < 1e-6:
            zero_loss_steps += 1
        if avg == 0:
            zero_reward_steps += 1

        print(
            f"[{exp_name}] {step + 1:3d}/{steps} | loss={loss_val:.4f} | reward={avg:.3f} | acc={avg * 100:.1f}%"
        )

        if (step + 1) % save_every == 0:
            tc.save_state(name=f"s{step + 1}")
            ckpt = _await_remote(
                tc.save_weights_for_sampler(name=f"s{step + 1}"),
                f"checkpoint save s{step + 1}",
            )
            sc = tc.create_sampling_client(model_path=ckpt.path)
            print(f"[{exp_name}]   -> ckpt s{step + 1}")

    tc.save_state(name="final")
    f = _await_remote(tc.save_weights_for_sampler(name="final"), "final save_weights_for_sampler")
    last10 = step_rewards[-10:]
    avg10 = sum(last10) / len(last10) if last10 else 0
    first5 = step_rewards[:5]
    avg_first5 = sum(first5) / len(first5) if first5 else 0
    max_r = max(step_rewards) if step_rewards else 0

    print(f"\n[{exp_name}] === FINAL REPORT ===")
    print(f"[{exp_name}] Model: {model_name} | LoRA rank: {rank}")
    print(f"[{exp_name}] Steps: {steps} | Group: {group_size} | LR: {lr}")
    print(f"[{exp_name}] First-5 avg accuracy: {avg_first5 * 100:.1f}%")
    print(f"[{exp_name}] Last-10 avg accuracy: {avg10 * 100:.1f}%")
    print(f"[{exp_name}] Peak accuracy: {max_r * 100:.1f}%")
    print(
        f"[{exp_name}] Zero-loss steps: {zero_loss_steps}/{steps} ({100 * zero_loss_steps / steps:.0f}%)"
    )
    print(
        f"[{exp_name}] Zero-reward steps: {zero_reward_steps}/{steps} ({100 * zero_reward_steps / steps:.0f}%)"
    )
    print(f"[{exp_name}] Run ID: {tc.model_id}")
    print(f"[{exp_name}] Sampler: {f.path}")
    print(f"[{exp_name}] Reward trace: {[round(r, 3) for r in step_rewards]}")

    return {
        "exp_name": exp_name,
        "model_name": model_name,
        "rank": rank,
        "steps": steps,
        "completed_steps": len(step_rewards),
        "failed": len(step_rewards) < steps,
        "first5_avg": avg_first5,
        "last10_avg": avg10,
        "peak": max_r,
        "zero_loss_steps": zero_loss_steps,
        "zero_reward_steps": zero_reward_steps,
        "reward_trace": step_rewards,
        "run_id": tc.model_id,
        "sampler_path": f.path,
    }
