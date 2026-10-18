"""Tensor-only same-stack helpers shared by runner and synthetic CPU tests."""
from .samestack_protocol import completion_mask, position_ids_from_attention


def rollout_attention(prompt_attention, completion_ids, eos_ids):
    import torch
    mask = completion_mask(completion_ids, eos_ids)
    return torch.cat([prompt_attention, mask.to(prompt_attention.dtype)], dim=1), mask


def completion_log_probs(model, full, attention, prompt_length, want_features=False):
    """Match generation's left-padding positions when rescoring sampled tokens."""
    import torch.nn.functional as F
    output = model(input_ids=full, attention_mask=attention,
                   position_ids=position_ids_from_attention(attention),
                   output_hidden_states=want_features, use_cache=False)
    logits = output.logits[:, prompt_length - 1:-1, :].float()
    targets = full[:, prompt_length:]
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    features = output.hidden_states[-1][:, prompt_length - 1, :].detach().float() if want_features else None
    return log_probs, features


def clipped_response_loss(log_probs, old_log_probs, advantages, mask, clip, batch_size):
    import torch
    ratio = torch.exp(log_probs - old_log_probs)
    advantages = advantages.unsqueeze(1)
    token_loss = -torch.minimum(ratio * advantages,
                               torch.clamp(ratio, 1 - clip, 1 + clip) * advantages)
    token_loss = token_loss * mask
    return (token_loss.sum(1) / mask.sum(1).clamp(min=1.0)).sum() / batch_size
