"""Executable candidate objective specification; not an alternate training optimizer.

These CPU functions define what the v2 experiment intends to compare. Backend
parity must be checked in the exact image before any matched empirical claim.
"""
import torch


def group_advantages(rewards, epsilon=1e-8):
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("Expected [prompt, group] rewards with group size >=2")
    return (rewards-rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, correction=1, keepdim=True)+epsilon)


def sequence_mean(values, mask):
    if values.shape != mask.shape or not torch.all(mask.sum(dim=-1) > 0):
        raise ValueError("Every response needs at least one unmasked token")
    return (torch.where(mask.bool(), values, 0).sum(dim=-1)/mask.sum(dim=-1)).mean()


def policy_loss(log_probs, old_log_probs, advantages, mask, epsilon=.2):
    ratio = torch.exp(torch.where(mask.bool(), log_probs-old_log_probs, 0))
    if advantages.ndim == 1:
        advantages = advantages[:, None]
    surrogate = torch.minimum(ratio*advantages, ratio.clamp(1-epsilon, 1+epsilon)*advantages)
    return -sequence_mean(surrogate, mask)


def kl_loss(log_probs, reference_log_probs, mask):
    log_ratio = torch.where(mask.bool(), log_probs-reference_log_probs, 0)
    return sequence_mean(torch.exp(-log_ratio)-1+log_ratio, mask)
