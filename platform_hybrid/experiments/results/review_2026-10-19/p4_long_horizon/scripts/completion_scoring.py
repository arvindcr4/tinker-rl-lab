"""Score only completion targets; retain prompt context and original masks."""
import torch
import torch.nn.functional as F


def completion_logps(model, full, attention_mask, plen):
    end = int(attention_mask.any(dim=0).nonzero()[-1].item()) + 1
    positions = torch.arange(plen - 1, end - 1, device=full.device)
    logits = model(input_ids=full[:, :end], attention_mask=attention_mask[:, :end],
                   logits_to_keep=positions, use_cache=False).logits.float()
    targets = full[:, plen:end]
    lp = -F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                          targets.reshape(-1), reduction='none').view_as(targets)
    return lp, attention_mask[:, plen:end].float()


def check_equivalence():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(781)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=43, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        pad_token_id=0, eos_token_id=2, attention_dropout=0.0))
    # Unequal prompt padding, unequal completion padding, nonzero EOS, and
    # a microbatch whose longest completion is shorter than the full batch.
    tokens = torch.randint(3, 43, (3, 17))
    tokens[0, :2] = 0
    tokens[1, :1] = 0
    tokens[0, 8:] = 0
    tokens[1, 13:] = 0
    tokens[0, 7] = 2
    attention = (tokens != 0).long()
    plen = 5
    weight = torch.tensor([1.3, -0.7, 0.2]).unsqueeze(1)
    for algo in ('grpo', 'dr_grpo'):
        model.zero_grad(set_to_none=True)
        logits = model(input_ids=tokens, attention_mask=attention).logits[:, :-1].float()
        reference = F.log_softmax(logits, -1).gather(
            -1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)[:, plen-1:]
        mask = attention[:, plen:].float()
        denom = mask.sum(1, keepdim=True) if algo == 'grpo' else 1024
        loss = -(reference * mask * weight / denom).sum() / 3
        loss.backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        expected = reference.detach()
        model.zero_grad(set_to_none=True)
        actual = []
        for start, stop in ((0, 2), (2, 3)):
            lp, cm = completion_logps(model, tokens[start:stop], attention[start:stop], plen)
            width = lp.shape[1]
            torch.testing.assert_close(lp * cm, expected[start:stop, :width] * cm,
                                       rtol=1e-5, atol=2e-6)
            divisor = cm.sum(1, keepdim=True) if algo == 'grpo' else 1024
            (-(lp * cm * weight[start:stop] / divisor).sum() / 3).backward()
            actual.append(F.pad(lp.detach(), (0, 12-width)))
        for name, p in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(p.grad, gradients[name], rtol=2e-4, atol=2e-6)
    print('EQUIVALENCE_OK: GRPO and Dr.GRPO valid-token logps and parameter gradients', flush=True)


if __name__ == '__main__':
    check_equivalence()
