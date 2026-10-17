"""Observation-conditioned rollout protocol v2.

The managed adapter must preserve prior token prefixes across chat turns. If its
rendering changes prior tokens, fail rather than assign probabilities to another
sequence. Observations and other environment tokens remain masked from loss.
"""

from types import SimpleNamespace


async def interactive_rollout(
    server, tokenizer, messages, execute, *, max_tokens, temperature=1.0, max_rounds=8
):
    messages = [dict(message) for message in messages]
    trace = []
    previous = None
    remaining_tokens = max_tokens
    async with server.managed_server(tokenizer=tokenizer) as managed:
        for _ in range(max_rounds):
            if remaining_tokens <= 0:
                raise ValueError("ReAct generated-token budget exhausted")
            completions = await managed.chat_completion(
                messages=messages,
                n=1,
                max_tokens=remaining_tokens,
                temperature=temperature,
                stop=["\nObservation:"],
            )
            if len(completions.choices) != 1:
                raise ValueError("Expected exactly one ReAct action")
            text = completions.choices[0].message.content
            if text.count("Action:") > 1 or "Observation:" in text:
                raise ValueError("Policy must emit one action without an invented observation")
            node = managed.get_state()["nodes"][-1]
            tokens, masks, logprobs = (
                list(node.tokens),
                list(node.masked_tokens),
                list(node.logprobs),
            )
            if not len(tokens) == len(masks) == len(logprobs):
                raise ValueError("Misaligned managed rollout")
            if previous is not None:
                n = len(previous.tokens)
                if tokens[:n] != previous.tokens:
                    raise ValueError("Managed chat rendering changed prior trajectory tokens")
                masks[:n], logprobs[:n] = previous.masked_tokens, previous.logprobs
            generated = sum(mask != -100 for mask in masks)
            if generated > max_tokens:
                raise ValueError("Backend exceeded the ReAct generated-token budget")
            remaining_tokens = max_tokens - generated
            previous = SimpleNamespace(tokens=tokens, masked_tokens=masks, logprobs=logprobs)
            trace.append(text)
            observation, done = execute("\n".join(trace), text)
            if done:
                return "\n".join(trace), previous
            messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Observation: " + observation},
                ]
            )
            trace.append("Observation: " + observation)
    raise ValueError("ReAct action budget exhausted without a final answer")
