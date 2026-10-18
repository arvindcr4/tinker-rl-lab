import asyncio


async def analyze_part(client, worker_id, evidence=None):
    if (
        not evidence
        or not isinstance(evidence, dict)
        or any(not text.strip() for text in evidence.values())
    ):
        raise ValueError("Explicit nonempty file partitions are required before any provider call")
    """Simulates a single subagent analyzing a portion of the repository."""
    prompt = (
        f"You are Subagent {worker_id} out of 45. Your task is to analyze "
        "a subset of the `tinker-rl-lab` repository to map the codebase structure to "
        "the 8 final papers (P1-P8). Cite exact file/function names; mark unsupported mappings unknown.\n"
        + str(evidence)
    )
    try:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash", contents=prompt, config={"max_output_tokens": 1024}
        )
        return f"--- Subagent {worker_id} Report ---\n{response.text}\n"
    except Exception as e:
        return f"--- Subagent {worker_id} Failed ---\nError: {str(e)}\n"


async def main():
    raise SystemExit(
        "This legacy 45-call sweep is disabled. Supply reviewed source partitions to analyze_part; provider budget and bounded orchestration require an explicit caller."
    )


if __name__ == "__main__":
    asyncio.run(main())
