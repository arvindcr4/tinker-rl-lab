import os
import asyncio
from google import genai

async def analyze_part(client, worker_id):
    """Simulates a single subagent analyzing a portion of the repository."""
    prompt = (
        f"You are Subagent {worker_id} out of 45. Your task is to analyze "
        "a subset of the `tinker-rl-lab` repository to map the codebase structure to "
        "the 8 final papers (P1-P8). Provide a very brief summary of your findings."
    )
    try:
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return f"--- Subagent {worker_id} Report ---\n{response.text}\n"
    except Exception as e:
        return f"--- Subagent {worker_id} Failed ---\nError: {str(e)}\n"

async def main():
    # The client automatically picks up the GEMINI_API_KEY from the environment.
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is missing.")
        print("Please set it using: export GEMINI_API_KEY='your_key'")
        return

    print("Initializing Google GenAI SDK Client...")
    client = genai.Client()

    print("Deploying 45 parallel asynchronous subagents...")
    # Launch 45 tasks concurrently
    tasks = [analyze_part(client, i+1) for i in range(45)]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)

    print("\nAll subagents completed. Aggregating results...\n")
    
    # Prompt a final synthesis call to aggregate all 45 reports
    aggregation_prompt = (
        "You are the main agent. Aggregate the following 45 subagent reports "
        "and produce a final markdown table mapping the codebase files to the 8 papers:\n\n" +
        "\n".join(results)
    )
    
    print("Synthesizing final table (this will take a moment)...\n")
    try:
        final_response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=aggregation_prompt
        )
        print("=== FINAL AGGREGATED TABLE ===\n")
        print(final_response.text)
    except Exception as e:
        print(f"Aggregation failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
