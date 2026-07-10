import asyncio
from google.antigravity import Agent, LocalAgentConfig, types

async def main():
    # Configure the agent with Gemini 3.5 Flash and enable subagents
    config = LocalAgentConfig(
        model="gemini-3.5-flash",
        capabilities=types.CapabilitiesConfig(
            enable_subagents=True,
        )
    )

    print("Initializing Google Antigravity Agent...")
    async with Agent(config) as agent:
        prompt = (
            "Deploy 45 subagents in parallel to analyze the files in this repository "
            "(`tinker-rl-lab`). Your task is to study how each file or major directory "
            "contributes to the final 8 papers (P1 through P8 found in `sem 4 work/papers/`). "
            "Once all 45 subagents have completed their analysis, aggregate their findings "
            "and create a detailed markdown table mapping the core files to the final 8 papers."
        )
        
        print("Instructing the agent to deploy 45 subagents... This will take a while.")
        response = await agent.chat(prompt)
        
        print("\n=== Final Analysis Output ===\n")
        print(await response.text())

if __name__ == "__main__":
    asyncio.run(main())
