# ai-scientist-v2-integration/ai_scientist/treesearch/ — INDEX

**Purpose:** Patched BFTS (best-first tree-search) agent driver from AI-Scientist-v2.

**Key files:**
- `parallel_agent.py` — large (~112KB) parallel agent: ProcessPoolExecutor rollouts, Journal/Node search tree, LLM query backend, code exec + metric parsing. Patch adds `tinker` to the agent's list of available packages (so it uses the Tinker SDK) and wires optional codecarbon emissions tracking.

**Find it fast:**
- to change which packages the agent believes are available → search `tinker` in `parallel_agent.py`
