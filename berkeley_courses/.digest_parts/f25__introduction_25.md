### CS 294/194-196: Agentic AI
*Speaker:* Prof. Dawn Song

#### Key claims / techniques
- LLM agents extend LLMs with environment interaction via reasoning/planning, tool use, retrieval, memory, and action; solving real-world tasks requires trial-and-error, external tools, and external knowledge retrieval.
- Agent workflows facilitate complex tasks through task decomposition, allocation of subtasks to specialized modules, division of labor, and multi-agent generation.
- AI progress is upper-bounded by evaluation ("you can only improve what you can measure"), making rigorous agent evaluation a first-order research problem.
- Open agent evaluation faces three core challenges: lack of standardization (agents need manual tweaks per benchmark), limited openness (no public agents, environments, or clear leaderboards), and low reproducibility (no shared hosting or open setups).
- AgentBeats is proposed as an open platform for agent evaluation and risk assessment, providing standardization (unified SDK + A2A/MCP protocols + consistent workflows), openness (public agents, benchmarks, hosted environments), reproducibility (auto-reset + hosted runs + automatic multi-level trace logging), and ease of use.
- AgentBeats supports two evaluation modes: Benchmark Mode (single-agent, absolute metrics for scoring/ranking) and Arena Mode (multi-agent, adversarial evaluation and competitions).
- In AgentBeats terminology, green agents host/manage assessments, white agents participate in assessments, and an assessment is a multi-agent procedure between one green agent and many white agents reflecting metrics of the white agents.
- The course project is split into an Agent Track (design green agents in Phase 1, then white/competition agents in Phase 2) and a Research Track (novel research toward workshop/conference publication).
- Example green-agent ideas span L1–L3 difficulty and cover coding (SciCode, SWE-bench, USACO), web browsing (Online-Mind2Web, WebShop, BrowserGym), research reproduction (CORE-Bench), QA (GAIA), tool use (τ-bench, AppWorld), computer use (OSWorld), security (Agent CTF, Smart Contract Exploit, OpenAgentSafety), DeFi operations, games (Chess, Werewolf, Minecraft), and embodied/text-world tasks (ALFWorld, TheAgentCompany).

#### Relevance hooks
- Agent evaluation methodology: central to the lecture; introduces a green/white-agent assessment framework, standardization via A2A/MCP, hosted environments, and trace logging for reproducible benchmarking.
- RL reproducibility standards: supported by the emphasis on reproducibility barriers and AgentBeats' auto-reset, hosted runs, and automatic multi-level trace logging.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: f25 | introduction_25.pdf | Prof. Dawn Song | Introductory lecture frames agentic AI and positions rigorous, standardized, reproducible open-agent evaluation as the core research agenda via the AgentBeats green/white-agent platform. | ok
