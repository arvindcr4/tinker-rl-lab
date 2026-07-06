### LLM Agent Evaluations & Project Overview
*Speaker:* Unknown

#### Key claims / techniques
- Evaluation is defined as systematic, repeatable measurement of models/agents to ground capability progress and risk assessment in reproducible evidence.
- Agents extend static LLMs with planning, tool-use, memory, and multi-step reasoning, so agent evaluation must handle dynamic environments and more complex success criteria.
- Evaluations are categorized as close-ended vs. open-ended, verifiable vs. non-verifiable, and static vs. dynamic; open-ended non-verifiable tasks often require human eval or LLM-as-a-judge.
- A taxonomy of agent evals covers (1) specific agent capabilities, (2) specific application domains, and (3) general sets of applications.
- A good eval system must have strong outcome validity; common failure modes include noisy/biased data, impracticality, gameable shortcuts, and insufficient challenge.
- Case-study benchmarks discussed: CyberGym (cybersecurity PoC reproduction), τ-bench/τ²-bench (tool-agent-user interaction), GDPval (economically valuable knowledge work), CRMArena (CRM workflows), and LegalAgentBench (Chinese legal workflows).
- The "green agent" hosts an assessment by preparing environments, distributing tasks to participant "white agents," verifying outcomes, and reporting metrics; it can be prompt-based or code-based and communicates via A2A/MCP/APIs.
- Course projects come in two flavors: integrating an existing benchmark (e.g., SWE-bench Verified, TerminalBench) with quality analysis and expansion, or building a new benchmark from scratch.
- SWE-bench Verified illustrates benchmark curation: human verification filtered out 68.3% of original samples, leaving 500 tasks, and GPT-4o resolved 33.2% on the verified set versus 16% on the original.
- When adapting a benchmark, key implementation steps are sorting the agent interface (human-solvable yet agent-friendly), designing the kickoff/workflow, implementing green/white agents, and integrating with the AgentBeats platform for reproducibility.

#### Relevance hooks
- Directly supports agent evaluation methodology, including verifiable/non-verifiable tasks, static/dynamic benchmarks, and programmatic vs. judge-based evaluation.
- Touches on RL reproducibility standards through emphasis on repeatable measurement, pass@k reporting for reliability/consistency, and contamination checks.
- Aligns with evals-with-error-bars via reporting pass@1 and pass@k in τ-bench/CRMArena to capture multi-run reliability.

#### Cited paper titles (verbatim only)
- CyberGym: Evaluating AI Agents' Cybersecurity Capabilities with Real-World Vulnerabilities at Scale
- τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains
- τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment
- GDPval: Evaluating AI Model Performance on Real-World Economically Valuable Tasks
- CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments

Index row: f25 | LLM_Agent_Evaluations_&_Project_Overview.pdf | Unknown | Lecture surveys agent-evaluation taxonomy, outcome-validity principles, and benchmark case studies, then maps them to green-agent course projects on AgentBeats. | ok
