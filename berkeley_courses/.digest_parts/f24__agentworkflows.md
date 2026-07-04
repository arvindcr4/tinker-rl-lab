### Agents for Enterprise Workflows
*Speaker:* Nicolas Chapados, Alexandre Drouin

#### Key claims / techniques
- LLM agents are LLM-powered entities that autonomously plan and take actions across multiple iterations; they contrast with RL agents by leveraging broad world knowledge for zero-shot task solving rather than long sandboxed training runs.
- Two agent modalities are distinguished: API agents (observations/actions via API calls, lower latency and risk, require available APIs) and web agents (observe human-visible pages plus accessibility tree/DOM, act via clicks/text entry, higher capability but higher latency/risk).
- Enterprise automation is framed as a progression from scripted workflows and RPA through conversational workflows to agentic workflows, where an AI agent engages in dynamic, self-reflective, iterative automation.
- TapeAgents is proposed as a holistic framework unifying agent development and optimization: an agent is a resumable modular state machine whose execution is recorded in a structured, granular log called the "tape," which serves as session state, audit trail, prompt-tuning data, and fine-tuning data.
- A cost-effective conversational-agent case study uses a 5-node LLaMA-405B teacher with 19 synthetic user agents to generate tapes, then distills a 1-node LLaMA-8B student that matches GPT-4o performance on the GREADTH metric (Grounded, Responsive, Accurate, Disciplined, Transparent, Helpful) at ~300-330x lower cost.
- Web agents are built by prompting an LLM with a task description, a textual representation of the web page (HTML/DOM/accessibility tree), and an action space; a minimal implementation combines Python + Playwright with a ReAct-style loop.
- WorkArena and WorkArena++ are open-source benchmarks of roughly 600-682 work-related tasks implemented on the ServiceNow platform, spanning basic UI interactions to complex decision-making workflows (scheduling, workload balancing, budget/expense management, offboarding); state-of-the-art models achieve single-digit success rates.
- BrowserGym and AgentLab provide a unified evaluation platform with standardized observation spaces (HTML, screenshot, accessibility tree) and action spaces (bid-based, coordinate-based, Python), grouping major web-agent benchmarks and supporting reproducible experiments.
- Realistic web-agent evaluation favors live-environment benchmarks that assess end results (e.g., database state) over gold-trace comparison, reducing trace memorization and accommodating alternative valid solutions.
- Identified web-agent failure modes include failure to plan, hallucinated controls, and incorrect action syntax; open challenges include long-context understanding, long-term planning, learning/adaptability, multimodality, cost/efficiency, and safety/alignment.

#### Relevance hooks
- Agent evaluation methodology: WorkArena++, BrowserGym, and AgentLab are concrete targets for benchmarking web agents on realistic enterprise workflows and unifying observation/action spaces across benchmarks.
- RL reproducibility standards: the lecture emphasizes reproducibility via standardized observation/action traces, experimental journals (date, versions, configuration, traces), and leaderboards with automatically reproduced scores.
- RL-inspired agent fine-tuning: the slides mention using RL-inspired approaches (e.g., Agent Q with MCTS + DPO) to finetune agent policies, and TapeAgents enables distillation/optimization from teacher tapes.

#### Cited paper titles (verbatim only)
- The economic potential of generative AI: The next productivity frontier
- Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality

Index row: f24 | agentworkflows.pdf | Nicolas Chapados, Alexandre Drouin | ServiceNow guest lecture introducing TapeAgents for agent distillation and WorkArena++/BrowserGym for realistic web-agent evaluation. | ok
