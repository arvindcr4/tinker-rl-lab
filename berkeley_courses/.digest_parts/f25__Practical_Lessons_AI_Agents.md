### Practical Lessons from Deploying Real-World AI Agents
*Speaker:* Clay Bavor

#### Key claims / techniques
- Real-world agent deployment is much harder than building demos: the core challenge is production readiness at scale with consistent, high-quality behavior.
- The "Agent Iceberg" shows that visible components (LLM, RAG, tool use) hide a large production stack including guardrails, observability, regression testing, failover, voice pipelines, PII handling, and experimentation frameworks.
- Voice agents use a three-stage architecture: Speech-To-Text (transcribe), LLM Response (reason and respond), and Text-To-Speech (synthesize).
- Word Error Rate (WER) is a poor metric for transcription quality in voice agents; audio issues include multiple speakers, background noise, and far-field speech.
- Voice synthesis challenges include natural entity synthesis (addresses, phone numbers, names), prosody maintenance, phrase quality, and emotive range matching conversation context.
- τ-Bench evaluates agents in realistic, holistic scenarios with complex databases/APIs, domain policies, an LLM-based user simulator, and a dual-control environment where both user and agent act via tools.
- τ-Bench uses objective, rule-based evaluation comparing final database state to ground truth and reports a Pass^k metric measuring the probability that all k independent trials succeed.
- The agent industry is shifting from "agents as technology" to "agents as product," and from transactional task resolution to long-term relationship-building across multiple channels.
- Sierra's Agent Data Platform spans agent memory, customer data, intelligent decisioning, and proactive engagement, with build-once-deploy-everywhere across phone, email, and chat.
- Optimization at Sierra uses AI to improve AI, with components for insights, explorer, and expert answers layered over retrieval, recollection, and guardrails.

#### Relevance hooks
- Agent evaluation methodology: τ-Bench provides a concrete framework for realistic, policy-constrained, multi-turn agent evaluation with grounded success checks.
- Evals-with-error-bars / reliability measurement: Pass^k explicitly tests reliability under conversational variability, which is more meaningful than single-trial accuracy for production agents.
- RL reproducibility standards: τ-Bench leaderboard and τ2-bench repo aim for transparency and community contribution, supporting benchmark reproducibility.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

Index row: f25 | Practical_Lessons_AI_Agents.pdf | Clay Bavor | Real-world agent deployment requires production-grade infrastructure, realistic evaluation via τ-Bench, and reliability metrics like pass^k. | ok
