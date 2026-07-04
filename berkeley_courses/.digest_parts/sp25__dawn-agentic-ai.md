### Advanced LLM Agents: Towards Building Safe & Secure Agentic AI
*Speaker:* Dawn Song

#### Key claims / techniques
- Agentic systems are hybrid/compound systems that combine symbolic and neural components; this architecture increases attack surface relative to traditional systems.
- Security goals for agentic hybrid systems include confidentiality (API keys, prompts, user data, model parameters), integrity (model integrity, untrusted/poisoned inputs), and availability (DoS, service availability); the safety goal is to avoid harmful consequences.
- A five-level model security taxonomy is introduced: L0 perfect model; L1 accurate but vulnerable; L2 inaccurate and vulnerable; L3 poisoned model (backdoors from malicious samples, RAG, knowledge base); L4 malicious model.
- LLM-generated outputs can serve as multiple stages of an attack chain: external output (information leakage), parameters for further computation (compounding bias/errors), branch/jump conditions (unexpected behavior), function-call parameters (SQL injection, SSRF), and code snippets (arbitrary code execution).
- Concrete vulnerabilities include SQL injection via LLM (CVE-2024-23751 in llama_index, CVE-2024-7764 in vanna-ai), remote code execution via LLM (CVE-2024-21552 in SuperAGI), direct and indirect prompt injection, and backdoor/poisoning attacks on RAG/knowledge bases.
- Prompt injection methods span heuristic approaches (concatenation, escape characters, context ignoring, fake completion) and optimization-based approaches (white-box gradient-guided search, black-box genetic algorithms and RL search).
- AgentXploit is a fuzzing-based end-to-end red-teaming framework for black-box agents; it uses seed attack instructions, mutations, adaptive scoring, and MCTS-based seed selection, and reports roughly 2× attack success rate over handcrafted baselines on AgentDojo and VWA-adv.
- Defense principles are defense-in-depth, least privilege / privilege separation, and safe-by-design / secure-by-design / provably secure design.
- Eight layered defense mechanisms are enumerated: model hardening, input-sanitization guardrails, policy enforcement on actions, privilege management, privilege separation, monitoring and anomaly detection, information flow tracking, and secure-by-design formal verification.
- Progent is a programmable privilege-control system for LLM agents that uses a domain-specific language for policies, a modular enforcement framework requiring only ~10 lines of code change, dynamic policy updates, and hybrid human-written plus LLM-generated policies; it reduces attack success rate while maintaining utility on AgentDojo and ASB benchmarks.

#### Relevance hooks
- Agent evaluation methodology: distinguishes stand-alone LLM evaluation from end-to-end agentic hybrid-system evaluation, and surveys multiple evaluation platforms/benchmarks (DecodingTrust, MMDT, RedCode, AgentXploit).
- Agent evaluation methodology: AgentXploit reports attack success rates and ablations on AgentDojo and VWA-adv, providing a concrete red-teaming benchmark for black-box agents.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- Formalizing and benchmarking prompt injection attacks and defenses
- AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases
- DecodingTrust: Comprehensive Trustworthiness Evaluation Platform for LLMs
- MMDT: Decoding the Trustworthiness and Safety of Multimodal Foundation Models
- RedCode: Risky Code Execution and Generation Benchmark for Code Agents
- AgentXploit: End-to-End Red-teaming of Black-Box AI Agents
- The Protection of Information in Computer Systems
- Progent: Programmable Privilege Control for LLM Agents
- Privtrans: Automatically Partitioning Programs for Privilege Separation
- DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks

Index row: sp25 | dawn-agentic-ai.pdf | Dawn Song | Survey of agentic AI safety/security covering attack taxonomy, evaluation benchmarks, and layered defenses including Progent privilege control | ok
