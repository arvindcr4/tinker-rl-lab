### Towards Building Safe & Trustworthy AI Agents and A Path for Science- and Evidence-based AI Policy
*Speaker:* Dawn Song

#### Key claims / techniques
- The lecture frames AI risk along three axes: misuse/malicious use, malfunction (bias, loss of control), and systemic risks (privacy, copyright, labor, environmental).
- Training-data privacy leakage is a concrete, measurable failure mode: LLMs can emit personally identifiable information and secrets, and extraction risk worsens as model size increases even with fixed data and training steps.
- Differential privacy (e.g., DP-SGD with gradient clipping and noise) is presented as a principled mitigation against memorization and re-identification attacks.
- Adversarial examples transfer across modalities and into the physical world; safety-aligned LLMs remain vulnerable to adversarial prompts, fine-tuning attacks, and jailbreaks.
- DecodingTrust is introduced as a comprehensive trustworthiness evaluation platform covering eight perspectives and benchmarking both benign and adversarial conditions.
- LLM agent safety expands the attack surface beyond the model itself to memory/RAG poisoning, tool use, prompt injection (direct and indirect), and supply-chain data poisoning.
- Defenses are layered into prompt-level, model-level, and system-level categories, but the slides note that current defenses are often ineffective against adaptive attacks and can degrade performance.
- Representation engineering is proposed as a top-down interpretability approach for reading and controlling model behavior (e.g., mitigating political leaning).
- Secure-by-design/construction and formal verification are advocated as long-term defenses, though they are difficult to apply to non-symbolic neural-network components and hybrid systems.
- Frontier AI is expected to intensify cyber offense more than defense in the near term because of asymmetries in failure tolerance, remediation cost, and deployment velocity.
- A science- and evidence-based AI policy is proposed around five priorities: better risk understanding, transparency, early-warning detection, mitigation/defense, and community trust.
- Marginal risk analysis is recommended as a framework for assessing the incremental impact of foundation models on existing risks and defenses.

#### Relevance hooks
- Agent evaluation methodology: the lecture emphasizes comprehensive trustworthiness evaluation (DecodingTrust), code-agent risk benchmarks (RedCode), and in-lab adversarial testing of agentic systems.
- RL reproducibility standards: the call for a "science of evaluation," transparency reporting, and post-deployment adverse-event reporting aligns with reproducibility and rigorous measurement practices in RL post-training.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- "The Secret Sharer: Measuring Unintended Neural Network Memorization & Extracting Secrets"
- "Extracting Training Data from Large Language Models"
- "Deep Learning with Differential Privacy"
- "LLM-PBE: Assessing Data Privacy in Large Language Models"
- "Explaining and harnessing adversarial examples"
- "Robust Physical-World Attacks on Machine Learning Models"
- "DecodingTrust: Comprehensive Trustworthiness Evaluation Platform for LLMs"
- "Universal and Transferable Adversarial Attacks on Aligned Language Models"
- "Are aligned neural networks adversarially aligned?"
- "Sleeper agents: Training Deceptive LLMs that Persist Through Safety Training"
- "Targeted backdoor attacks on deep learning systems using data poisoning"
- "Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!"
- "Formalizing and benchmarking prompt injection attacks and defenses"
- "AGENTPOISON: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases"
- "RigorLLM: Resilient Guardrails for Large Language Models against Undesired Content"
- "Hidden Persuaders: LLMs' Political Leaning and Their Influence on Voters"
- "GamePad: A Learning Environment For Theorem Proving"
- "LLM Agents can autonomously hack websites"
- "LLM Agents can Autonomously Exploit One-day Vulnerabilities"
- "On the Societal Impact of Open Foundation Models"
- "RedCode: Risky Code Execution and Generation Benchmark for Code Agents"

Index row: f24 | dawn-agent-safety.pdf | Dawn Song | Lecture surveys trustworthy AI and agent safety risks, then advocates a science-based AI policy built on evaluation transparency, adversarial testing, and marginal risk analysis. | ok
