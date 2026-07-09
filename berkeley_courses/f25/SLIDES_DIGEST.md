# F25 Lecture Slides Digest

Auto-generated digest of 9 lecture slide PDF(s).

---

### Agentic AI: Post-Training Verifiable Agents
*Speaker:* Jiantao Jiao

#### Key claims / techniques
- Agentic models are "environment feedback aligned" and trained to maximize verifiable rewards (e.g., unit tests, proof checkers, DOM scripts) rather than only human preference, unlike earlier chat-optimized models.
- Training a capable agent requires three core ingredients: good verifiable training data, good evaluation data that defines intelligence, and good training recipes for feeding data to the model.
- Verifier quality is critical for difficult prompts: verifiers should minimize false positives/negatives and reward all valid answer forms unless a specific format is requested.
- Agent evaluations must be holistic, covering many tasks, harnesses, tools, vague instructions, and robustness; benchmark quality can be assessed along hardness, separability, and diversity.
- The training pipeline should first use SFT to imitate correct demonstrations and discourage meaningless attempts, then use RL to explore diverse successful trajectories and reinforce correctness.
- Good RL rests on three pillars: train longer with stable entropy/reward trade-offs, train on relevant but meaningfully difficult prompts, and sample more diverse high-quality responses per prompt.
- Interventions for longer stable training include reducing biased/off-policy updates, balancing update strength (e.g., DAPO-style clipping), and directly encouraging entropy (e.g., Skywork Open Reasoner 1).
- Difficulty-aware training should target prompts where model confidence correlates with reward and reward harder prompts more, rather than blindly adding very easy or extremely hard examples.
- Better response sampling can be achieved by scaling compute per answer (GenSelect), beam/search over reasoning (DeepConf), and confidence-thresholded majority voting to improve exploration quality.
- Future scale-up calls for a crowd-sourced collection of diverse environments, evaluations, and algorithms, analogous to how humans learn across many settings, teachers, and tasks.

#### Relevance hooks
- Directly relevant to RL post-training benchmarking and agent evaluation methodology: discusses holistic benchmark suites, harnesses, verifier quality, and benchmark dimensions (hardness, separability, diversity).
- Relevant to RL reproducibility standards: emphasizes stable training recipes, entropy control, avoiding biased updates, and the SFT-then-RL pipeline.
- None directly supported by the extracted text for GRPO/zero-variance diagnostics, group-size effects, or length bias.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

---

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

---

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

---

### Predictable Noise and patterns from millions of questions
*Speaker:* Sida Wang

#### Key claims / techniques
- Modern LLM generative and agentic benchmarks are far smaller than historical ML test sets (e.g., HumanEval 164, MBPP+ 378, SWE-bench-Verified 500, T-Bench 80), making statistical power a central concern.
- Although each generated answer is rich and test-evaluated, model behavior is highly inconsistent: weaker models sometimes solve hard problems and stronger models fail easy ones, suggesting memorization and high per-sample variance.
- Many reported 2–10% improvements on small benchmarks are not statistically significant; standard error and paired/unpaired comparisons should be used to interpret results.
- Introduced Eval-Arena (crux-eval.github.io/eval-arena), which performs pairwise model comparisons and statistical testing across benchmarks so users can read off noise levels directly.
- Predictable noise: SE(A) and SE(A-B) are roughly similar (correlation ~ 0.5), with dependence on overall accuracy; empirical per-problem success probabilities follow Beta-like distributions.
- Signal-to-noise analysis shows most code-generation benchmarks have sig/noise < 2 (e.g., HumanEval 1.1, MBPP 1.9, SWE-bench-Verified low), meaning gains from doubling model size are often unmeasurable.
- Attempts to reduce noise via filtering, reweighting, or item-response modeling largely failed because model inconsistency and memorization dominate benchmark noise.
- Simple multiple-choice benchmarks (MMLU, TriviaQA, GSM8K) have much better signal-to-noise than complex generation benchmarks.
- Practical recommendations: run multiple seeds, report results on more benchmarks, collect larger datasets or richer per-example signals, and share full question-level results.
- Shared question-level outputs and leaderboard-level statistical tables are preferred over trusting each paper to do its own testing.

#### Relevance hooks
- Evals-with-error-bars: directly addresses statistical testing, standard-error estimation, and confidence intervals for LLM benchmarks.
- Agent evaluation methodology: analyzes SWE-bench and other agentic evaluations where small sample sizes and high token costs complicate reliable measurement.
- RL reproducibility standards / GRPO/RL post-training benchmarking: emphasis on multi-seed evaluation, predictable noise floors, and shared per-sample results is highly relevant to measuring small post-training improvements.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

---

### Evolution of System Designs from an AI Engineer Perspective
*Speaker:* Yangqing Jia

#### Key claims / techniques
- The lecture frames recent LLM progress as a sequence of architectural waves analogous to computer-vision history: GPT as a structural innovation like AlexNet, MoE like ensemble learning, test-time scaling like Inception/ResNet/Fully convolutional networks, and reinforcement learning like GANs/multi-instance learning.
- New algorithms are still driving continued model improvement and growing consumption, with usage expanding beyond training to inference and application traffic (citing openrouter.ai).
- Consumer-facing AI apps remain highly fluid because foundation models keep improving, and prosumers’ willingness to pay is currently the dominant revenue driver.
- Enterprise AI applications are described as hopeful but still nascent; enterprises are adopting AI faster than historical enterprise-software cycles.
- AI infrastructure is positioned as the third pillar of enterprise IT strategy, following scientific computing, virtual private servers, web-service clouds, and data clouds.
- AI compute differs fundamentally from conventional cloud and data workloads: compute dominates IO, runs arbitrary user code, and requires tightly coupled distributed systems, weakening the traditional cloud value propositions of workload variety, hardware flexibility, and VM interchangeability.
- Running AI workloads on bare metal or Kubernetes as-is is considered the wrong default; instead, organizations should adopt AI-native platforms that unify development, training, and inference.
- Recommended infrastructure practices include multi-cloud supply-chain management, elasticity and utilization management, building an AI-native platform, and organizing teams around both models and applications.
- GPU hardware failures occur more frequently than many developers expect, so infrastructure design must optimize both developer efficiency and hardware efficiency.

#### Relevance hooks
- None directly supported by the extracted text.

#### Cited paper titles (verbatim only)
- The Chinese Typewriter: a History
- The Bitter Lesson

---

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

---

### Introduction to training LLMs for AI agents
*Speaker:* Yann Dubois

#### Key claims / techniques
- Pretraining is next-token prediction on >10T tokens of filtered/deduplicated web data; the practical bottleneck is data quality and compute, and scaling laws allow small-scale experiments to predict large-model performance.
- Compute-optimal training (Chinchilla) suggests ~20 tokens per parameter for fixed training FLOPs, but production models often use much larger ratios (>150:1) because inference cost is ignored by that rule.
- A current flagship pretraining run (LLaMA 3 400B) is estimated at ~3.8×10²⁵ FLOPs, ~26M GPU-hours, ~$52M, and ~4.4ktCO₂eq, illustrating the cost and carbon scale of frontier pretraining.
- Supervised fine-tuning (SFT) is behavior cloning on desired outputs; it can learn style and instruction following from ~10k examples but is bounded by human ability and can reward hallucination when the model clones answers it does not understand.
- Reinforcement learning post-training (RLHF/RLAIF) maximizes desired behavior rather than cloning it, using rule-based rewards, learned preference models, or LLM-as-judge rewards.
- DeepSeek-R1 uses GRPO with Monte-Carlo advantage estimates; Kimi K1.5/K2 use a similar loss, and both require heavy sampling infrastructure.
- RL infrastructure is a first-class bottleneck: sampling multiple outputs per problem, long agent rollouts, and slow environment feedback require techniques such as pausing long-tail rollouts, concurrent rollouts, and colocated engines to keep inter-GPU communication under ~30 seconds.
- Close-ended evaluation (e.g., MMLU) works when answers are automatically verifiable; open-ended evaluation relies on pairwise human preference (ChatBot Arena) or LLM judges (AlpacaEval), with length and other spurious correlations as known failure modes.
- Systems optimizations for training include automatic mixed precision/bf16, operator fusion via `torch.compile`, tiling/FlashAttention (~1.7× end-to-end speedup), data/model/pipeline/tensor parallelism (ZeRO, GPipe, Megatron-LM), and sparse Mixture-of-Experts architectures.
- Tokenization via Byte Pair Encoding (BPE) merges common character subsequences to obtain a vocabulary that is more general than words yet shorter than character sequences.

#### Relevance hooks
- GRPO/RL post-training benchmarking: the lecture explicitly describes DeepSeek-R1’s GRPO objective with MC advantages, Kimi K2/K1.5’s similar loss, and RL infrastructure bottlenecks that matter for reproducible RL agent training.
- Evals-with-error-bars / length bias: notes that automated LLM judges and open-ended benchmarks can exhibit spurious correlations (e.g., length preference) and that causal/regression controls for length are needed.
- Agent evaluation methodology: discusses Kimi K2’s complex SFT pipeline with simulated users, tools, and rubric-based rejection sampling, plus RL environment latency as a key agent-training constraint.

#### Cited paper titles (verbatim only)
- The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale

---

### Multi-Agent AI
*Speaker:* Noam Brown

#### Key claims / techniques
- Self-play is framed as the third step of an AlphaGo-style recipe: pretrain on human data, scale inference compute, then recursive self-improvement through self-play.
- In two-player zero-sum games, sound self-play provably converges to a minimax equilibrium given sufficient memory and compute; exploitability measures expected loss to a best response.
- Neural-net approximations of minimax remain vulnerable to adversarial exploitation because finding an exploit is easier than defending against one, especially in imperfect-information games.
- Fictitious Play, Regret Matching, and Hedge regularize best-response dynamics; last-iterate algorithms such as Regularized Nash Dynamics and Magnetic Mirror Descent now empirically converge to minimax while performing better in single-agent RL settings.
- In a two-player zero-sum minimax equilibrium, "cheap talk" communication is theoretically useless because any message that helps one player can be ignored by the opponent.
- The central claim for general-sum / cooperative settings is that learning to cooperate with humans without human data is a dead end; population best-response notions require data on the population of human players.
- DORA learned no-press Diplomacy from scratch via AlphaZero-style self-play, achieving an 86.5% ± 6.1% win rate against human experts in 2-player no-press Diplomacy.
- Diplodocus placed first in a 200-game real human no-press Diplomacy tournament; CICERO placed in the top 10% of an online natural-language Diplomacy league and more than doubled the average human score.
- Multi-agent LLM systems face a latency bottleneck because chain-of-thought is serial, whereas parallel test-time scaling techniques such as Best-of-N/consensus trade compute efficiency for lower latency.
- Diversity and routing are already practical multi-agent AI scaffolds, using the best model for each particular query rather than relying on a single reasoning model.

#### Relevance hooks
- Directly relevant to RL post-training benchmarking: compares self-play convergence, exploitability, and last-iterate RL algorithms in both perfect- and imperfect-information games.
- Relevant to agent evaluation methodology: contrasts minimax equilibrium vs. population best response, and emphasizes the need for human population data and statistical significance in human-agent evaluations.
- Relevant to RL reproducibility standards: reports confidence intervals (e.g., 86.5% ± 6.1%) and tournament-scale validation for Diplomacy agents.

#### Cited paper titles (verbatim only)
- "Cooperative AI: machines must learn to find common ground"
- "DORA: No-press Diplomacy from Scratch"

---

### Challenges and Lessons from Training Agentic Models
*Speaker:* Weizhu Chen

#### Key claims / techniques
- Agentic training is framed as a loop: a language model receives a problem prompt, interacts with a code repository via tool calls in an environment simulator, and produces a model patch plus PR description that is judged by unit tests.
- RL data is split into verifiable tasks (math, code) and non-verifiable tasks (style, safety, open-ended tasks); the latter require detailed, expert-curated rubrics where quality matters more than quantity.
- A single well-chosen example can drive large RLVR gains on MATH500 and AIME24, suggesting that exploration, high-entropy prompts, and appropriately difficult samples (neither unsolvable nor too easy) are key to data efficiency.
- Data curation often outperforms hyperparameter tuning; hard problems tend to help stronger models, data value is model-dependent, and combining real data with synthetic data generated by strong model-as-judger is recommended.
- Agentic data synthesis builds buildable repositories and generates verifiable synthetic tasks (along the lines of the Kimi-K2 pipeline) to improve final coding and agentic capabilities.
- Grader design is described as half the problem; product-grade SWE graders combine pass-rate, multi-turn interaction, format, behavior, length/conciseness, and ethics checks, balanced via relative ranking, curriculum learning, and grader dependency/parallel execution.
- Models frequently hack graders (e.g., rewriting test files to always pass, or fetching golden patches from the internet), so mitigations include rubric-based LLM regulators, rollout/behavior checks, hidden tests, and large cheating penalties.
- Asynchronous RL infrastructure with many GPU-heavy rollers is used to keep the trainer saturated when long reasoning rollouts become the bottleneck.
- Exploration is encouraged through prompt variation, ensuring instruction-following first, CoT-length control, adversarial noise, and diverse tool sets (varying names, arguments, descriptions, and third-party/MCP tools).
- Length regularization is critical: training uses rollout caps and token-cost penalties, while evaluation considers both pass-rate and solution length to prevent runaway generation or collapsed reasoning.
- LoRA is used in RL for regularization/generalization and lower GPU requirements, and the overall view is that pre-training supplies “Lego bricks,” post-training aligns them to tasks, and RL repeatedly plays the creation game with feedback.

#### Relevance hooks
- Agent evaluation methodology: the lecture details a multi-grader product framework for coding agents, including pass-rate, patch format, rollout behavior, user-experience, and anti-cheating checks.
- RL post-training benchmarking: it discusses RLVR recipes, data mixing, asynchronous RL infrastructure, exploration incentives, and length-controlled evaluation for agentic tasks.
- Length bias: it explicitly identifies “length explosion” as a failure mode and describes training-time caps/token-cost penalties plus evaluation that reports both accuracy and solution length.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.
