# SP25 Lecture Slides Digest

Auto-generated digest of 13 lecture slide PDF(s).

---

### Coding Agents and AI for Vulnerability Detection
*Speaker:* Charles Sutton

#### Key claims / techniques
- LLM agents are defined as multi-turn LLMs with tool use, emphasizing dynamic computation time, external information, hypothesis testing, and action; planning, chain-of-thought, and multi-agent are de-emphasized.
- Coding-agent evaluation evolved from HumanEval/MBPP/pass@k to SWE-Bench and SWE-Bench Verified; evaluations drive model/agent design but all have a shelf life due to leakage, ceiling effects, and noise.
- SWE-Agent uses a ReAct loop with a carefully designed "agent-computer interface" of information-gathering, acting, and feedback tools to balance model capability with environment control.
- Agentless replaces dynamic LLM control with procedural Python control flow (localization → repair → validation), which can avoid tool-use errors and keep trajectories from going "off the rails."
- AutoCodeRover sits between Agentless and SWE-Agent, using procedural control with separate trajectories/phases and search tools, allowing some exploration while maintaining structure.
- The broader design space spans tool choice, control flow (dynamic, state machines, tree search, recursive/multi-agent), prompting, and acting/sandboxing; the best design point shifts as base models improve.
- CTF competitions (NYU CtF Bench, InterCode-CtF) provide security-focused LLM-agent benchmarks; EnIGMA adapts a ReAct coding agent with command-line, decompiler, debugger, and pwntools-style tools.
- Vulnerability-detection datasets include BigVul, CVEfixes, CrossVul, DiverseVul, and PrimeVul; core challenges are input scope, tangled commits, correctness judgment, and vague CVE descriptions.
- Big Sleep uses a code browser, Python interpreter, and debugger as tools so the LLM can navigate code, hypothesize vulnerabilities in natural language, and verify them by execution; it achieved strong gains on CyberSecEval 2 and found a real SQLite variant-analysis bug.
- Security applications remain wide open; agentic techniques are natural but require larger-scale system understanding and a move from CTF-style tasks to real-world security work.

#### Relevance hooks
- Agent evaluation methodology: compares HumanEval, SWE-Bench, SWE-Bench Verified, CTF benchmarks, and CyberSecEval 2, and discusses data leakage, overfitting, and noisy tests.
- RL reproducibility standards: emphasizes eval shelf life, leaderboard overfitting, and the need for verified, less-noisy benchmarks.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, or length bias.

#### Cited paper titles (verbatim only)
- Program Synthesis with Large Language Models
- Evaluating Large Language Models Trained on Code
- SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
- ReAct: Synergizing Reasoning and Acting in Language Models
- Agentless: Demystifying LLM-based Software Engineering Agents
- AutoCodeRover: Autonomous Program Improvement
- Evaluating Agent-based Program Repair at Google
- RepairAgent: An Autonomous, LLM-Based Agent for Program Repair
- EnIGMA: Enhanced Interactive Generative Model Agent for CTF Challenges
- NYU CTF Bench: A Scalable Open-Source Benchmark Dataset for Evaluating LLMs in Offensive Security
- The Mayhem Cyber Reasoning System
- Unleashing Mayhem on Binary Code

---

### Learning to Self-Improve & Reason with LLMs
*Speaker:* Jason Weston

#### Key claims / techniques
- **Self-Rewarding Language Models**: A single LLM can be trained both to follow instructions and to judge the quality of its own outputs, then iterate data creation/curation and preference training (DPO) without human reward labels.
- **Recipe for self-improvement**: Start from a pretrained LLaMA-2-70B; multitask fine-tune on seed instruction-following (IFT) and evaluation-following (EFT) data; then repeatedly generate K candidate responses, score them with LLM-as-a-Judge, form DPO preference pairs from best-vs-worst, and retrain.
- **Evaluation results**: After two self-rewarding iterations the model nearly matches GPT-4-0314 on AlpacaEval 2.0 and improves on both instruction-following and reward-modeling (OpenAssistant validation) metrics.
- **System 2 reasoning over System 1**: Deliberative multi-call approaches (Chain-of-Verification, System 2 Attention, Branch-Solve-Merge) are introduced to mitigate hallucination, sycophancy, and spurious correlations from soft attention and LM objectives.
- **Iterative Reasoning Preference Optimization**: For reasoning tasks, generate multiple CoTs + answers, build preference pairs from correct vs. incorrect final answers, and train DPO + NLL; the NLL term is important to avoid assigning similar probability to chosen and rejected generations.
- **Thought Preference Optimization (TPO)**: Extends CoT training to general instruction-following tasks, yielding gains on AlpacaEval (3rd place) and ArenaHard (best 8B model) after multiple iterations of CoT optimization.
- **Meta-Rewarding**: The model acts, judges, and meta-judges its own judgments via LLM-as-a-Meta-Judge; meta-judgments create preference pairs to train both action and evaluation, and outperform Self-Rewarding even with the same length-control method.
- **Length control (LC)**: When choosing the DPO preferred response, select the shorter candidate when two high-scoring responses are close, providing a simple response-length mitigation.
- **EvalPlanner**: A “Thinking-LLM-as-a-Judge” trained to generate planning and reasoning CoTs for evaluation; synthetic good/bad response pairs convert evaluation into a verifiable task, giving strong RewardBench and newer benchmark performance with a Llama 3.1 70B base.
- **Verifiable rewards as a unifying driver**: Extracting reward after a final answer enables iterative self-training for both reasoning (IRPO / DeepSeek / o1-style CoT) and evaluation (EvalPlanner), and better judges can in turn improve non-verifiable tasks.

#### Relevance hooks
- **GRPO / RL post-training benchmarking**: The slides explicitly mention 2025 work applying “RL (GRPO - Group Relative Policy Optimization)” as part of the reasoning-training timeline.
- **Length bias**: The Meta-Rewarding work introduces a new LC method that selects the shorter DPO chosen response when two good responses have similar scores, directly addressing response-length bias in preference optimization.
- **Agent evaluation methodology / LLM-as-a-Judge**: EvalPlanner trains evaluation-specific CoTs and reaches SOTA on RewardBench among LLM-as-a-Judge models, tying into automated reward-model and agent-evaluation methodology.

#### Cited paper titles (verbatim only)
- Chain-of Verification Reduces Hallucination in Large Language Models
- Branch-Solve-Merge for Evaluating and Improving Language Generation
- Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback
- Llama 2: Open Foundation and Fine-Tuned Chat Models
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- AlpacaEval: An Automatic Evaluator of Instruction-following Models
- Thinking LLMs: General Instruction Following with Thought Generation
- Self-Rewarding LLMs
- Meta-Rewarding LLMs

---

### From Perception to Action: Multimodal Agent
*Speaker:* Caiming Xiong

#### Key claims / techniques
- **OSWorld** is presented as the first scalable, real computer environment for benchmarking multimodal agents, covering 369 real-world tasks across web/desktop apps, OS file I/O, and multi-app workflows, with execution-based evaluation rather than static demos.
- OSWorld tasks are defined by a JSON task config that specifies the initial VM state setup and an execution-based evaluation script (e.g., `compare_table` against a gold file), enabling interactive, reproducible agent evaluation.
- Agents observe via screenshots, accessibility trees, set-of-marks, and custom streams; they act through a PyAutoGUI-style keyboard/mouse action space within a repeated interaction loop.
- Baseline results show LLMs/VLMs remain far below human performance on real computer tasks; higher screenshot resolution and longer text-based trajectory history improve agents, while screenshot-only is weaker but viewed as the long-term target configuration.
- **Agenttrek** synthesizes agent trajectories by guiding replay with web tutorials, converting freely available tutorial text into realistic GUI trajectories to bypass expensive human annotation.
- **TACO** trains multimodal action models using synthetic **Chains-of-Thought-and-Action (CoTA)**; CoTA fine-tuning outperforms few-shot prompting, and CoTA data quality matters more than quantity, yielding average gains of 1–4% and up to 15% on MMVet.
- **Aguvis** is a unified pure-vision GUI agent that operates across web, mobile, and desktop using only visual observations, trained in two stages: 1M+ grounding examples followed by 35K multi-step trajectories with explicit inner monologue.
- Inner monologue in Aguvis is crucial for both high-level reasoning and low-level action grounding, and enables cross-platform generalization despite training only on web and mobile data.
- **xGen-MM-Vid (BLIP-3-Video)** compresses videos to 32–128 tokens via a temporal encoder, allowing the model to scale to more frames and improve long-video understanding efficiency.
- **GenS** is a generative frame sampler built on a long-context VideoLLM that predicts relevant frame spans with confidence scores; it is trained on GenS-Video-150K (150K videos, ~647s average) and fine-tuned on Aria, improving long-video QA and temporal grounding.

#### Relevance hooks
- **Agent evaluation methodology:** OSWorld provides a real-VM, execution-based benchmark with configurable initial states and evaluation scripts, relevant for evaluating computer-use agents.
- **RL reproducibility standards:** Agenttrek is positioned as a trajectory source for moving from imitation learning (SFT) to reinforcement learning in environment (SFT→RL).
- None of the other research targets (GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, length bias, evals-with-error-bars) are directly supported by the extracted text.

#### Cited paper titles (verbatim only)
- SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering
- World of Bits: An Open-Domain Platform for Web-Based Agents
- Mind2Web: Towards a Generalist Agent for the Web
- WebArena: A Realistic Web Environment for Building Autonomous Agents
- Browsergym: a Gym Environment for Web Task Automation
- On the Effects of Data Scale on Computer Control Agents
- OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
- Agenttrek: agent trajectory synthesis via guiding replay with web tutorials
- 🌮TACO: Multi-modal Action Models with Synthetic Chains-of-Thought-and-Action (CoTA)
- Aguvis: Uniﬁed Pure Vision Agents for Autonomous GUI Interaction
- BLIP-3-Video: You Only Need 32 Tokens to Represent a Video Even in VLMs

---

### Open Training Recipes for Reasoning in Language Models
*Speaker:* Hanna Hajishirzi

#### Key claims / techniques
- Open, reproducible, fully-open language-model ecosystems (OLMo, Tülu) are prerequisites for accelerating the science of LMs, rather than relying on closed APIs.
- A modern LLM post-training pipeline has four stages: supervised instruction tuning (SFT), preference tuning, RL with verifiable rewards (RLVR), and verifier/reward-model components.
- Data quality and targeted data mixing dominate post-training performance; persona-driven synthetic data is especially useful for scaling reasoning, math, code, and precise-instruction-following skills.
- Adding more persona-driven synthetic math data consistently improves MATH performance, while adding grade-school math helps GSM8K; self-consistency voting lets the authors prune ~40% of synthetic CoT data with no loss.
- PPO generally outperforms DPO by roughly 1%, but at higher implementation complexity, memory cost, and throughput cost; DPO’s cheapness makes it more practical for development.
- Scaling reward models does not always translate to better downstream models, and using in-domain prompts during preference tuning can yield further gains.
- Tülu 3 uses length-normalized DPO after experimenting with SimPO; preference tuning primarily improves style/chat while continuing SFT capability gains with smaller absolute magnitude.
- RL with verifiable rewards uses binary correctness signals (e.g., ground-truth answers for GSM8K, MATH, IFEval) instead of neural reward models, avoiding over-optimization and working “out of the box” with PPO.
- OLMo 2 chains multiple RLVR stages and is reported on par or better than Llama 3 and Qwen 2.5, rivaling DeepSeek and GPT-4o.
- Simple test-time scaling (s1) shows that a 1,000-sample high-quality, hard, diverse reasoning dataset plus inference-time budget forcing (e.g., appending “Wait”) can match or exceed o1-level MATH performance.
- Two-stage base-model training—pre-training on trillions of tokens followed by mid-training on ~50B high-quality in-domain/synthetic tokens—is presented as an efficient way to upgrade base-model quality.

#### Relevance hooks
- RL post-training benchmarking: the talk directly compares PPO, DPO, SimPO, and length-normalized DPO, and reports RLVR training curves and final results across GSM8K, MATH, IFEval, and BBH.
- Length bias: the discussion of length-normalized DPO and the decision to avoid SimPO signals attention to response-length and reward-hacking effects in preference optimization.
- RL reproducibility standards: the OLMo/Tülu recipes are framed as fully open, documented, and reproducible alternatives to closed models, emphasizing decontamination, license checks, and public data/evaluation ingredients.

#### Cited paper titles (verbatim only)
- Tülu 1
- Tülu 3
- OLMo
- Direct Preference Optimization (DPO)
- Proximal Policy Optimization (PPO)
- SimPO
- s1: Simple test-time scaling

---

### AlphaProof: When RL meets Formal Maths
*Speaker:* Thomas Hubert

#### Key claims / techniques
- AlphaProof applies AlphaZero-style search and reinforcement learning inside the Lean proof assistant: the action space is Lean tactics, states are Lean proof states, and the reward signal is exact formal verification of a finished proof.
- The training pipeline has four stages: (1) an auto-formalisation model that translates natural-language problems into Lean; (2) supervised pre-training on Mathlib (≈100k definitions, 200k theorems, 300k lines of proof) to learn a strong tactic prior; (3) AlphaZero RL on generated formal problems with Lean-based verification; and (4) test-time RL on variants of a target problem to produce a specialist checkpoint.
- The core bet is that formal mathematics is the right target for scalable RL because it provides both an in-silico exploration environment and a perfect, verifiable reward signal, even though the corpus is smaller than informal math.
- At IMO 2024 AlphaProof solved problems P1, P2 and P6 in algebra/number theory, while AlphaGeometry solved P4; the combined system reached a silver-medal score, missing the gold threshold by one point.
- For "determine the answer" IMO problems, AlphaProof ran in "hard mode": it generated O(100) candidate answers with Gemini, filtered out easily disprovable ones, and attempted to prove or disprove the rest using test-time RL.
- The system fully solved P6, described as one of the hardest IMO problems in the last ten years (only 5 of 609 human participants solved it fully), including a non-obvious construction praised by Timothy Gowers.
- The talk frames superhuman Alpha systems as arising from scaled-up trial-and-error, a grounded feedback signal, search, and curriculum; AlphaProof satisfies the first three, while curriculum generation for mathematics remains an open question.
- Key remaining challenges include Mathlib gaps (especially geometry and combinatorics), orders-of-magnitude more compute than human contestants, and the difficulty of creative theory building and "interestingness" in proofs.

#### Relevance hooks
- Formal theorem proving in Lean offers a perfect, verifiable reward signal, making it a strong setting for RL post-training and for benchmarking reasoning methods such as GRPO-style training on verified proofs.
- The IMO 2024 protocol (formalised problem inputs, human judging, partial scoring, and compute/time budgets) provides a concrete example of rigorous agent-evaluation methodology for mathematical reasoning.
- The test-time RL / specialist-training step, which generates variants of a hard target problem and re-trains, is a form of post-training adaptation whose transfer effectiveness can be benchmarked.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

---

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

---

### Inference-Time Techniques for LLM Reasoning
*Speaker:* Xinyun Chen

#### Key claims / techniques
- Chain-of-thought (CoT) prompting increases reasoning performance by eliciting step-by-step intermediate reasoning; CoT gains scale strongly with model size and enable strategies such as decomposition and planning.
- Zero-shot CoT instructions like “Let’s think step by step” can trigger CoT generation without labeled exemplars, but still lag behind few-shot CoT.
- Analogical prompting instructs the model to self-generate relevant exemplars and high-level knowledge for each problem, improving over both zero-shot CoT and manually designed few-shot CoT.
- LLM-driven prompt engineering—generating candidate prompts, scoring them on a small validation set, and iteratively optimizing instructions—can match or exceed few-shot CoT accuracy.
- Task-specific decomposition methods (least-to-most prompting, Self-Discover) improve compositional generalization by breaking complex problems into subproblems and composing reasoning structures.
- Self-consistency improves over single-sample decoding by sampling multiple reasoning paths and selecting the most common final answer; performance scales with sample count and benefits from diverse sampling.
- Consistency-based selection extends beyond math reasoning to code generation (AlphaCode execution-based clustering) and open-ended generation (Universal Self-Consistency), though the latter depends on long-context capability.
- Trained LLM verifiers can outperform consistency-based selection; process-supervised reward models (PRM) scale better with more samples than outcome-supervised reward models (ORM), but verifier quality and task transfer are limiting factors.
- Tree-of-thoughts and related search methods integrate step-level state evaluation with tree search (BFS, MCTS) to explore partial solution spaces and scale inference compute more effectively than single-path CoT.
- Iterative self-improvement and reflection can help when reliable external feedback is available (e.g., code execution, task-specific heuristics), but without oracle feedback LLMs often fail to self-correct and can degrade performance; multi-agent debate does not outperform self-consistency without a good evaluator.
- The optimal allocation of inference-time compute—parallel vs. sequential sampling, model size vs. number of samples—is model- and task-specific and empirically determined.

#### Relevance hooks
- Closely related to RL post-training benchmarking: the lecture explicitly contrasts outcome-supervised vs. process-supervised reward models and discusses verifier quality and scaling, which are central to RL/GRPO-style post-training evaluation.
- Relevant to evals-with-error-bars: self-consistency, verifier-based selection, and search results depend heavily on sample budget, sampling diversity, and selection method, so reported benchmark numbers should be interpreted with those design choices in mind.
- Relevant to agent evaluation methodology: tree-of-thoughts, self-reflection, self-debugging, and multi-agent debate are core inference-time agent workflows; the lecture cautions that their benefits require reliable external evaluation or strong self-evaluation.

#### Cited paper titles (verbatim only)
- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- Show Your Work: Scratchpads for Intermediate Computation with Language Models
- Emergent Abilities of Large Language Models
- Large Language Models are Zero-Shot Reasoners
- Large Language Models as Analogical Reasoners
- How to solve it
- Large Language Models are Human-Level Prompt Engineers
- Large Language Models as Optimizers
- Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
- Compositional Semantic Parsing with Large Language Models
- Measuring Compositional Generalization: A Comprehensive Method on Realistic Data
- SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures
- Self-Consistency Improves Chain of Thought Reasoning in Language Models
- Competition-level Code Generation with AlphaCode
- Universal Self-Consistency for Large Language Model Generation
- Training Verifiers to Solve Math Word Problems
- Let’s Verify Step by Step
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models
- Reflexion: Language Agents with Verbal Reinforcement Learning
- Self-Refine: Iterative Refinement with Self-Feedback
- Teaching Large Language Models to Self-Debug
- Language Models can Solve Computer Tasks
- Large Language Models Cannot Self-Correct Reasoning Yet
- Improving Factuality and Reasoning in Language Models through Multiagent Debate
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for LLM Problem-Solving

---

### On Memory, Reasoning, and Planning of Language Agents
*Speaker:* Yu Su

#### Key claims / techniques
- Defines language agents as contemporary AI agents with integrated LLMs that use language as a vehicle for reasoning and communication, contrasting an "LLM-first" scaffolding view with an "Agent-first" view that retains classic agent challenges (perception, world models, planning).
- Argues that token generation serves as a unified mechanism for perception, intuitive inference, and symbolic reasoning in LLMs, with self-reflection as a meta-reasoning action operating over an internal "inner monologue" environment.
- Presents HippoRAG, a neurobiologically inspired non-parametric long-term memory for LLMs modeled on hippocampal indexing theory; it partitions memory into neocortex (perception/language/reasoning), parahippocampus (working-memory bridge), and hippocampus (indexing/auto-associative memory) to improve pattern separation and completion over dense RAG.
- Notes that parametric continual learning in LLMs suffers from catastrophic forgetting and ripple effects, making structured non-parametric memory (e.g., HippoRAG, GraphRAG) a promising direction for sensemaking and multi-hop associativity.
- Presents Grokked Transformers as a mechanistic study of implicit (non-CoT) reasoning: standard GPT-2-scale transformers can learn implicit compositional reasoning, but only through a grokking phase transition from rote memorization to generalization.
- Finds that systematic generalization varies by reasoning type and depends more on data distribution than on raw scale; identifies parallel versus staged generalizing circuits via logit lens and causal tracing, and improves systematicity with cross-layer parameter sharing.
- Proposes WebDreamer, a model-based planner for web agents that uses LLM-predicted state transitions to simulate action outcomes before execution, avoiding irreversible real-world interactions and costly tree search.
- Reports that model-based planning (WebDreamer) is more accurate than reactive planning and more efficient than tree search on VisualWebArena.
- Frames modern language-agent planning as facing open-ended action spaces, fuzzy natural-language goals, and difficult or non-binary goal tests, illustrated by web agents (Mind2Web/SeeAct/UGround) and travel planning (TravelPlanner).
- Identifies open questions: reliable rewards for o1/R1-style reasoning, better world models, balancing reactive vs. model-based planning, endogenous vs. exogenous safety risks, and applications such as agentic search and science agents.

#### Relevance hooks
- Agent evaluation methodology: discusses web-agent benchmarks (Mind2Web, VisualWebArena, TravelPlanner) and contrasts reactive, tree-search, and model-based planning paradigms.
- RL post-training / GRPO relevance: hypothesizes that o1/R1-style long CoT emerges when a capable base model already knows basic reasoning constructs and RL learns to combine them, raising the question of how to obtain reliable rewards for such reasoning.
- None directly supported by the extracted text for ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"
- "Grokking of Implicit Relations in Transformers: A Mechanistic Journey to the Edge of Generalization"
- "Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization"
- "Is Your LLM Secretly a World Model of the Internet? Model-based Planning for Web Agents"
- "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency"
- "LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks"
- "Mind2Web: Towards a Generalist Agent for the Web"
- "TravelPlanner: A Benchmark for Real-World Planning with Language Agents"
- "Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts"
- "Why Does New Knowledge Create Messy Ripple Effects in LLMs?"
- "Human-level control through deep reinforcement learning."
- "In Search of Memory: The Emergence of a New Science of Mind"
- "Language Agents: Foundations, Prospects, and Risks"

---

### CS 294/194-280: Advanced Large Language Model Agents
*Speaker:* Prof. Dawn Song

#### Key claims / techniques
- LLM agents extend LLMs by coupling them with an environment through reasoning & planning, tool use, retrieval, memory, and action.
- Solving real-world tasks with agents is typically a trial-and-error process.
- External tools and retrieval from external knowledge expand LLM capabilities beyond parametric knowledge.
- Agent workflows facilitate complex tasks via task decomposition, allocation of subtasks to specialized modules, division of labor, and multi-agent generation.
- LLM agents have transformed applications including code generation (Cursor, GitHub Copilot, Devin, Google Jules), personal assistants (Google Astra, OpenAI GPT-4o), computer use (Anthropic Claude, Google Jarvis, OpenAI Operator), and robotics (Figure AI, Tesla Optimus, NVIDIA GR00T), with relevance to education, law, finance, healthcare, and cybersecurity.
- Reasoning models have progressed rapidly: OpenAI o1 (Sep 2024), Gemini 2.0 Flash Thinking and OpenAI o3 (Dec 2024), DeepSeek-R1 and Kimi k1.5 (Jan 2025).
- Google DeepMind's AlphaProof and AlphaGeometry 2 achieved silver-medal performance at IMO 2024.
- OpenAI o3 is ranked in the top 200 in Codeforces competitive programming contests.
- The course will cover fundamental reasoning techniques, inference-time techniques, training techniques, search and planning, code generation and verification, autoformalization and theorem proving, agentic workflows, and safety/ethics.

#### Relevance hooks
- The hackathon track on creating innovative AI agent evaluation benchmarks and the course focus on agentic workflows connect directly to agent evaluation methodology research.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

---

### Formal Reasoning Meets LLMs: Towards AI for Mathematics and Verification
*Speaker:* Kaiyu Yang

#### Key claims / techniques
- Math and coding serve as proxies for complex reasoning and planning and are relatively easy to evaluate: math answers can be checked and code can be run through unit tests.
- State-of-the-art math LLMs are built from strong pretrained models plus supervised finetuning (SFT), reinforcement learning (RL), and substantial engineering.
- SFT on mathematical data uses problems paired with human- and LLM-curated step-by-step or tool-integrated solutions (e.g., using `sympy`); the largest public datasets contain roughly 900K examples.
- RL on verifiable math problems compares final answers against ground truth and optimizes rewards; GRPO is highlighted as the RL algorithm behind DeepSeek-R1.
- RL’s verifiability requirement limits it to problems with numeric or otherwise checkable answers and does not directly apply to open-ended proofs.
- Two major gaps are identified: moving from pre-college math (AIME/IMO) to advanced mathematical research, and moving from answer guessing to valid proof generation.
- Formal mathematical reasoning grounds reasoning in formal systems (first/higher-order logic, dependent type theory, programs with formal specifications) so proof assistants can verify proofs automatically and provide feedback, mitigating data scarcity.
- Proof assistants such as Lean allow theorems and proofs to be represented formally; AlphaProof is cited as an example of large-scale search and RL using Lean feedback.
- LLMs can be trained for theorem proving to generate either next tactics or complete proofs, with individual tactics assembled into full proofs via search algorithms.
- LeanDojo provides open-source training data, model checkpoints, and interaction tools for Lean, containing 98,641 theorems, 217,639 tactics, and 129,162 premises.
- ReProver (Retrieval-Augmented Prover) retrieves accessible premises, concatenates them with the current proof state, and uses the combined context for tactic generation.
- LIPS (LLM-based Inequality Prover with Symbolic Reasoning) synergizes neural and symbolic reasoning for Olympiad inequalities, categorizing proof steps as scaling (applying lemmas such as Cauchy-Schwarz or AM-GM) or rewriting (equivalent transformations), and prunes scaling tactics symbolically.
- LIPS solves 16/20 Olympiad-level inequality problems, exceeding IMO gold medalists (15/20) and DeepSeek-R1 (4/20), and discovers novel proof paths using AM-GM that human experts considered hopeless.
- Autoformalization faces two core challenges: evaluating whether an informal theorem was formalized correctly is hard (equivalence checking is infeasible, human evaluation is expensive, proxy metrics are inaccurate), and informal proofs contain explicit and implicit reasoning gaps that formal proofs must fill.
- LeanEuclid is introduced as a benchmark for autoformalizing Euclidean geometry, with 48 problems from Euclid’s Elements and 125 from UniGeo, including the first faithful formalization of Euclid’s proofs; diagrammatic reasoning gaps are addressed via a formal system E implemented in Lean and automated with SMT solvers.

#### Relevance hooks
- GRPO/RL post-training benchmarking: the lecture explicitly describes GRPO as the RL algorithm used to optimize verifiable math rewards, popularized by DeepSeek-R1, and contrasts this with the need for formal proof verification when numeric rewards are unavailable.
- Evals-with-error-bars / agent evaluation methodology: formal proof assistants provide automatic, rigorous verification of reasoning steps, offering a pathway for more trustworthy evaluation of LLM reasoning than answer-key matching alone.
- RL reproducibility standards: LeanDojo is presented as an open-source resource releasing data, model checkpoints, and interaction tools to support reproducible research on neural theorem proving.

#### Cited paper titles (verbatim only)
- "Can AI do maths yet? Thoughts from a mathematician"
- "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad"
- "Formal Mathematical Reasoning: A New Frontier in AI"
- "AI achieves silver-medal standard solving International Mathematical Olympiad problems"
- "TacticToe: Learning to Prove with Tactics"
- "GamePad: A Learning Environment for Theorem Proving"
- "Learning to Prove Theorems via Interacting with Proof Assistants"
- "HOList: An Environment for Machine Learning of Higher-Order Theorem Proving"
- "Generative Language Modeling for Automated Theorem Proving"
- "HyperTree Proof Search for Neural Theorem Proving"
- "Proof Artifact Co-training for Theorem Proving with Language Models"
- "LeanDojo: Theorem Proving in Lean using Language Models"
- "Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning"
- "Autoformalization with Large Language Models"
- "A formal system for Euclid's Elements"

---

### Multimodal Autonomous AI Agents
*Speaker:* Russ Salakhutdinov

#### Key claims / techniques
- **VisualWebArena** is introduced as a benchmark of realistic, visually grounded web tasks for multimodal agents, moving beyond text-and-HTML WebArena tasks.
- Raw HTML is insufficient for web agents: it is often minified/compressed, interactive elements render incorrectly, spatial layout is lost, and pages can exceed 100k tokens.
- Multimodal agents use a **VLM + Set-of-Marks (SoM)** representation over interactable elements to simplify cluttered accessibility trees.
- Agent architecture decomposes into high-level planning/reasoning, observation parsing, and low-level action generation (click, type, hover, stop).
- Agent performance suffers from **exponential error compounding**: even 99% single-step accuracy falls to ~73% at 30 steps and ~60.5% at 50 steps.
- A **best-first tree search** method improves agents by combining a baseline policy, environment backtracking, and a prompted multimodal LLM (GPT-4o) as a value function to score intermediate states.
- Search yields clear gains on both VisualWebArena and WebArena for Llama-3-70B and GPT-4o baselines; ablations show both the policy and value function have substantial headroom.
- **InSTA (Internet-Scale Training For Agents)** proposes a scalable pipeline where Llama generates synthetic agentic tasks from ~150k live web domains, then verifies task success automatically.
- Synthetic Llama-generated data outperforms human demonstrations on step accuracy (+89.5% on Mind2Web, +122.1% on WebLINX) and generalization (+149% WebLINX, +156.3% Mind2Web).
- **Plan-Seq-Learn (PSL)** extends the same planning/parsing/learning architecture to long-horizon robotic manipulation, using structured language plans and a single RL policy trained with local observations.

#### Relevance hooks
- Agent evaluation methodology: VisualWebArena defines execution-based, visually grounded web-agent benchmarks with realistic POMDP tasks.
- RL reproducibility / scaling: InSTA automates large-scale synthetic task generation and verification, and PSL trains RL policies for long-horizon robotics from language plans.
- Inference-time scaling: tree search demonstrates how test-time compute (value-guided backtracking) mitigates error accumulation in sequential agents.

#### Cited paper titles (verbatim only)
- REACT Synergizing Reasoning and Acting in Language Models
- Chain of Thought Prompting Elicits Reasoning in Large Language Models
- WebGPT: Browser-assisted Question-Answering with Human Feedback
- MIND2WEB: Towards a Generalist Agent for the Web
- Toolformer: Language Models can Teach Themselves to Use Tools
- ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings
- SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
- WebArena: A Realistic Web Environment for Building Autonomous Agents
- Autonomous Evaluation and Refinement of Digital Agents
- VisualWebArena -- Evaluating Multimodal Agents on Realistic Visual Web Tasks
- Tree Search for Language Model Agents
- Towards Internet-Scale Training For Agents
- Plan-Seq-Learn (PSL): Language Model Guided RL for Solving Long Horizon Robotics

---

### Abstraction and Discovery with Large Language Model Agents
*Speaker:* Swarat Chaudhuri

#### Key claims / techniques
- LLM agents for discovery should systematically search spaces of hypotheses, conjectures, and proofs; use prior knowledge to prioritize search; learn from experience; and discover reusable abstract concepts/tools.
- Frontier tool-integrated math LLMs can solve competition-style problems with Python/SymPy, but the neural-only approach faces data scarcity and lack of verifiability beyond high-school/competition settings.
- Formal representations (e.g., Lean/Coq) provide a verifiable alternative path: informal statements are autoformalized, then a neural prover searches for a formal proof checked by the proof assistant.
- Copra is an in-context-learning agent for formal theorem proving that prompts a frontier LLM, parses tactics, executes them in the proof environment, augments the prompt with error/goal feedback, backtracks, and queries a lemma database.
- Copra can integrate informal natural-language reasoning with formal reasoning hierarchically (e.g., ask for an informal solution, split into subgoals, then solve each subgoal formally), and can repair informal LLM hallucinations using proof-assistant feedback.
- AlphaProof uses reinforcement learning for theorem proving, learning from both successful proofs and failed disproofs, including misformalized problems, and complements training-time RL with test-time RL on problem variants.
- LaSR extends symbolic regression with a learned concept library: an LLM-driven evolutionary loop performs mutation, crossover, initialization, and abstraction over a population of programmatic hypotheses, producing symbolic abstractions such as "power-law trend" or "exponential growth/decay."
- LaSR outperforms PySR even when using local language models and limited LLM budget, and user-provided or LLM-generated concepts can accelerate hypothesis search, though concepts are not guaranteed to be factual or correct.
- LLM scaling laws can be discovered from benchmark data by treating law discovery as symbolic regression; the recovered laws reveal interactions between training hyperparameters and test-time shots.
- A self-evolving visual concept library uses VLMs to score textual concept descriptors contrastively against images, then refines the descriptors through evolution to enable zero-shot visual classification.

#### Relevance hooks
- Agent evaluation methodology: Copra, LaSR, and the visual-concept library are concrete agent systems evaluated against proof assistants, symbolic-regression benchmarks, and zero-shot classification tasks, respectively, using external, task-specific feedback rather than human judgment alone.
- RL post-training benchmarking: AlphaProof is presented as a reinforcement-learning system trained on verifiable proof/disproof signals and test-time search, making formal theorem proving a relevant domain for RL post-training and reasoning benchmarks.
- RL reproducibility standards: The emphasis on formal proof assistants provides deterministic, checkable correctness criteria, which is a model for reproducible evaluation of RL-trained reasoning agents.

#### Cited paper titles (verbatim only)
- How NuminaMath Won the 1st AIMO Progress Prize
- Formal Mathematical Reasoning: A New Frontier in AI.
- An In-Context Learning Agent for Formal Theorem-Proving
- LaSR: Symbolic Regression with a Learned Concept Library
- Training Compute-Optimal Large Language Models
- Self-Evolving Visual Concept Library Using Vision-Language Critics
- Alphageometry
- Proofwala
- Process-Driven Autoformalization
- Minimo

---

### Bridging Informal and Formal Mathematical Reasoning with AI
*Speaker:* Sean Welleck

#### Key claims / techniques
- Formal mathematics treats proofs as source code that compiles iff correct, enabling collaboration, trust, and instant feedback; informal math is flexible but hard to check.
- The informal-formal gap is a core bottleneck: informal intuitions require detailed specification and deep formal-system knowledge to express.
- Lean-STaR trains a 7B model to generate informal "thoughts" before each formal proof step via expert-iteration-style reinforcement learning, improving miniF2F test pass rates over direct tactic generation.
- Increasing search budget is more effective when the model interleaves thoughts with tactics, suggesting thoughts add useful computational capacity and diversify search.
- Draft, Sketch, Prove (DSP) is a three-stage pipeline: (1) LLM drafts an informal proof, (2) LLM translates it into a formal sketch, (3) a low-level prover (Sledgehammer) fills the gaps.
- LeanHammer brings Sledgehammer-style automation to Lean by combining a neural premise selector, external automated theorem provers (e.g., Zipperposition), and the Aesop tree-search tactic.
- LeanHammer's premise selector frames retrieval as contrastive learning over (state, positive premises, negative premises) using a transformer encoder; even sub-100M-parameter retrievers substantially raise proof rates on held-out Mathlib theorems.
- miniCTX benchmarks neural theorem provers on real Lean projects ("future Mathlib", PFR, PrimeNumberTheorem) with long, cross-file context, exposing a gap between competition-problem performance and research-level formalization.
- Premise selection and file-tuning on preceding in-file/cross-file context improve handling of realistic project dependencies; the resulting model is deployed in the LLMLean tool.
- Open-source artifacts are emphasized: models, data, extraction tools, and evaluation code are released to lower the accessibility gap for research-level formalization.

#### Relevance hooks
- Lean-STaR is an instance of RL post-training for reasoning: it uses expert iteration to bootstrap a thought-generating policy, with implications for how reward signals and search can improve theorem-proving agents.
- miniCTX advances agent evaluation methodology by testing provers on real, long-context formal-mathematics projects rather than self-contained competition benchmarks, making it relevant to robust eval design for reasoning agents.
- The talk's emphasis on releasing data, models, and evaluation pipelines connects to RL reproducibility standards for reasoning benchmarks.

#### Cited paper titles (verbatim only)
- Lean-STaR: Learning to Interleave Thinking and Proving
- STaR: Bootstrapping Reasoning with Reasoning
- Draft, Sketch, Prove: Guiding Formal Theorem Provers with Informal Proofs
- Premise Selection for a Lean Hammer
- miniCTX: Neural Theorem Proving with (Long-)Contexts
