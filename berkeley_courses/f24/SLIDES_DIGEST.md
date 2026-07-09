# F24 Lecture Slides Digest

Auto-generated digest of 11 lecture slide PDF(s).

---

### Towards a unified framework of Neural and Symbolic Decision Making
*Speaker:* Yuandong Tian

#### Key claims / techniques
- Even state-of-the-art LLMs (GPT-4-turbo, o1) struggle with real-world planning benchmarks such as TravelPlanner and NATURAL PLAN.
- The lecture organizes solutions into three directions: scaling laws, hybrid deep-model-plus-solver systems, and emergent symbolic structures learned by neural networks.
- A hybrid travel-planning system, To the Globe (TTG), converts natural language requests into symbolic descriptions and solves them as a Mixed Integer Linear Program (MILP), outperforming pure LLM planning in end-to-end human evaluation.
- Searchformer reformulates A* search as a token-prediction task and trains Transformers on solver traces with teacher forcing; search-augmented models are substantially more parameter- and data-efficient than solution-only models on maze navigation and Sokoban.
- Repeated bootstrapping and fine-tuning improve the Improved Length Ratio (ILR) of search traces while preserving optimal plans, framing plan shortening as a reinforcement-learning task.
- DualFormer switches automatically between fast System-1 and slow System-2 modes and exceeds dedicated single-mode models and o1-preview on math problems.
- SurCo learns linear surrogate objectives for combinatorial nonlinear optimization, enabling gradient-based optimization in a latent space and is applied to embedding-table sharding and inverse photonic design.
- Follow-up work addresses SurCo limitations: Landscape Surrogate (optimization under partial information) and GenCO (diverse solution generation).
- On modular addition, gradient descent learns a Fourier-basis representation rather than a lookup table, and the loss landscape has algebraic (ring-homomorphism) structure.
- Global optimizers for reasoning tasks can be composed from partial optimizers, and empirically most gradient-descent solutions factorize into the constructed order-4/order-6 forms with small error.

#### Relevance hooks
- RL post-training / GRPO: Searchformer trace-shortening is cast as fine-tuning/RL that preserves optimality; DualFormer connects to fast/slow reasoning tradeoffs relevant to post-trained reasoning models.
- Agent evaluation methodology: TravelPlanner and NATURAL PLAN are concrete planning/agent benchmarks; the hybrid LLM-solver pipeline is a design pattern for rigorous agent evaluation.
- RL reproducibility / interpretability: the modular-addition analysis shows gradient descent converging to explicit, factorizable symbolic structures, informing what RL/post-trained models might learn internally.

#### Cited paper titles (verbatim only)
- TravelPlanner: A Benchmark for Real-World Planning with Language Agents
- NATURAL PLAN: Benchmarking LLMs on Natural Language Planning
- Training Compute-Optimal Large Language Models
- To the Globe (TTG): Towards Language-Driven Guaranteed Travel Planning
- Towards Full Delegation: Designing Ideal Agentic Behaviors for Travel Planning
- Beyond A*: Better Planning with Transformers via Search Dynamics Bootstrapping
- Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces
- SurCo: Learning Linear Surrogates For Combinatorial Nonlinear Optimization Problems
- Landscape Surrogate: Learning Decision Losses for Mathematical Optimization Under Partial Information
- GenCO: Generating Diverse Solutions to Design Problems with Combinatorial Nature
- Pre-trained Large Language Models Use Fourier Features to Compute Addition
- Composing Global Optimizers to Reasoning Tasks via Algebraic Objects in Neural Nets

---

### LLM Agents — Enterprise Trends for Generative AI
*Speaker:* Burak Gokturk

#### Key claims / techniques
- Scale (compute, data, model size) has been the dominant driver of ML capability gains, from ImageNet and LibriSpeech to modern foundation models.
- Foundation models are built on transformers + autoregressive next-token prediction pre-trained on trillions of tokens.
- The standard path to useful agents is Supervised Fine-Tuning (SFT) followed by Reinforcement Learning from Human Feedback (RLHF) with a reward model.
- Gemini was designed as a natively multimodal model from the start; the project began February 2023 and had public releases in December 2023 (1.0) and February 2024 (1.5).
- Gemini 1.5 demonstrates long-context retrieval up to 10M tokens, with the claim that context-window information is "clearer" (less perturbed by gradient descent) and therefore reduces hallucination and enables in-context learning.
- A "needle in a haystack" evaluation across text, audio, and video reports >99.7% recall out to 10M tokens.
- Enterprise trends highlighted: AI development is accelerating; separate task-specific models are giving way to single generalizing models; dense models are moving toward efficient sparse models; single-modality models are moving toward multimodal models; API cost is approaching zero; search and LLMs are converging.
- Key production success factors are broad model choice, a managed production platform, ability to customize with proprietary data, and flexibility/avoidance of vendor lock-in.
- Customization toolkit includes fine-tuning, distillation (teacher/student with soft labels and temperature scaling), grounding, and function calling/extensions.
- Parameter-efficient adaptation methods covered include conventional fine-tuning, prompt tuning (Lester et al.), and LoRA, which decomposes weight updates into low-rank matrices and can be applied to attention layers.
- Grounding mitigates hallucinations and stale knowledge via Retrieve-Augment-Generate (RAG), private-document retrieval, fresh web content, and Natural Language Inference (NLI) based post-hoc corroboration with citations.
- Function calling gives developers structured output and external API/tool integration, enabling real-time data retrieval, database queries, and autonomous agent workflows.

#### Relevance hooks
- Agent evaluation methodology: the lecture explicitly discusses grounding, attribution scoring (supporting/contradicting sources), retrieval-augmented generation, and function calling as core components for building reliable enterprise agents.
- RL post-training benchmarking: only indirectly related—RLHF is mentioned as the standard post-SFT tuning stage, but no benchmarking methodology is discussed.
- None directly supported by the extracted text for GRPO/ZVF diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- Gemini: A Family of Highly Capable Multimodal Models
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context

---

### Building a Multimodal Knowledge Assistant
*Speaker:* Jerry Liu

#### Key claims / techniques
- A production knowledge assistant should accept any task (simple/complex questions, research) and return any output form (short answer, structured output, report).
- Naive RAG suffers from poor data processing, weak query understanding/planning, no function calling, and no memory, limiting time savings and decision-making value.
- A production-ready stack requires four capabilities: high-quality multimodal RAG, complex output generation, agentic reasoning over complex inputs, and a scalable full-stack application.
- Data quality is a prerequisite: “garbage in = garbage out”; ETL for LLMs entails parsing, chunking, and indexing into clean structured data.
- LlamaParse is presented as an LLM-native parser for complex documents (embedded tables, charts, images, irregular layouts), producing text chunks, tables, diagrams, and metadata without requiring per-element bounding boxes or exhaustive JSON.
- A true multimodal RAG pipeline parses documents into interleaved text and image chunks, links them via metadata, embeds/indexes text chunks, and feeds both text and retrieved images into a multimodal LLM during synthesis.
- Complex inputs (summarization, comparison, multi-part, research tasks) require agentic capabilities: tool use, query planning, memory, and reflection.
- There is a reliability–expressiveness trade-off: constrained flows (routers, fixed pipelines) are more reliable, while unconstrained agent/orchestrator flows are more expressive but less reliable.
- LlamaIndex Workflows are proposed as an event-driven, composable, code-first alternative to graph-based pipelines, intended to be more readable, maintainable, observable, and production-deployable.
- Multimodal report generation can be implemented via structured outputs (e.g., interleaving `TextBlock` and `ImageBlock`) with separate researcher and writer agent steps.
- Production agent systems require encapsulation/reusability, standardized communication interfaces, scalability, human-in-the-loop support, and debugging/observability tooling.
- `llama-deploy` is offered as a microservices architecture for agentic workflows, using a central message queue, distributed tool execution, and human-in-the-loop as a service.

#### Relevance hooks
- Maps to **agent evaluation methodology**: the lecture explicitly decomposes agentic systems into tool use, planning, memory, reflection, routing, and orchestration, and discusses reliability–expressiveness trade-offs that are central to agent benchmarks.
- Touches on **RL reproducibility standards** indirectly through the emphasis on debugging, observability, standardized interfaces, and human-in-the-loop guardrails for production agent deployments.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.

---

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

---

### Measuring Agent Capabilities and Anthropic’s RSP
*Speaker:* Ben Mann

#### Key claims / techniques
- Anthropic’s Responsible Scaling Policy (RSP) is a public commitment to ensure model capability does not outstrip the ability to build effective guardrails and mitigate harm.
- The RSP is organized around AI Safety Levels (ASL-1 through ASL-4), with Anthropic currently preparing for ASL-3, the “significantly higher risk” tier.
- RSP goals include structuring hard safety decisions, public accountability, learning to iterate on safe decisions, and providing a template for policymakers and industry.
- Standard capability benchmarks are described as not lasting; performance relative to human baselines saturates quickly (Kiela et al., 2023).
- Anthropic emphasizes measuring task-completion time relative to humans, citing a METR August 2024 evaluation finding that Claude 3.5 Sonnet completes work that would take human developers ~30 minutes in seconds.
- Claude 3.5 Sonnet is claimed to outperform OpenAI o1-preview on SWE-bench Verified while being cheaper and faster.
- Similar claims are made on Aider’s code-editing and refactoring benchmarks, including an 18% lead on the refactoring benchmark, which tests producing long code chunks without skipping sections or making mistakes.
- A “computer use” case study is presented as a concrete agentic capability, covering demos, technical implementation, safety considerations, and future implications.
- Practical safety measures include ASL-standard implementation, security measures, deployment safeguards, and year-1 lessons.
- Future priorities include scaling governance, capability measurement, and academic collaboration.

#### Relevance hooks
- Agent evaluation methodology: the lecture discusses SWE-bench Verified, Aider benchmarks, METR task-time evaluation, and the challenge that benchmarks saturate relative to human performance.
- RL reproducibility standards: the RSP framing ties measurement rigor to accountable deployment decisions, though it does not discuss RL training reproducibility per se.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, length bias, or evals-with-error-bars.

#### Cited paper titles (verbatim only)
- Mathematical Framework for Transformer Circuits
- Toy Models of Superposition
- Sleeper Agents: Deceptive LLMs

---

### Agentic AI Frameworks & AutoGen
*Speaker:* Chi Wang

#### Key claims / techniques
- Future AI applications are shifting from pure generative models to compound, agentic AI systems that execute complex tasks on behalf of humans.
- Agentic programming improves response quality through divide-and-conquer, grounding & validation, and natural iteration, while remaining modular and amenable to human participation.
- A useful agentic framework should provide an intuitive unified agentic abstraction, flexible multi-agent orchestration (static/dynamic, NL/PL, centralized/decentralized, cooperation/competition, intervention/automation), and built-in implementations of core design patterns (conversation, prompting/reasoning, tool use, planning, multimodal/memory integration).
- AutoGen is positioned as a programming framework for agentic AI based on conversable & customizable agents plus conversation programming, with support for code/function execution, nested chats, and group chat.
- Concrete AutoGen patterns demonstrated include two-agent reflection for blogpost writing, nested chats for advanced reflection, conversational chess, and group chat for complex task planning and solving.
- The broader AutoGen ecosystem targets lowering the programming barrier (AutoBuild, Composable Actor Platform), making agents smarter (AgentOptimizer, EcoAssistant, Learn to Cooperate), and supporting agent-based evaluation (AgentEval, AutoDefense, Observability).
- Reported adoption metrics include ~30K GitHub stars, ~200K monthly downloads, and usage/contributions across enterprise and academic organizations.
- The framing research questions center on designing optimal multi-agent topology, creating more capable agents, and enabling scale, safety, and human agency while balancing quality, monetary cost, latency, and manual effort.

#### Relevance hooks
- Agent evaluation methodology: the slides explicitly call out agent-based evaluation tools such as AgentEval, AutoDefense, and Observability.

#### Cited paper titles (verbatim only)
- The Shift from Models to Compound AI Systems

---

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

---

### Compound AI Systems & Natural Language Programming with DSPy
*Speaker:* Omar Khattab

#### Key claims / techniques
- Compound AI Systems are modular programs that use language models as specialized components, yielding gains in quality, control, transparency, efficiency, and inference-time scaling (e.g., retrieval-augmented generation, multi-hop RAG, compositional report generation).
- Current compound systems are often “stringly-typed,” coupling architecture with incidental prompt text; DSPy proposes instead to program them with fuzzy natural-language-typed modules that learn their behavior.
- A DSPy program separates concerns into Modules, Signatures, Adapters, Predictors, Metrics/Assertions, and Optimizers.
- Signatures specify the input/output mapping (e.g., `context, question -> query`) rather than the prompt text; modules such as `dspy.ChainOfThought` provide the inference-time strategy.
- Example multi-hop RAG program (`MultiHop`) iteratively generates search queries, retrieves passages, and generates an answer, all expressed as a Python class with DSPy modules.
- Adapters translate signatures into basic prompts; Optimizers then tune them, e.g., raising a HotPotQA multi-hop QA score from 33% to 55% with GPT-3.5, 50% with Llama2-13B, and 39% with T5-770M after `BootstrapFinetune` from 200 answers.
- `BootstrapFewShot` uses rejection sampling to build task demonstrations and can search over demonstration sets (`BootstrapFewShotWithRandomSearch`).
- OPRO is extended from single-prompt optimization to multi-stage programs; Module-Level OPRO updates multiple module prompts in parallel, and grounding the proposer with bootstrapped demos, dataset summaries, program-code summaries, and generation tips improves instruction proposals.
- MIPRO (Multi-prompt Instruction PRoposal Optimizer) bootstraps demonstrations, proposes instruction candidates via an LM program, and jointly optimizes instructions and few-shot demonstration sets with Bayesian optimization using a surrogate model for credit assignment.
- LangProBe benchmark experiments show that bootstrapped demonstrations are critical, instruction optimization helps most on tasks with many conditional rules, and MIPRO (combining both) is often the most effective approach.
- DSPy has been used in production systems and SoTA research systems (PATH, IReRa, STORM, EDEN, Efficient Agents, ECG-Chat) and a U of Toronto MEDIQA competition win.

#### Relevance hooks
- Agent evaluation methodology: the lecture frames multi-hop RAG, tool use, and ReAct-style reasoning as modular LM programs and introduces LangProBe as a benchmark for language-model programs.
- RL reproducibility standards: LangProBe results are reported as averages over 5 runs with Wilcoxon signed-rank significance tests, an example of statistical rigor in optimizer benchmarking.
- None directly supported by the extracted text for GRPO/RL post-training benchmarking, ZVF/zero-variance diagnostics, group-size effects, or length bias.

#### Cited paper titles (verbatim only)
- "DSPY: COMPILING DECLARATIVE LANGUAGE MODEL CALLS INTO SELF-IMPROVING PIPELINES"
- "Large Language Models as Optimizers"
- "STORM: Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models"

---

### A Tale of Two Kittens
*Speaker:* Dr. Jim Fan, NVIDIA Research

#### Key claims / techniques
- Frames embodied AI with the 1963 Held & Hein kitten experiment contrasting passive and active experience, updated with a ChatGPT → Embodied AI narrative.
- Argues the timing for humanoid robotics is driven by falling cost/time curves: examples place NASA Robonaut ($1.5M, 2001) → Unitree G1 ($30K, 2024), alongside Tesla Optimus, Boston Dynamics e-Atlas, and Figure F.02.
- Introduces Project GR00T ("Generalist Robot 0 0 Technology") as an AI brain for humanoid robots.
- Proposes three design principles: Data Pyramid, Foundation Agent, and "The Matrix".
- Data Pyramid layers: Internet Data (EB/day) → Simulation Data (TB/GPU-day) → Real Robot Data (24 hours/robot-day), supported by Omniverse Cloud teleoperation and Isaac Lab.
- Core simulation claim: "It's easier to simulate a problem than to solve it"; training robots in large-scale simulation is viable.
- Describes a generative simulation pipeline: Text-to-3D models (Stable Diffusion), USD generation (ChatUSD), and task/code generation with GPT-4o/Claude-3.5, culminating in the RoboCasa framework.
- Demo amplification chain: 1 human demo → N synthetic demos (RoboCasa) → N×M demos (MimicGen), including bimanual strategies (parallel, coordination, sequential) and DexMimicGen for humanoid tasks.
- Cross-embodiment foundation policy: MetaMorph tokenizes a robot's kinematic tree as a graph of joints and trains a single Transformer from observation to action across varied terrains and robot forms.
- Eureka: a coding-LLM approach that writes reward functions from task descriptions and environment code, then runs massively parallel GPU training with reward candidates, automated feedback, and self-reflection.
- DrEureka extends Eureka by using an LLM to generate sim2real domain-randomization configurations; Eureka++ further generalizes the loop to new tasks and new simulations.
- GR00T stack: OVX for synthetic data/token generation, DGX for foundation model training, AGX for edge deployment, plus HOVER and digital-twin evaluation, orchestrated by OSMO.

#### Relevance hooks
- None directly supported by the extracted text.

#### Cited paper titles (verbatim only)
- RoboCasa: Generative Simulation Framework

---

### Agents for Software Development
*Speaker:* Graham Neubig

#### Key claims / techniques
- Software developers spend most of their time on bugfixing (36%), coding (17%), communication (14%), documents/reviews (10%), testing (8%), and other tasks (15%) (Meyer et al. 2019), suggesting coding agents should support more than just code writing.
- Two complementary support modes exist: synchronous development copilots (GitHub Copilot, Cursor) and autonomous development agents (SWE-Agent, Aider, Devin, OpenHands), with OpenHands-resolver targeting autonomous issue resolution.
- Coding benchmarks span simple Python functions (HumanEval/MBPP), broader StackOverflow-style libraries (CoNaLa/ODEX), data-science notebooks (ARCADE), real-world GitHub issues (SWEBench), and web-to-code generation (Design2Code).
- Pass@K is the standard execution-based metric for code generation; variance is reduced by generating N > K samples with C correct answers and computing the expected value.
- When execution is hard, lexical/semantic overlap metrics are used, including BLEU, CodeBLEU (syntax and semantic flow), and CodeBERTScore (CodeBERT-based BERTScore).
- Dataset leakage is a serious concern: ARCADE shows novel notebooks are harder than online ones, and LiveCodeBench finds some code LMs outperform on existing HumanEval problems while failing on newly collected ones.
- Effective coding agents must understand repository structure, read existing code, modify or produce code, and run/debug it; action-space designs include CodeAct (bash/Jupyter execution), SWE-Agent (specialized repo tools), and OpenHands (event stream with agent skills).
- File localization strategies include offloading to the user, equipping the agent with search tools, a-priori repo maps (Aider repomap, Agentless hierarchical search), and retrieval-augmented code generation.
- Planning and error-recovery approaches range from hard-coded pipelines (Agentless: localize file, localize function, generate patch, apply patch) to LLM-generated plans (CodeR), iterative revisiting (CoAct), and feedback from execution error messages (InterCode).
- Safety risks include accidental destructive actions (e.g., pushing to main, deleting tests to pass) and intentional misuse (hacking); mitigations include sandboxed execution (OpenHands Docker), least-privilege credentialing, and post-hoc action auditing.

#### Relevance hooks
- Strong connection to agent evaluation methodology: compares SWEBench, Pass@K, execution-based vs. overlap metrics, and discusses leakage and benchmark freshness.
- Relevant to evals-with-error-bars: notes that raw Pass@K is high variance and describes the standard N>K correction.
- Touches on RL reproducibility standards through its emphasis on dataset leakage, execution-based evaluation, and carefully constructed code-generation benchmarks.

#### Cited paper titles (verbatim only)
- Today was a Good Day: The Daily Life of Software Developers
- Why Software is Eating the World

---

### Open-Source and Science in the Era of Foundation Models
*Speaker:* Percy Liang

#### Key claims / techniques
- Foundation model capabilities have risen sharply while access has narrowed from full paper/code/data/weights to API-only releases.
- Access level shapes the kind of science possible: API access is like cognitive-science behaviorism, open-weight access like neuroscience probing internals, and open-source access like full systems-level control.
- API-based agents can be composed of tools and verifiers to solve complex problems in ML engineering and cybersecurity, and can simulate social behavior (e.g., realistic interview simulations).
- Open-weight models enable reproducible research on interpretability, fine-tuning, distillation, and model merging, and findings such as adversarial attacks often transfer to API models.
- A permutation-based hypothesis test can assess whether two weight checkpoints were independently trained by permuting hidden units to build a null distribution of weight similarities.
- Empirical lineage checks link models such as Miqu-70B to Llama-2-70B, StripedHyena-Nous-7B to Mistral-7B-v0.1, and Llama-3.1-8B to Llama-3.2-3B.
- Open-source language modeling efforts mentioned include OLMo, OLMoE, RedPajama, DCLM-BASELINE, MAP-Neo, OpenCoder, FineWeb, and SmolLM.
- A workable open-source AI definition can require data information and processing code rather than raw copyrighted data, plus sufficient documentation and compute to retrain.
- Training techniques referenced in the open-source context include distributionally robust optimization, diagonal Hessian with clipping, and precise model editing.
- Strategies to scale open research include building downward-extrapolating scaling laws, pooling idle consumer GPUs for decentralized training, and public funding for shared infrastructure.

#### Relevance hooks
- Agent evaluation methodology: the lecture explicitly contrasts problem-solving agents (ML engineering, cybersecurity) with simulation agents (social-behavior digital twins) and discusses success-rate evaluation over repeated trials.
- RL reproducibility standards: argues that open-weight and open-source access are prerequisites for reproducible mechanistic research, model-derivation studies, and independent lineage verification.

#### Cited paper titles (verbatim only)
- None explicitly listed in extracted text.
