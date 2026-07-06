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

Index row: f24 | 102824-yuandongtian.pdf | Yuandong Tian | Neural networks can learn interpretable symbolic/algorithmic structures and be combined with combinatorial solvers for decision-making | ok
