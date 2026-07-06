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

Index row: sp25 | Multimodal_Agent_caiming.pdf | Caiming Xiong | Multimodal agents need real executable environments (OSWorld), scalable trajectory/data synthesis (Agenttrek/TACO), and unified vision-based reasoning/grounding (Aguvis) plus efficient video memory (BLIP-3-Video/GenS). | ok
