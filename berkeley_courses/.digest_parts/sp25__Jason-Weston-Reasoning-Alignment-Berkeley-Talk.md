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

Index row: sp25 | Jason-Weston-Reasoning-Alignment-Berkeley-Talk.pdf | Jason Weston | Self-rewarding and meta-rewarding LLMs with System-2 reasoning, iterative DPO/GRPO, and trainable LLM-as-a-Judge evaluation improve instruction following and reward modeling. | ok
