# atropos/tinker_atropos/environments/ — INDEX

**Purpose:** Atropos GRPO environment implementations. Each subclasses `atroposlib` `BaseEnv`, defines a dataset + reward, and runs as `python <env>.py serve --config <yaml>`. Registered in `__init__.py`.

**Key files:**
- `gsm8k_tinker.py` — `GSM8kEnv`: GSM8K grade-school math; reward = symbolic `\boxed{}` verify (math_verify).
- `math_tinker.py` — `MathEnv`: Hendrycks MATH competition problems; same boxed-answer symbolic scoring.
- `math_curriculum_tinker.py` — `MATHCurriculumEnv` (E7): two-phase reward, format-compliance first then answer correctness.
- `tool_use_tinker.py` — `ToolUseEnv`: function-calling (glaive dataset); binary reward on tool name + required args.
- `humaneval_tinker.py` — `HumanEvalEnv`: Python code gen; reward = passes all unit tests via sandboxed subprocess.
- `logp_steering.py` — `LogpSteeringEnv`: on-policy self-distillation using same-model teacher logprobs (WildChat prompts, tinker `/logprobs`).
- `moe_routing_tinker.py` — `MoERoutingEnv` (E6): measures within-group reward variance from sparse MoE routing divergence.
- `bootstrap_threshold_tinker.py` — `BootstrapThresholdEnv` (E5): tests GRPO's non-zero seed-signal requirement via solution-length difficulty bins.
- `multihop_react_tinker.py` — ReAct multi-hop QA agent over a KB (HotpotQA); search/lookup/finish tools.
- `multistep_tool_math_tinker.py` — ReAct tool-use math agent (GSM8K) with calculate/store/recall tools.
- `__init__.py` — exports the 7 primary env classes.

**Find it fast:**
- to add a new task → copy `gsm8k_tinker.py`, change dataset + `score`/reward
- for agentic/multi-turn tasks → `multihop_react_tinker.py`, `multistep_tool_math_tinker.py`
- for distillation-style reward → `logp_steering.py`
