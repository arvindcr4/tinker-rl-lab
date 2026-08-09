# Pavlov's List task contract for useful post-training

Status: **$18 Tinker cap authorized; other paid providers remain unauthorized**

This addendum changes the unit of post-training from an isolated answer to a
stateful environment episode. GSM8K is retained only as a small calibration
control. It cannot supply primary training coverage, primary evidence, or a
claim that a model is useful to the companies on Pavlov's List.

The machine-readable source of truth is
[`pavlovs_domain_contract.json`](pavlovs_domain_contract.json). Run
`python3 pavlovs_domain_contract.py` from this directory before generating any
launch manifest. The check fails closed unless all 53 companies and all 16
domain tags in the 2026-08-03 Pavlov's List snapshot inherit both a training
suite and a held-out primary evaluation suite.

`python3 build_pavlovs_campaign_manifest.py` prints the deterministic campaign
preview. It never launches a job and remains `BLOCKED` until budget, licensing,
revision, and decontamination receipts are complete.

## What “useful” means here

The campaign targets transferable capabilities similar to the work these
companies build or evaluate: operating tools and browsers, modifying live
state, producing inspectable artifacts, repairing repositories, doing
professional finance/enterprise work, running scientific or ML workflows,
auditing security targets, writing Verilog, and sustaining long-horizon
plans. Passing a domain suite is evidence of task-family usefulness; it is not
evidence that a model meets any individual company's private production bar.
Company-specific acceptance tests remain a deployment requirement.

## Frozen company-to-task-family coverage

| Pavlov domain | Companies represented by the snapshot | Training analogue | Held-out primary test |
|---|---|---|---|
| alignment | dmodel; Trajectory Labs | AgentDojo safety/tool trajectories | AgentHarm |
| browser | BenchFlow; Plato | BrowserGym environments | WebBench |
| chip design | Phinity | RTLCoder Verilog synthesis | VerilogEval |
| code | 23 listed companies | SWE-Gym and terminal/tool tasks | SWE-bench Pro, FrontierSWE, SDAB |
| computer use | Cua; Deeptune; Chakra Labs; Originator; Refresh; Vetto AI | BrowserGym and visual app tasks | WebBench and AppBench |
| design | Verita AI; Taste Labs | visual application construction | AppBench artifact grading |
| enterprise | AfterQuery; Halluminate; Collinear; BenchFlow; Metaphi; Fleet; Akhara; Plato; Theta | browser and office-tool workflows | APEX-Agents, BankerToolBench, SDAB |
| finance | Halluminate; General Reasoning; Dissei | API-Bank RLVR episodes | BankerToolBench and APEX-Agents |
| games | Good Start Labs | Crafter and OpenReward environments | unseen OpenReward game families |
| long horizon | 11 listed companies | multi-step repository, browser, science, and game episodes | APEX-Agents, SDAB, SWE and science suites |
| math | Ulam; Hillclimb | OpenR1 Math RL training | private FrontierMath; MATH-500 remains secondary |
| ML | EdotEnv; Emulated; Vmax; dmodel; Preference Model; Diffuse Labs | OpenReward and Unix-CTF environments | MLE-bench, FrontierSWE, SDAB |
| multi-domain | Handshake; Mercor; Scale; Snorkel; Surge; Pareto; Fleet | mixed OpenReward environments | APEX-Agents |
| science | Latch; ReasonCore; Tacit Labs; Sepal AI | ScienceWorld and OpenReward science | LifeSciBench |
| security | Quesma; ARIMLABS | Unix-CTF and AgentDojo | BinaryAudit and AgentHarm |
| tool use | Pareto; Chakra Labs | BFCL plus stateful browser/environment episodes | BankerToolBench, APEX-Agents, AgentHarm |

The snapshot and mappings come from <https://pavlovslist.com/>. The task design
is anchored in benchmarks that exercise real work rather than prompt-only
accuracy: APEX-Agents uses high-context professional workspaces and client-ready
artifacts (<https://www.mercor.com/blog/introducing-apex-agents/>),
BankerToolBench grades banking work against environment state
(<https://joinhandshake.com/research/ai/gandalf-the-grader/>), and SDAB uses
live production-system work such as CI/CD, migrations, incident response, and
distributed systems (<https://emulated.so/sdab>).

## Model and training contract

The primary candidate is `Qwen/Qwen3.6-35B-A3B`; the replication candidate is
`Qwen/Qwen3.5-9B`. Both are required to retain multimodal input, tool calling,
long context, computer use, and code generation. A text-only math specialist is
out of scope because it cannot cover the computer-use and design portions of
the company set.

Every training batch must cover at least six domain families. Math may be at
most 5% of the mixture, at least 60% of examples must be stateful episodes, and
at least 50% must end in a native artifact or externally visible state change.
Domains are inverse-frequency weighted so the many code companies do not cause
the model to ignore smaller science, finance, hardware, or security families.

The reward is a normalized vector rather than exact-answer accuracy alone:

1. environment task success;
2. artifact or state integrity;
3. rubric-based partial credit;
4. safety and policy compliance;
5. token, tool-call, latency, and failure cost.

Primary verifiers inspect environment state, test results, or native artifacts
whenever correctness depends on them. Hidden behavioral checks remain outside
the policy-visible workspace.

## Evaluation and claim boundary

Results are sliced by domain, horizon, reward type, verifier type, artifact vs.
stateful task, and seen vs. unseen environment family. The primary comparison
unit is a seed within model x environment-family x stack. Aggregate gains may
not hide a regression in any company-inherited domain.

Training task IDs, repositories, seeds, and hidden tests must be disjoint from
primary evaluation. Every run needs pinned dataset revisions, licenses, task-ID
hashes, split-manifest hashes, and container/environment digests.

This contract does **not** claim that training has run, that either model has
improved, that one adapter serves every company without specialization, or that
domain coverage equals production readiness. Those claims require live results
and, ultimately, company-specific acceptance tests.

## Launch gate

Tinker usage is authorized up to $18, with a $16.50 operational cap and $1.50
safety reserve because billing telemetry can lag. Hugging Face Jobs and every
other paid provider remain disabled. Tinker runs still require immutable
train/eval splits and a local dry run. GSM8K cannot be promoted out of its
`calibration_only` role.
