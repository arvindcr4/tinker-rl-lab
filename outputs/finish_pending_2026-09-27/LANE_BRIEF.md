# Shared brief for pending-lane completion (2026-09-27)

Repo: /Users/arvind/Developer/agentic_repos/tinker-rl-lab
Authorization: outputs/finish_pending_2026-09-27/AUTHORIZATION.json. The lead (repo owner) issued
"finish all pending lanes". This receipt satisfies every "lead-only launch", "bound authorization
receipt", "sealed reservation" and "owner IAM binding" gate in the older finish-era docs. Do not wait
for any further human seal; reference this receipt in your lane's records instead.

Ledger context: outputs/PES_Phase2_Review_2026-09-12/finish/Pending_Experiments.md,
outputs/UNBLOCK_CARRYOUT_2026-09-21.md, outputs/E1_E14_FINAL_RESULTS_2026-09-19.md.

## Actor (shared, already deployed and warm — DO NOT stop or redeploy it)
- Runbook: outputs/e1_e14_small_scale_2026-09-26/TRAINED_ACTOR_ENDPOINT.md
- Load creds: `set -a; source outputs/e1_e14_small_scale_2026-09-26/trained_actor/.env.local; set +a`
  (TRAINED_ACTOR_BASE_URL, TRAINED_ACTOR_API_KEY). Never print or commit the key.
- Model id `pavlov-public-portfolio-bf16` (seed809 merged, the campaign actor). Temperature/template per
  the lane's own recorded protocol; default non-thinking (`chat_template_kwargs.enable_thinking=false`).
- Shared by 6 lanes on one H200: keep your concurrency <= 4. It scales to zero after 300 s idle; the next
  request cold-starts (3-5 min) — just retry with a long timeout.

## Credentials available
Modal CLI authed; GCP gcloud authed as project owner (project electric-armor-388216, billing on, 3000 vCPU
quota in us-central1); W&B key in ~/.netrc works (entity arvindcr4-pes-university; `wandb verify` CLI is
flaky, ignore it); TINKER_API_KEY in repo .env; OPENAI_API_KEY and HF_TOKEN in env. AWS EC2 quota is 1 vCPU
in both regions (unusable) — use GCP or Modal instead and record the deviation.

## Rules
- Spend cap for your lane is in AUTHORIZATION.json caps_usd. Track spend in your lane's receipt. Stop at the cap.
- Delete every cloud resource you create (VMs, disks, Modal apps other than the shared actor) before you finish,
  and verify absence.
- Evidence rule: replacement-scope numbers are never pooled with original-contract numbers. Errors/timeouts
  count as failures in the denominator. Use the benchmark's native grader.
- Reuse existing tracked code (zvf-program/<lane>/, zvf-program/flagship/) where it works; if the fail-closed
  machinery blocks on paperwork only, bypass it with a thin runner and record why. Do not fabricate results.
- Write everything under outputs/finish_pending_2026-09-27/<LANE>/ : code you add, raw outputs, grader logs,
  result.json (benchmark, scope, n_attempted, n_graded, metric, score, CI, spend_usd, deviations, caveats),
  README.md (short). Don't edit other lanes' dirs, the thesis, or shared ledgers. Don't git commit.
- WARNING: never commit secrets; vendored benchmark repos with hidden tests/answer keys stay out of git
  (add a .gitignore in your lane dir).
- If a lane is genuinely impossible to finish (e.g. dataset unobtainable), get as far as possible, record the
  exact blocker, and return. Don't loop on a failing approach more than ~3 times.
- Final message: <=15 lines: status, score, n, spend, resources cleaned, blockers.

## Resume addendum (2026-09-27 ~01:05 UTC)
The first lead session (68378039) exited at ~00:52 UTC and its six lane agents died mid-setup (no lane had
graded anything). Lanes are relaunched from lead session tinker-rl-lab-de. Each dead agent's full transcript is at
~/.claude/projects/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/68378039-aa3d-4f67-ab65-8b7c3f15b171/subagents/agent-<id>.jsonl
(JSONL; skim the tool_use inputs and tool results near the end to recover what was learned — don't redo work).
Its scratchpad (/private/tmp/claude-501/-Users-arvind-Developer-agentic-repos-tinker-rl-lab/68378039-aa3d-4f67-ab65-8b7c3f15b171/scratchpad/)
still exists and may be reused. Live resources at relaunch: GCP VMs e6-webarena-0927 and e9-mldevbench-0927
(us-central1-a, both RUNNING, owned by E6/E9 respectively). Spend already incurred by those VMs counts toward
their lane caps. This brief file was found truncated to 0 bytes at relaunch — never write to it.
