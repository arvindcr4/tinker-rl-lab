# AppBench access, licence, deployment, and grading request

**Status update (2026-08-24):** Sent by email on 2026-08-22 and opened as
[Hugging Face discussion #2](https://huggingface.co/datasets/AfterQuery/App-Bench/discussions/2)
on 2026-08-24. No licence, harness, grader access, or score has been received.

**To:** research@afterquery.com
**CC:** support@afterquery.com
**Subject:** AppBench reproducibility request — licence, official deployment harness, and grading protocol

Hello AfterQuery team,

I am running an independent academic evaluation campaign over agentic benchmarks and would like to include AppBench without substituting a locally invented protocol.

We have pinned the six public tasks from `AfterQuery/App-Bench` at Hugging Face revision `de80d5bcd404adee5307311571e512b5c37e6112`. The repository currently exposes the CSV but no dataset card, licence file, official deployment harness, or reproducible grader package. Could you provide or confirm:

1. A written licence or permission grant covering evaluation use of the six tasks and rubrics, plus publication of aggregate results.
2. The official Next.js/Supabase starter, deployment runner, environment image or immutable revision, credentials policy, and reset procedure used for the leaderboard.
3. The artifact and side-effect verification procedure used after deployment.
4. The human-grading protocol: grader qualifications, rubric instructions, number of independent graders, disagreement threshold, and re-adjudication procedure.
5. Whether AfterQuery can provide the two qualified graders as a service, and the price and expected turnaround.
6. Whether the six public tasks are the complete official evaluation suite and whether any contamination or held-out controls apply.

The model is `Qwen/Qwen3.6-35B-A3B` at revision `995ad96eacd98c81ed38be0c5b274b04031597b0`. We will not train on AppBench tasks and will not report a score until the task revision, environment, and grading receipt are bound immutably.

If the official protocol must remain hosted, we are happy to submit generated applications for AfterQuery-run evaluation instead of receiving the grader artifacts.

Thank you,

Arvind CR  
Tinker RL Lab independent research campaign  
arvindcr4@gmail.com
