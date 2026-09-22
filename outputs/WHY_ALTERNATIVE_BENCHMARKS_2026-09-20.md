# Why we are doing this: alternative benchmarks + jev + experiments

## The situation (last semester → now)

Last semester's campaign scored E1 and E11 exact and left 12 lanes
score-null, each blocked by a named external input (private bundles,
licences, verifiers, credentials, authorizations). The audit rule is
fail-closed: partial, recovery, and readiness evidence never becomes a
suite score, and lane adapters explicitly ban substitutes. That work is
preserved in `outputs/E1_E14_Terminal_Status_2026-08-29.json` and the
2026-09-19 jev triage receipts (`outputs/jev_receipts/`), which classified
every lane's next action with `jev choice`.

## Why alternative benchmarks

Waiting on providers is not a research strategy. The exact suites stay
score-null until providers grant access — but model and harness development
still needs empirical signal on the same capabilities (web navigation, ML
engineering, binary analysis, agentic tool-use, advanced math, safety
refusal). Open proxies supply that signal with zero provider dependence:
public data, usable licences, runnable locally or on Modal today.

## Why these specific benchmarks

Each was picked by a scout that read the lane's adapter and test bodies,
then verified the candidate live (URL + licence, 2026-09-20):
same-capability family, pullable now, licence compatible with our use,
runnable on resources we have. Map:
`outputs/E2_E14_ALTERNATIVE_BENCHMARK_MAP_2026-09-20.json`;
readiness: `outputs/E2_E14_PROXY_READINESS_2026-09-20.json`.
Known licence cautions (EffiBench, KernelBench dataset, FinanceBench NC
terms, Design2Code data) are recorded, not hidden.

## What jev is doing here (connection to last sem)

Last sem jev triaged blockers (`choice` over next-action classes). This sem
jev judges the proxy layer: `jev ask` picks the best-first proxy per lane
and `jev noul` truth-checks proxy-run claims before they enter receipts.
Same CLI, same receipt discipline, new decision surface. Every jev call
lands in `outputs/jev_receipts/` next to last sem's.

## The experiment plan

1. Local-first proxy runs on user-approved lanes (qwen3:8b via ollama):
   Omni-MATH, LAB-Bench, AgentHarm-public, MiniWoB++, KernelBench.
2. Each run records its own proxy receipt under its own benchmark name.
3. Exact lanes stay score-null; E12 exact recovery proceeds via the
   AfterQuery contact draft on user approval.
4. jev validates each receipt claim (`noul`) before it counts as evidence.

## The boundary (unchanged)

No alternative number is, approximates, or bounds an official E-lane score.
If a provider grants access, the exact suite runs and the proxy becomes
development history. That is the whole justification: keep the exact
record honest while keeping the research moving.
