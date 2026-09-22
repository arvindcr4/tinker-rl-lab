import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/arvind/Developer/tinker-rl-lab";
const WORK = path.join(ROOT, ".codex-pptx/e1-e14-status-2026-08-22");
const STARTER = path.join(WORK, "source-progress/template-starter.pptx");
const STARTER_INSPECT = path.join(WORK, "source-progress/template-starter.pptx.inspect.ndjson");
const RENDER_DIR = path.join(WORK, "rendered");
const FINAL_LAYOUT_DIR = path.join(WORK, "final-layout");
const OUTPUT = path.join(ROOT, "outputs/Tinker_RL_Progress_Update_2026-08-22.pptx");
const CAMPAIGN = path.join(ROOT, "outputs/modal_e1_e14/2026-08-16");

const anchors = [
  ["sh/7ahw3mhs", "sh/698vah07", "sh/l8ze1czm", "sh/k7qd87i1", "sh/54jedwz6", "sh/43axkbyl", "sh/7idojm18", "sh/8jmpsr2t", "sh/yt47mhkb", "sh/judof21w", "sh/lgv6hcj2", "sh/ipo7y10v", "sh/3qx8r61g"],
  ["sh/lk3i9cji", "sh/kju1072x", "sh/be103ilg", "sh/xgji5sjm", "sh/u9cj6hkz", "sh/0v2xcvy9", "sh/1wbyl0fu", "sh/zu9gjqx4", "sh/q50f6lg7", "sh/r69wfqhs", "sh/43yx4byh", "sh/50juhcrq", "sh/ix8v6x8z", "sh/hwzudsre", "sh/wvqd4nqt", "sh/ut8v2x83", "sh/elkjupkj", "sh/zmt0nul4", "sh/0nm1wz2p", "sh/qxkjqpkn", "sh/ryt0jals", "sh/cz21sf2d", "sh/at4j29kr"],
  ["sh/cfqd8zmt", "sh/dgzeh4ne", "sh/2pov2p4r", "sh/4r6d4fmx", "sh/fmdc7a50", "sh/b6xo7690", "sh/q5ony18f", "sh/c7q50bql", "sh/zahobq9w", "sh/y98n2l8r", "sh/0bq54vqh", "sh/fed4zm9s", "sh/ed4nqh87", "sh/jul87utg", "sh/4vu90za1", "sh/5w3q94bm", "sh/v6l83ut4", "sh/w7u9wzap", "sh/x83q5kba", "sh/n218zat8"],
  ["sh/943i5c3e", "sh/o3u1w7mt", "sh/b6507ml4", "sh/ra10vil0", "sh/q9sz2d4f", "sh/4je90bmd", "sh/3il8r6l8", "sh/cne94bmp", "sh/donaxg3a", "sh/sra9srml", "sh/ts3qlwnq", "sh/9wby9obi", "sh/m90fy9sr", "sh/x0bydobe", "sh/wz2x4ja9", "sh/54vyh8bq", "sh/k3mh83a5", "sh/29wv258n", "sh/3a5cva98", "sh/pc7ux0re", "sh/repwza94", "sh/dg7e1kra", "sh/z2pw3u9g", "sh/mxcvup0z", "sh/90nupkjq", "sh/8zedwfi5", "sh/a1wvyp0v"],
  ["sh/gvuhkfmt", "sh/hw3ytk3y", "sh/65sfe54b", "sh/87axgfmh", "sh/y18fal4f", "sh/j2hgjql0", "sh/jq1wzuxo", "sh/5cje1kfe", "sh/7elw3uxk", "sh/6dcvapgz", "sh/tg3e54fq", "sh/nihwrahg", "sh/qdgz6d0j", "sh/bepgfyho", "sh/cfyh83i9", "sh/29wz2t07", "sh/obeh43id", "sh/pc7yd8zy", "sh/ulwzyt0b", "sh/hcvatgna", "sh/vadsr65k", "sh/u94byloz", "sh/98bapgne", "sh/7mtsn658", "sh/lkbalwn2", "sh/kj29sr6x", "sh/0ba9cb2h", "sh/e9sra1kb", "sh/ozutgb2t", "sh/903a9gjy", "sh/mdsbelkn", "sh/4jq94r29", "sh/7ypc3ulc", "sh/6xwvap4r", "sh/t07e543i", "sh/ju5czalg", "sh/lwnu1k36", "sh/kved8fm1", "sh/b65cval4", "sh/e1cve5or", "sh/03udgf6h", "sh/143ep072", "sh/m5cvi5on", "sh/orexkf6t", "sh/atwfmpoz", "sh/bu5wvu54", "sh/50rq18vi", "sh/jy98zids", "sh/t4bq5sbu", "sh/s32pcnu9", "sh/r2983ito", "sh/1sbq9sbq", "sh/ojy9gb6l", "sh/pk7a9g76", "sh/2hwre1of", "sh/0ve9cbq9", "sh/etwra18j", "sh/fu5836po", "sh/sru98rqd", "sh/vmlsbm9k", "sh/ho3adwrq"],
  ["sh/gzyx0vmp", "sh/107e903a", "sh/21gf254v", "sh/q50f65kr", "sh/b69gfalc", "sh/7epo3y5s", "sh/8fypw36d", "sh/zapozy5w", "sh/y9g7qtob", "sh/j698bi50", "sh/i5072x4f", "sh/edojqlcz", "sh/1gf2l0bq", "sh/2hojulsb", "sh/nixknqtg", "sh/p4f2p0bm", "sh/3md0b6tc", "sh/kzalozu9", "sh/yxs3mpwj", "sh/83a5sjul", "sh/61s3q9cf", "sh/o7qlgfe1", "sh/do76xsfa", "sh/r25ovix4"],
  ["sh/n21wr6ls", "sh/m18vil47", "sh/x8jex03u", "sh/v61wvqlo", "sh/a5svml43", "sh/kvuxsv25", "sh/nah0nuhk", "sh/9cjipkzq", "sh/ve10ruhg", "sh/hgjit4zm", "sh/bix0fahc", "sh/el4fu1wj", "sh/gnmxwbep", "sh/6hkfqhwn", "sh/sj2xsret", "sh/bexgvqxc", "sh/qdofmlwr", "sh/8zytgnu9", "sh/907upsvu", "sh/mxgbedc3", "sh/wnidknu5"],
  ["sh/lkj21ozq", "sh/0ja183y5", "sh/nm1k3yhw", "sh/3qh0rehs", "sh/2p8jy907", "sh/wnehwbat", "sh/vm5g36t8", "sh/kreh0bap", "sh/5s7i9gba", "sh/0vuhoral", "sh/lw3yxwb6", "sh/hgb65ozi", "sh/udkna9g7", "sh/p4v69ozu", "sh/43m5gjy9", "sh/21knetg3"],
  ["sh/knalk32p", "sh/lojmto3a", "sh/mps3mtkf", "sh/8ba5o32l", "sh/vel4zilc", "sh/rapsnudc", "sh/69gbupwr", "sh/j65sjudg", "sh/i5wrqpwv", "sh/58nal4v6", "sh/210b2tcz", "sh/yxkvudw7", "sh/zytwnixs", "sh/ml4vyxw3", "sh/nmdwr2xo", "sh/8nmd07e9", "sh/2p0vmtgz", "sh/3q9cfyxk", "sh/5kzy1wz2", "sh/kjqhsbyh", "sh/ihozq1gb", "sh/tozy5gzy", "sh/snqhwbyt", "sh/rmhg36h8", "sh/9sfytczu", "sh/8r6xkri9", "sh/cjetojm9", "sh/xknahonu", "sh/bi5sfe5o"],
  ["sh/pcvypknq", "sh/obmxwfml", "sh/f6dgja5o", "sh/18vyl0nu", "sh/ritwfq5s", "sh/qhkfml47", "sh/w3a5kbep", "sh/v214r6xk", "sh/8za5gbet", "sh/90jmpgfe", "sh/svu5svex", "sh/tw3m10fi", "sh/1czm507m", "sh/fah43qpw", "sh/ep8nalob", "sh/snql8b65"],
  ["sh/wbap0fed", "sh/hcj69kvy", "sh/ids72pwj", "sh/4fap4fep", "sh/yhon6lwf", "sh/bm9gfml4", "sh/xoryhw3a", "sh/ny9gbmls", "sh/9krydc3y", "sh/zupg72lw", "sh/2pwzaxo3", "sh/4ryhc769", "sh/qtgzehof", "sh/cvyhg765", "sh/ehgzihob", "sh/9cb2hg7y", "sh/ob2lobqd", "sh/nqtkf6p8", "sh/mp03m18n", "sh/xgb2l07a", "sh/wf2lsvqp", "sh/adk3ql8j"],
];

async function exists(p) {
  try { await fs.access(p); return true; } catch { return false; }
}

async function readJson(p) {
  return JSON.parse(await fs.readFile(p, "utf8"));
}

async function readFirstJson(paths) {
  for (const p of paths) if (await exists(p)) return { path: p, value: await readJson(p) };
  return null;
}

async function generationCounts() {
  const tasksDir = path.join(CAMPAIGN, "e1_swe_bench_pro_full/seed1818/tasks");
  const counts = { GENERATED: 0, GENERATION_FAILED: 0, GENERATION_ARTIFACT_LOST: 0, UNKNOWN: 0 };
  if (!(await exists(tasksDir))) return counts;
  for (const task of await fs.readdir(tasksDir)) {
    const p = path.join(tasksDir, task, "generation.json");
    if (!(await exists(p))) continue;
    const receipt = await readJson(p);
    const status = receipt.status ?? receipt.generation_status ?? "UNKNOWN";
    counts[status] = (counts[status] ?? 0) + 1;
  }
  return counts;
}

function findNumeric(root, keys) {
  if (!root || typeof root !== "object") return null;
  for (const key of keys) if (typeof root[key] === "number") return root[key];
  for (const value of Object.values(root)) {
    const hit = findNumeric(value, keys);
    if (hit !== null) return hit;
  }
  return null;
}

function pct(x, digits = 1) {
  return typeof x === "number" ? `${(100 * x).toFixed(digits)}%` : "—";
}

function money(x) {
  return typeof x === "number" ? `$${x.toFixed(2)}` : "—";
}

function flattenLanes(lanes) {
  return lanes.flatMap((lane) => [lane.id, lane.name, lane.status, lane.detail]);
}

async function writeBlob(p, blob) {
  if (blob && typeof blob.arrayBuffer === "function") {
    await fs.writeFile(p, new Uint8Array(await blob.arrayBuffer()));
    return;
  }
  if (blob instanceof Uint8Array || Buffer.isBuffer(blob)) {
    await fs.writeFile(p, blob);
    return;
  }
  if (blob instanceof ArrayBuffer) {
    await fs.writeFile(p, new Uint8Array(blob));
    return;
  }
  if (blob && blob.data !== undefined) {
    await writeBlob(p, blob.data);
    return;
  }
  throw new TypeError(`unsupported export payload: ${Object.prototype.toString.call(blob)} keys=${Object.keys(blob ?? {}).join(",")} ctor=${blob?.constructor?.name ?? "unknown"}`);
}

async function main() {
  const inspectTextById = new Map();
  for (const line of (await fs.readFile(STARTER_INSPECT, "utf8")).split(/\r?\n/)) {
    if (!line.trim()) continue;
    const record = JSON.parse(line);
    if (typeof record.id === "string" && typeof record.text === "string") {
      inspectTextById.set(record.id, record.text);
    }
  }
  const counts = await generationCounts();
  const e1Terminal = counts.GENERATED + counts.GENERATION_FAILED + counts.GENERATION_ARTIFACT_LOST;
  const e1Pending = Math.max(0, 731 - e1Terminal);
  const e1Full = await readFirstJson([
    path.join(CAMPAIGN, "e1_swe_bench_pro_full/seed1818/receipt.json"),
    path.join(CAMPAIGN, "e1_swe_bench_pro_full/seed1818/full_receipt.json"),
  ]);
  const e1State = await readJson(path.join(CAMPAIGN, "e1_swe_bench_pro_full/seed1818/run_state.json"));
  const e11Full = await readFirstJson([
    path.join(CAMPAIGN, "e11_full_receipt.json"),
    path.join(CAMPAIGN, "e11/e11_full_receipt.json"),
    path.join(CAMPAIGN, "e11/receipt.json"),
  ]);
  const e4 = await readJson(path.join(CAMPAIGN, "e4_recovery_pass16_receipt.json"));
  const e4Boundary = await readJson(path.join(CAMPAIGN, "e4_recovery_evidence_boundary_2026-08-22.json"));
  const e4Metric = e4Boundary.attempt_metric.value;
  const paidExecution = await readJson(path.join(ROOT, "outputs/e2_e7_paid_execution_receipt_2026-08-22.json"));
  const e2PaidAttempt = paidExecution.attempts.find((attempt) => attempt.lane === "E2");
  const e7PaidAttempt = paidExecution.attempts.find((attempt) => attempt.lane === "E7");
  const e5Aggregate = await readJson(path.join(ROOT, "outputs/e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json"));
  const e5AttemptMean = e5Aggregate.metrics.attempt_prefix_mean_failure_as_zero;
  const e5ScoredMean = e5Aggregate.metrics.scored_receipt_mean;
  const e5Attempted = e5Aggregate.selection.attempted_tasks;
  const e5Scored = e5Aggregate.selection.native_scored_tasks;
  const e5Unscored = e5Aggregate.selection.native_unscored_failures;
  const e5Interrupted = e5Aggregate.selection.operator_interrupted_tasks;

  const e1Score = e1Full ? findNumeric(e1Full.value, ["score", "pass_at_1", "pass@1"]) : null;
  const e1Resolved = e1Full ? findNumeric(e1Full.value, ["resolved", "passed", "success_count"]) : null;
  const e11Score = e11Full ? findNumeric(e11Full.value, ["score"]) : null;
  const e11Raw = e11Full?.value?.pass_at_1?.raw?.pass_at_1 ?? null;
  const e11Corrected = e11Full?.value?.pass_at_1?.corrected?.pass_at_1 ?? null;
  const e11Passed = e11Full?.value?.pass_at_1?.raw?.passes ?? null;
  const e11Cost = e11Full ? findNumeric(e11Full.value, ["estimated_actual_usd", "estimated_tinker_cost_usd", "tinker_cost_usd", "cost_usd"]) : null;
  const e7Reward = e7PaidAttempt.native_task_reward;
  const bridgeCharged = paidExecution.authorization.ending_charged_usd;
  const bridgeIncremental = paidExecution.authorization.incremental_charged_usd;
  const bridgeRemaining = paidExecution.authorization.remaining_under_persistent_cap_usd;

  const e1RunState = e1Full ? "SCORED" : e1State.status === "EVALUATING" ? "EVALUATING" : "RUNNING";
  const e1RunDetail = e1Full
    ? `${pct(e1Score)} exact pass@1${e1Resolved !== null ? ` · ${e1Resolved}/731 resolved` : ""}`
    : `${e1Terminal}/731 terminal · ${e1State.valid_candidates ?? counts.GENERATED} candidates in native evaluation`;
  const e11RunState = e11Full ? "SCORED" : "RUNNING";
  const e11RunDetail = e11Full
    ? `${pct(e11Score)} canonical raw · ${e11Passed}/312 passed`
    : "312 problems · Tinker + native verifier active";

  const lanes = [
    { id: "E1", name: "SWE-bench Pro", status: e1RunState, detail: e1Full ? `${pct(e1Score)} · ${e1Resolved ?? "—"}/731 resolved` : `${e1Terminal}/731 terminal · ${e1State.valid_candidates ?? counts.GENERATED} eval` },
    { id: "E2", name: "Frontier-SWE", status: "PARTIAL EXACT", detail: `${e2PaidAttempt.native_task_reward.toFixed(4)} native · 8/8 checks · suite:null` },
    { id: "E3", name: "SDAB", status: "BLOCKED", detail: "Private 80 tasks + runtime/grader" },
    { id: "E4", name: "Banker ToolBench", status: "RECOVERY", detail: `${e4Metric.toFixed(4)} attempt metric · suite score:null` },
    { id: "E5", name: "Apex Agents", status: "PARTIAL EXACT", detail: `${e5AttemptMean.toFixed(4)} prefix mean · ${e5Attempted}/480 · suite:null` },
    { id: "E6", name: "WebBench", status: "BLOCKED", detail: "Official env + ground truth" },
    { id: "E7", name: "BinaryAudit", status: "PARTIAL EXACT", detail: `${e7Reward.toFixed(1)} native task · output absent · suite:null` },
    { id: "E8", name: "LifeSciBench", status: "BLOCKED", detail: "Official package + license + grader" },
    { id: "E9", name: "MLE-bench", status: "PARTIAL EXACT", detail: "6/75 native · suite:null" },
    { id: "E10", name: "AgentHarm", status: "BLOCKED", detail: "Private split + policy grader" },
    { id: "E11", name: "VerilogEval", status: e11RunState, detail: `${pct(e11Score)} canonical raw · ${e11Passed}/312` },
    { id: "E12", name: "AppBench", status: "BLOCKED", detail: "License + deployment + humans" },
    { id: "E13", name: "OpenReward Games", status: "BLOCKED", detail: "Licensed provider holdout absent" },
    { id: "E14", name: "FrontierMath", status: "BLOCKED", detail: "Hosted Epoch evaluation only" },
  ];

  const todayHeadline = e1Full && e11Full ? "Two suites + new\nnative task evidence" : e11Full ? "Today: E11 + tasks\nadvanced" : "Today: E1 resumed,\nE11 on Tinker";
  const todaySubtitle = e1Full && e11Full
    ? `E1 and E11 have exact-suite outcomes. E5 reached ${e5Attempted} exact tasks; fresh bounded runs added native task receipts for E2 and E7. Partial-suite scores remain null.`
    : e11Full
      ? `E11 closed through Tinker and its native verifier at ${pct(e11Corrected ?? e11Score)} corrected pass@1; E1 has ${e1Terminal}/731 terminal generations and ${e1State.valid_candidates ?? counts.GENERATED} frozen candidates in native evaluation.`
      : "E1 is filling only missing generation receipts while E11 runs independently through Tinker and the native VerilogEval verifier.";

  const newTexts = [
    [
      "TINKER RL LAB  ·  DAILY STATUS",
      todayHeadline,
      todaySubtitle,
      "22 AUG 2026   /   LIVE RECEIPT SNAPSHOT",
      "14 AUG",
      "Lane map +\none-task smoke",
      "16 AUG",
      "E4 recovery\nmetric recorded",
      "17 AUG",
      "14 lanes\nassessed",
      "22 AUG",
      "E1 + E11 suites\nE2/E7 native tasks",
      "This week adds exact-suite outcomes plus bounded native task receipts—without relabeling partial coverage as a suite score.",
    ],
    [
      "GOALS  ·  RESEARCH PROGRAM",
      "Trustworthy evidence, portable execution",
      "02",
      "Four goals keep scientific claims, engineering, provenance, and spend aligned.",
      "SCIENCE",
      "Exact native evidence",
      "Measure capability with immutable tasks and benchmark-owned graders; never inherit substitute scores.",
      "[TRUST]",
      "ENGINEERING",
      "Portable + resumable",
      "Run the same evidence chain across GPU, Modal, and Tinker while preserving terminal states.",
      "[PORTABILITY]",
      "INTEGRITY",
      "Immutable provenance",
      "Bind source, split, model, W&B run, Hugging Face checkpoint, evaluator, and receipt.",
      "[RECEIPT]",
      "OPERATIONS",
      "Fail-closed gates",
      "Gate cost, side effects, private data, licenses, and provider access before launch.",
      "[CONTROL]",
      "BOTTOM LINE",
      "Scale coverage without expanding claim scope",
      "Every score must bind the exact task, environment, model, native grader, and final receipt.",
    ],
    [
      "ARCHITECTURE  ·  FOUR LAYERS",
      "Contracts become scores through one controlled path",
      "03",
      "Each layer preserves identity, artifacts, and evidence boundaries.",
      "01",
      "CONTRACT",
      "Immutable source, task, split, license, and native grader.",
      "02",
      "ADAPTERS",
      "E1–E14 launch, resume, caching, and terminal-state preservation.",
      "03",
      "COMPUTE + MODEL",
      "GPU or Modal executes; Tinker is gated; W&B and Hugging Face bind provenance.",
      "04",
      "NATIVE EVALUATOR",
      "Benchmark-owned grader writes the final receipt and scientific score.",
      "RULE",
      "Receipt before claim",
      "Only native evaluation creates a score. Preflights remain operational evidence.",
      "Sources: exact benchmark contracts and dated run receipts",
    ],
    [
      "STATUS  ·  TODAY'S LEDGER",
      "Broad coverage; exact scores remain sparse",
      "04",
      "14",
      "LANES ASSESSED",
      "Contract and readiness evidence audited; blocked gates remain explicit.",
      `${e1Terminal}/731`,
      "E1 TERMINAL",
      `${counts.GENERATED} generated · ${counts.GENERATION_FAILED + counts.GENERATION_ARTIFACT_LOST} terminal failures`,
      e11Full ? pct(e11Score) : "RUNNING",
      "E11 EXACT",
      e11Full ? "312-problem native result receipt" : "Tinker + native verifier in progress",
      e4Metric.toFixed(4),
      "E4 RECOVERY",
      "Attempt metric · 37/128 criteria · suite score:null",
      "14/14",
      "STATUS COVERAGE",
      "Every E1–E14 lane is terminal locally and keeps its evidence label",
      "What that actually means",
      e1Full ? "E1 · full exact receipt now available." : `E1 · all 731 generation records are terminal; ${e1State.valid_candidates ?? counts.GENERATED} frozen candidates are in native evaluation.`,
      `E2 · one exact task closed at ${e2PaidAttempt.native_task_reward.toFixed(4)} native reward with ${e2PaidAttempt.correctness_passed}/${e2PaidAttempt.correctness_total} hidden correctness checks; suite score:null.`,
      "E4 · recovery-grade replay without resampling; not a clean campaign pass.",
      `E5 · ${e5Attempted}/480 attempted; ${e5Scored} native-scored; prefix mean ${e5AttemptMean.toFixed(4)}; suite:null. E7 · native verifier closed one exact task at ${e7Reward.toFixed(1)} because the required output file was absent; suite:null.`,
      "TAKEAWAY",
      "Today increases exact execution, not claim scope.\n\nBlocked lanes remain score:null until immutable tasks and native verifiers exist.",
      "Readiness is not usefulness.",
      "Sources: exact receipts; E2/E7 paid receipt; E4 recovery",
    ],
    [
      "STATUS  ·  E1–E14 SCOREBOARD",
      "Every lane keeps its evidence label",
      "05",
      "Operational progress is visible, but only receipt-backed exact evaluation becomes a score.",
      ...flattenLanes(lanes),
      "Sources: outputs/modal_e1_e14/2026-08-16 and today's live run receipts",
    ],
    [
      "E1 FOCUS  ·  SWE-BENCH PRO",
      e1Full ? "Full suite scored without resampling terminal receipts" : "Resume the full suite without resampling terminal receipts",
      "06",
      `${e1Terminal}/731`,
      "TERMINAL",
      "Generation receipts at snapshot",
      `${counts.GENERATED}`,
      "GENERATED",
      "Candidate patches preserved",
      `${counts.GENERATION_FAILED + counts.GENERATION_ARTIFACT_LOST}`,
      "FAILED / LOST",
      "Terminal states are not resampled",
      `${e1Pending}`,
      "PENDING",
      e1Full ? "Exact evaluation complete" : "Native evaluation in progress",
      "WHAT WE CAN SAY",
      "The exact 731-task dataset and native evaluator are pinned.",
      "All existing terminal receipts are preserved; W&B is online before generation.",
      e1Full ? `The exact-suite receipt reports ${pct(e1Score)} pass@1.` : "The live state is native evaluation progress, not yet a benchmark score.",
      "WHAT WE CANNOT SAY",
      "The earlier one-task 0.0 is not the 731-task suite result.",
      e1Full ? `${counts.GENERATED} generated patches do not mean ${counts.GENERATED} resolved tasks; only ${e1Resolved ?? 0} passed.` : "Generated patches are not resolved tasks until native evaluation completes.",
      "No missing or failed task is silently replaced by a new sample.",
      "Source: E1 full-suite receipt and run state",
    ],
    [
      "RESULT  ·  PORTFOLIO CHECK",
      "The training-source gain is calibration, not E1–E14",
      "07",
      "AS OF 22 AUG 2026",
      null, null, null, null, null, null, null, null, null, null,
      "INTERPRETATION",
      "Useful calibration",
      "Training-source behavior only",
      "The local portfolio check does not substitute for held-out E1–E14 native evaluation.",
      "Evidence boundary",
      "Keep it out of exact-suite score totals. [LOCAL DIAGNOSTIC · NOT PRIMARY EVAL]",
      "Source: archived local portfolio check from the prior status deck",
    ],
    [
      "GATES  ·  WHY OTHER LANES STAY BLOCKED",
      "No evidence, no launch",
      "08",
      "PRIVATE ASSETS",
      "E3 · E6 · E8 · E9",
      "Dated receipts confirm provider packages, live environments, licenses, or native graders are still missing.",
      "EXACT TASK / RECOVERY",
      "E2 · E5 · E7",
      `E2 closed one exact task at ${e2PaidAttempt.native_task_reward.toFixed(4)} with 8/8 correctness checks; E5 attempted ${e5Attempted} exact APEX tasks with ${e5Scored} native scores; E7 closed one native task at 0.0. All suite scores remain null.`,
      "SAFETY / ACCESS",
      "E10 · E14",
      "E10 is safety-gated; E14 is private at Epoch. No public substitute is allowed.",
      "RECEIPT-ONLY",
      "E12 · E13",
      "Native fixtures or local runtimes pass, but provider-held-out results still do not exist.",
      "Sources: terminal receipts; E5 prefix aggregate",
    ],
    [
      "EXECUTION  ·  TINKER ALONGSIDE MODAL",
      "Parallelize disjoint lanes, not the same receipt",
      "09",
      "Modal hosts official environments; Tinker gates model calls; E1 uses the GPU backend.",
      "E1",
      `${e1Terminal}/731`,
      "terminal generations",
      "E11",
      e11Full ? pct(e11Score) : "LIVE",
      e11Full ? "exact native pass@1" : "Tinker run active",
      "BRIDGE CAP",
      "$10.00",
      `${money(bridgeCharged)} charged · ${money(bridgeRemaining)} remains`,
      "E4 JUDGE",
      money(e4.verifier?.cost_usd),
      "recorded recovery cost",
      "01",
      "GATED",
      "Gate first",
      "The receipt binds online W&B initialization before Tinker client construction; the Hugging Face checkpoint is immutable.",
      "02",
      "PARALLEL",
      "Run independently",
      `E1, E2, E5, E7, and E11 preserve independent receipts; this authorization added ${money(bridgeIncremental)} of charged model calls.`,
      "03",
      "RECEIPT",
      "Promote by receipt",
      "Only a completed native evaluator receipt changes the scoreboard to SCORED.",
      "Sources: E1 GPU resume preflight, E11 launch preflight, W&B/HF receipts",
    ],
    [
      "NEXT  ·  ORDERED ACTIONS",
      e1Full && e11Full ? "Package evidence; unblock external access" : "Finish E1, then unblock exact evidence",
      "10",
      "The order below minimizes resampling risk and avoids spending against incomplete contracts.",
      "01",
      e1Full ? "Package E1 exact receipt" : "Finish E1 terminal coverage",
      e1Full ? `Complete — exact evaluator receipt records ${pct(e1Score)} pass@1 and ${e1Resolved ?? 0}/731 resolved.` : `All generations are terminal; finish the one native replay over ${e1State.valid_candidates ?? counts.GENERATED} frozen candidates.`,
      "02",
      e11Full ? "Package E11 exact receipt" : "Close E11 native result",
      e11Full ? `Complete — ${pct(e11Score)} canonical raw exact pass@1 recorded.` : "Let the 312-problem Tinker run finish and persist its native verifier receipt.",
      "03",
      "Package bounded tasks; unblock external evidence",
      `Record E2 at ${e2PaidAttempt.native_task_reward.toFixed(4)} and E7 at ${e7Reward.toFixed(1)} as native task evidence; keep suite:null and pursue provider access.`,
      "EVIDENCE RULE",
      "No blocked suite may inherit a substitute score.",
      "Source: status ledger and canonical blockers",
    ],
    [
      "CLOSE  ·  22 AUG 2026",
      "Today advances execution, not rhetoric",
      "11",
      "DAILY STATUS",
      "E1",
      e1RunState,
      e1RunDetail,
      e1Full ? "Exact receipt available" : "No full-suite score yet",
      "E11",
      e11RunState,
      e11RunDetail,
      e11Full ? "Exact receipt available" : "Paid run still in progress",
      "PORTFOLIO",
      "14 ASSESSED",
      `E2 ${e2PaidAttempt.native_task_reward.toFixed(4)} · E4 recovery · E5 ${e5AttemptMean.toFixed(4)} prefix · E7 ${e7Reward.toFixed(1)}`,
      "No substitute suites; blocked lanes remain score:null",
      "NEXT CHECK",
      e1Full && e11Full ? "Package exact receipts" : "Finish native evaluation",
      e1Full && e11Full ? "Package receipts; pursue external access." : e11Full ? "Finish E1 native evaluation; E11 is closed." : "Finish E1 native evaluation and E11 native verification.",
      "OWNER",
      "Tinker RL Lab",
      "Source: live campaign ledger and immutable receipts",
    ],
  ];

  const sourceNotes = [
    ["outputs/modal_e1_e14/2026-08-16/PROGRESS.md", "outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818", "outputs/modal_e1_e14/2026-08-16/e11"],
    ["outputs/E1_E14_Terminal_Status_2026-08-22.json", "outputs/modal_e1_e14/2026-08-16/PROGRESS.md", "outputs/modal_e1_e14/2026-08-16/e11/launch_preflight_receipt.json"],
    ["outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818/gpu_resume_preflight.json", "outputs/e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json", "outputs/modal_e1_e14/2026-08-16/e11/launch_preflight_receipt.json"],
    ["outputs/e2_e7_paid_execution_receipt_2026-08-22.json", "outputs/modal_e1_e14/2026-08-16/e4_recovery_evidence_boundary_2026-08-22.json"],
    ["outputs/modal_e1_e14/2026-08-16/preflight_summary.json", "outputs/modal_e1_e14/2026-08-16/NON_E11_READINESS.md", "outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json"],
    ["outputs/modal_e1_e14/2026-08-16/e1_swe_bench_pro_full/seed1818"],
    ["outputs/Tinker_RL_Progress_Update_2026-08-14.pptx"],
    [
      "outputs/e3_sdab/terminal_receipt_2026-08-22.json",
      "outputs/e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json",
      "outputs/e6_webbench/terminal_receipt_2026-08-22.json",
      "outputs/e8_lifescibench/terminal_receipt_2026-08-22.json",
      "outputs/e9_mle_bench/terminal_receipt_2026-08-22.json",
      "outputs/e10_agentharm/terminal_receipt_2026-08-22.json",
      "outputs/e12_appbench/terminal_receipt_2026-08-22.json",
      "outputs/e13_openreward_games/terminal_receipt_2026-08-22.json",
      "outputs/e14_frontiermath/terminal_receipt_2026-08-22.json",
    ],
    ["outputs/e2_e7_paid_execution_receipt_2026-08-22.json", "outputs/e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json", "outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json", "outputs/modal_e1_e14/2026-08-16/e4_recovery_evidence_boundary_2026-08-22.json"],
    ["outputs/modal_e1_e14/2026-08-16/PROGRESS.md"],
    ["outputs/modal_e1_e14/2026-08-16", "outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json"],
  ];

  const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER));
  for (let slideIndex = 0; slideIndex < anchors.length; slideIndex += 1) {
    const oldTexts = anchors[slideIndex].map((id) => inspectTextById.get(id));
    const requested = newTexts[slideIndex];
    const editableShapes = presentation.slides.items[slideIndex].shapes.items.filter((shape) => shape.text.toString().trim().length > 0);
    if (oldTexts.some((text) => typeof text !== "string") || requested.length !== oldTexts.length || editableShapes.length !== oldTexts.length) {
      throw new Error(`slide ${slideIndex + 1}: source=${oldTexts.length}, imported=${editableShapes.length}, new=${requested.length}`);
    }
    for (let i = 0; i < oldTexts.length; i += 1) {
      if (requested[i] === null || requested[i] === oldTexts[i]) continue;
      editableShapes[i].text.replace(oldTexts[i], requested[i]);
    }
    const notes = [
      "[Sources]",
      ...sourceNotes[slideIndex].map((source) => `- ${source}`),
      "- Snapshot date: 2026-08-22 (Asia/Kolkata)",
      "- Evidence rule: operational/preflight receipts are not scientific scores unless a native exact-suite result receipt is present.",
      "[/Sources]",
    ].join("\n");
    presentation.slides.items[slideIndex].speakerNotes.textFrame.setText(notes);
    presentation.slides.items[slideIndex].speakerNotes.setVisible(true);
  }

  await fs.mkdir(RENDER_DIR, { recursive: true });
  await fs.mkdir(FINAL_LAYOUT_DIR, { recursive: true });
  for (let i = 0; i < presentation.slides.items.length; i += 1) {
    const slide = presentation.slides.items[i];
    await writeBlob(path.join(RENDER_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`), await presentation.export({ slide, format: "png", scale: 2 }));
    await fs.writeFile(path.join(FINAL_LAYOUT_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await (await slide.export({ format: "layout" })).text());
  }
  await writeBlob(path.join(WORK, "final-montage.webp"), await presentation.export({ format: "webp", montage: true, scale: 1 }));
  await writeBlob(OUTPUT, await PresentationFile.exportPptx(presentation));
  await fs.writeFile(path.join(WORK, "status-snapshot.json"), JSON.stringify({
    recorded_at: new Date().toISOString(),
    e1: { counts, terminal: e1Terminal, pending: e1Pending, run_state: e1State.status, valid_candidates: e1State.valid_candidates ?? counts.GENERATED, full_receipt: e1Full?.path ?? null, score: e1Score },
    e2: { paid_execution_receipt: "outputs/e2_e7_paid_execution_receipt_2026-08-22.json", native_task_reward: e2PaidAttempt.native_task_reward, correctness_passed: e2PaidAttempt.correctness_passed, correctness_total: e2PaidAttempt.correctness_total, suite_score: e2PaidAttempt.suite_score },
    e7: { paid_execution_receipt: "outputs/e2_e7_paid_execution_receipt_2026-08-22.json", native_task_reward: e7PaidAttempt.native_task_reward, native_failure_reason: e7PaidAttempt.native_failure_reason, suite_score: e7PaidAttempt.suite_score },
    paid_bridge: paidExecution.authorization,
    e5: { exact_prefix_aggregate: "outputs/e5_apex_agents/e5_exact_sequential_prefix_aggregate_2026-08-22.json", attempted_tasks: e5Attempted, native_scored_tasks: e5Scored, native_unscored_failures: e5Unscored, operator_interrupted_tasks: e5Interrupted, attempt_prefix_mean_failure_as_zero: e5AttemptMean, scored_receipt_mean: e5ScoredMean, official_suite_tasks: 480, suite_score: null },
    e11: { full_receipt: e11Full?.path ?? null, canonical_raw_score: e11Score, secondary_corrected_score: e11Corrected, cost_usd: e11Cost },
    output: OUTPUT,
  }, null, 2));
  console.log(JSON.stringify({ output: OUTPUT, e1Terminal, e1Pending, e1Score, e11CanonicalRawScore: e11Score, e11SecondaryCorrectedScore: e11Corrected, e11Cost }, null, 2));
}

try {
  await main();
} catch (error) {
  console.error(error instanceof Error ? error.stack : String(error));
  process.exitCode = 1;
}
