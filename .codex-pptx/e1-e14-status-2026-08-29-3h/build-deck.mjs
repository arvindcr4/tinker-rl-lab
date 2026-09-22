import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/arvind/Developer/agentic_repos/tinker-rl-lab";
const WORK = path.join(ROOT, ".codex-pptx/e1-e14-status-2026-08-29-3h");
const OUT = path.join(ROOT, "outputs/Tinker_RL_E1_E14_Completion_Update_2026-08-29.pptx");
const audit = JSON.parse(await fs.readFile(path.join(ROOT, "outputs/E1_E14_Terminal_Status_2026-08-29.json"), "utf8"));
const spend = JSON.parse(await fs.readFile(path.join(ROOT, "outputs/e1_e14_incremental_spend_ledger_2026-08-29.json"), "utf8"));

const P = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  ink: "#0A0A0A", muted: "#5E6673", line: "#D9DEE7", panel: "#F3F5F8",
  blue: "#2764E7", blueSoft: "#E8F0FF", green: "#157A55", greenSoft: "#E7F6EF",
  amber: "#A45B00", amberSoft: "#FFF1D8", red: "#B42318", redSoft: "#FDECEA", white: "#FFFFFF"
};
const FONT = "Helvetica Neue";
const lanes = audit.lanes;
const exact = lanes.filter(x => x.benchmark_status === "SCORED_EXACT_SUITE");
const partial = lanes.filter(x => x.benchmark_status.includes("PARTIAL"));
const blocked = lanes.filter(x => x.benchmark_status === "BLOCKED_EXTERNAL");
const benchmarkName = {
  E1: "SWE-bench Pro", E2: "FrontierSWE", E3: "SDAB", E4: "BankerToolBench",
  E5: "APEX-Agents", E6: "WebBench", E7: "BinaryAudit", E8: "LifeSciBench",
  E9: "MLE-bench", E10: "AgentHarm", E11: "VerilogEval", E12: "AppBench",
  E13: "OpenReward Games", E14: "FrontierMath"
};

function box(slide, left, top, width, height, fill=C.panel, radius="roundRect", line=C.line, name="box") {
  return slide.shapes.add({ geometry: radius, name, position: { left, top, width, height }, fill, line: { style: "solid", fill: line, width: 1 } });
}
function txt(slide, value, left, top, width, height, fontSize=22, color=C.ink, bold=false, align="left", name="text") {
  const s = slide.shapes.add({ geometry: "textbox", name, position: { left, top, width, height }, fill: "none", line: { style: "solid", fill: "none", width: 0 } });
  s.text = String(value);
  s.text.style = { typeface: FONT, fontSize, color, bold, alignment: align, verticalAlignment: "middle", autoFit: "shrinkText", insets: { top: 0, right: 0, bottom: 0, left: 0 } };
  return s;
}
function line(slide, x1, y1, x2, y2, color=C.line, width=2) {
  return slide.shapes.add({ geometry: "line", position: { left: x1, top: y1, width: x2-x1, height: y2-y1 }, fill: "none", line: { style: "solid", fill: color, width } });
}
function base(title, kicker, n) {
  const s = P.slides.add(); s.background.fill = C.white;
  txt(s, kicker.toUpperCase(), 42, 30, 500, 24, 13, C.blue, true, "left", "kicker");
  txt(s, title, 42, 58, 1120, 64, 39, C.ink, true, "left", "title");
  line(s, 42, 132, 1238, 132, C.ink, 1);
  txt(s, String(n).padStart(2,"0"), 1184, 675, 54, 18, 13, C.muted, false, "right", "slide-number");
  return s;
}
function note(slide, entries, cue) {
  slide.speakerNotes.textFrame.setText(`${cue}\n\n[Sources]\n${entries.map(x=>`- ${x}`).join("\n")}\n[/Sources]`);
}
function pill(slide, label, left, top, width, kind) {
  const map = { exact:[C.greenSoft,C.green], partial:[C.amberSoft,C.amber], blocked:[C.redSoft,C.red], info:[C.blueSoft,C.blue] };
  const [fill,color] = map[kind]; box(slide,left,top,width,28,fill,"roundRect",fill,`pill-${label}`); txt(slide,label,left+10,top,width-20,28,13,color,true,"center");
}
function metric(slide, value, label, left, top, width, kind="info") {
  const fills={info:C.blueSoft,exact:C.greenSoft,partial:C.amberSoft,blocked:C.redSoft};
  const colors={info:C.blue,exact:C.green,partial:C.amber,blocked:C.red};
  box(slide,left,top,width,178,fills[kind],"roundRect",fills[kind],`metric-${label}`);
  txt(slide,value,left+24,top+26,width-48,72,47,colors[kind],true);
  txt(slide,label,left+24,top+110,width-48,42,18,C.ink,true);
}
function statusKind(lane) { return lane.benchmark_status === "SCORED_EXACT_SUITE" ? "exact" : lane.benchmark_status.includes("PARTIAL") ? "partial" : "blocked"; }
function shortScope(l) {
  if (l.lane === "E1") return "731/731 terminal · 713 native";
  if (l.lane === "E2") return "1/17 tasks";
  if (l.lane === "E3") return "private 80-task bundle absent";
  if (l.lane === "E4") return "1/100 tasks · 37/128 criteria";
  if (l.lane === "E5") return "11/480 attempted · cap-bound";
  if (l.lane === "E6") return "live env + scorer absent";
  if (l.lane === "E7") return "1/46 tasks";
  if (l.lane === "E8") return "0/750 · package absent";
  if (l.lane === "E9") return l.native_grade_coverage;
  if (l.lane === "E10") return "private tasks + grader absent";
  if (l.lane === "E11") return "129/312 canonical raw";
  if (l.lane === "E12") return "deployment + tasks absent";
  if (l.lane === "E13") return "profile active · suite absent";
  return "private hosted eval absent";
}

// 1 — cover
{
  const s=P.slides.add(); s.background.fill=C.white;
  txt(s,"TINKER RL LAB · EVIDENCE-GATED RESEARCH",42,36,700,28,15,C.blue,true);
  txt(s,"How reliable are",42,150,720,72,57,C.ink,true);
  txt(s,"RL-trained agents?",42,218,760,86,66,C.ink,true);
  txt(s,"Fourteen real-world evaluations across code, tools, science, safety, hardware, and mathematics",42,355,690,102,26,C.muted,false);
  txt(s,"29 Aug 2026 · editable briefing",42,625,460,28,16,C.muted,false);
  const concepts=[["SOFTWARE","Code repair"],["TOOLS","Browser + APIs"],["SCIENCE","ML + life science"],["SAFETY","Alignment + security"],["DESIGN","Apps + hardware"],["MATH","Frontier reasoning"]];
  concepts.forEach((c,i)=>{const x=820+(i%2)*202,y=154+Math.floor(i/2)*112; box(s,x,y,184,92,i<2?C.blueSoft:(i<4?C.greenSoft:C.amberSoft),"roundRect",i<2?C.blueSoft:(i<4?C.greenSoft:C.amberSoft)); txt(s,c[0],x+16,y+12,152,22,13,i<2?C.blue:(i<4?C.green:C.amber),true); txt(s,c[1],x+16,y+40,152,38,18,C.ink,true);});
  txt(s,"2 complete · 5 partial · 7 provider-gated",820,520,386,62,20,C.ink,true);
  note(s,["Local terminal audit: outputs/E1_E14_Terminal_Status_2026-08-29.json","zvf-program/flagship/pavlovs_domain_contract.json"],"Start with the research question and the six capabilities being tested. The E-codes are traceability labels, not the story.");
}

// 2 — executive outcome
{
  const s=base("Two benchmarks are complete; five are informative but partial", "Executive readout",2);
  txt(s,"Seven evaluations require provider-controlled tasks, graders, environments, licenses, or hosted access before a scientific score can exist.",42,155,1145,68,24,C.ink,true);
  metric(s,"2 / 14","exact full-suite scores",42,266,360,"exact");
  const e9=lanes.find(x=>x.lane==="E9");
  metric(s,e9.native_grade_coverage.replace(" unique competitions","").replace("/"," / "),"MLE-bench native coverage",460,266,360,"partial");
  const charged=spend.total_counted_incremental_spend_usd ?? spend.counted_new_spend_usd ?? spend.incremental_spend_usd ?? 0;
  metric(s,`$${Number(charged).toFixed(2)}`,"incremental spend of $50 cap",878,266,360,"info");
  box(s,42,486,1196,118,C.ink,"roundRect",C.ink);
  txt(s,"No suite score is inferred from partial coverage. Missing native receipts remain score: null.",72,509,1136,72,25,C.white,true,"center");
  note(s,["outputs/E1_E14_Terminal_Status_2026-08-29.json","outputs/e1_e14_incremental_spend_ledger_2026-08-29.json"],"Lead with the scientific outcome. SWE-bench Pro and VerilogEval are complete; MLE-bench is a coverage count, not a suite score.");
}

// 3 — concepts
{
  const s=base("What the evaluation portfolio actually tests", "Research concepts",3);
  const concepts=[
    ["Software engineering","Repair real repositories and reason over long code trajectories","SWE-bench Pro · FrontierSWE · SDAB"],
    ["Tool-using agents","Operate browsers, APIs, finance tools, and enterprise workflows","BankerToolBench · APEX-Agents · WebBench"],
    ["Scientific problem solving","Run end-to-end ML and life-science investigations","MLE-bench · LifeSciBench"],
    ["Safety under action","Resist harmful objectives while using tools and code","AgentHarm · BinaryAudit"],
    ["Artifact creation","Produce verifiable hardware and visual applications","VerilogEval · AppBench"],
    ["Long-horizon reasoning","Plan through games and hard mathematical problems","OpenReward Games · FrontierMath"]
  ];
  concepts.forEach((c,i)=>{const x=42+(i%3)*406,y=164+Math.floor(i/3)*220; const soft=i%3===0?C.blueSoft:i%3===1?C.greenSoft:C.amberSoft; const col=i%3===0?C.blue:i%3===1?C.green:C.amber; box(s,x,y,374,190,soft,"roundRect",soft); txt(s,c[0],x+22,y+18,330,38,23,C.ink,true); txt(s,c[1],x+22,y+64,330,62,17,C.muted,false); line(s,x+22,y+136,x+352,y+136,col,2); txt(s,c[2],x+22,y+146,330,30,14,col,true);});
  txt(s,"The common question: can one trained policy produce valid, reproducible outcomes across very different interactive domains?",42,622,1170,34,18,C.ink,true);
  note(s,["zvf-program/flagship/pavlovs_domain_contract.json"],"Explain the six capability families before discussing status. This is a cross-domain reliability study, not fourteen unrelated test runs.");
}

// 4 — all lanes
{
  const s=base("Every benchmark and its current evidence boundary", "Benchmark register",4);
  const renderCol=(items,x)=>items.forEach((l,i)=>{const y=157+i*68; const k=statusKind(l); const c=k==="exact"?C.green:k==="partial"?C.amber:C.red; box(s,x,y,568,56,C.panel,"roundRect",C.panel,`row-${l.lane}`); box(s,x,y,7,56,c,"rect",c); txt(s,l.lane,x+20,y,42,56,14,C.muted,true); pill(s,k.toUpperCase(),x+70,y+14,86,k); txt(s,benchmarkName[l.lane],x+172,y+5,170,23,16,C.ink,true); txt(s,shortScope(l),x+172,y+28,370,24,14,C.muted,false);});
  renderCol(lanes.slice(0,7),42); renderCol(lanes.slice(7),654);
  note(s,["outputs/E1_E14_Terminal_Status_2026-08-29.json","zvf-program/flagship/pavlovs_domain_contract.json"],"Use benchmark names first; the E-labels exist only for audit traceability. Pause on the two complete results, MLE-bench coverage, and the seven provider dependencies.");
}

// 5 — E9
{
  const e9=lanes.find(x=>x.lane==="E9"); const s=base("Why native receipts matter: the MLE-bench recovery funnel", "Scientific validity",5);
  const coverage=/([0-9]+)\/([0-9]+)/.exec(e9.native_grade_coverage) ?? [null,"0","75"];
  const inventory=/([0-9]+) receipts across ([0-9]+) competition IDs/.exec(e9.receipt_inventory) ?? [null,"0","0"];
  const breakdown=/([0-9]+) competition IDs have only failed receipts.*?([0-9]+) competition IDs have no run receipt/.exec(e9.remaining_breakdown) ?? [null,"0","0"];
  const graded=coverage[1], total=coverage[2], receiptIds=inventory[2], failedOnly=breakdown[1], noReceipt=breakdown[2];
  const latest=e9.latest_competition ?? {};
  const latestName=String(latest.competition_id ?? "native recovery").split("-").map(w=>w ? w[0].toUpperCase()+w.slice(1) : w).join(" ");
  const latestScore=Number.isFinite(Number(latest.score)) ? Number(latest.score).toFixed(5) : "score unavailable";
  const stages=[{v:total,l:"official competitions",c:C.ink},{v:receiptIds,l:"competition IDs with receipts",c:C.blue},{v:graded,l:"unique native grades",c:C.green},{v:String(e9.remaining_without_native_grade),l:"still without native grade",c:C.red}];
  stages.forEach((a,i)=>{const x=42+i*300; box(s,x,174,264,146,i===3?C.redSoft:(i===2?C.greenSoft:C.panel),"roundRect",i===3?C.redSoft:(i===2?C.greenSoft:C.panel)); txt(s,a.v,x+22,190,220,62,44,a.c,true); txt(s,a.l,x+22,258,220,40,16,C.ink,true); if(i<3){txt(s,"→",x+268,217,28,40,25,C.muted,true,"center");}});
  box(s,42,352,1196,174,C.blueSoft,"roundRect",C.blueSoft);
  txt(s,"Latest native recovery",66,374,260,28,16,C.blue,true);
  txt(s,`${latestName} · ${latestScore}`,66,410,1040,48,30,C.ink,true);
  const latestState=[latest.valid_submission ? "valid submission" : "invalid submission",latest.any_medal ? "medal" : "no medal",latest.above_median ? "above median" : "below median",latest.is_lower_better === true ? "lower is better" : latest.is_lower_better === false ? "higher is better" : null,latest.native_grader_pandas_version ? `pandas ${latest.native_grader_pandas_version} verifier` : "native verifier"].filter(Boolean).join(" · ");
  txt(s,latestState,66,468,1040,32,18,C.muted,false);
  txt(s,"Smartphone + Freesound graded; Plant + Jigsaw timeouts receipted · suite score null",66,502,1040,22,15,C.blue,true);
  box(s,42,550,575,84,C.amberSoft,"roundRect",C.amberSoft); txt(s,`${failedOnly} failed-only`,64,560,180,28,22,C.amber,true); txt(s,"no saved submission",64,592,260,24,16,C.ink,false);
  box(s,663,550,575,84,C.redSoft,"roundRect",C.redSoft); txt(s,`${noReceipt} no receipt`,685,560,180,28,22,C.red,true); txt(s,"no native-grade artifact exists",685,592,300,24,16,C.ink,false);
  note(s,["outputs/e9_mle_bench/modal_streaming/smartphone-decimeter-2022-1207910d9504/receipt.json","outputs/e9_mle_bench/modal_streaming/freesound-audio-tagging-2019-c160e87615be/receipt.json","outputs/e9_mle_bench/modal_streaming/plant-pathology-2021-fgvc8-1afc1ef65a3a/receipt.json","outputs/e9_mle_bench/modal_streaming/jigsaw-unintended-bias-in-toxicity-classification-be21ca31edd2/receipt.json","https://arxiv.org/abs/2410.07095","https://github.com/openai/mle-bench"],`Use the funnel to teach the evidence rule: code execution is not evaluation, a valid submission is not a suite score, and native grading must preserve the official denominator. Guarded Smartphone and Freesound repairs added two native grades; ${e9.remaining_without_native_grade} missing competitions keep the ${total}-task aggregate null.`);
}

// 6 — exact results
{
  const s=base("Two complete evaluations answer two different questions", "Exact evidence",6);
  box(s,42,166,568,424,C.greenSoft,"roundRect",C.greenSoft); pill(s,"SOFTWARE REPAIR",68,192,150,"exact"); txt(s,"SWE-bench Pro",68,244,450,58,34,C.ink,true); txt(s,"0.002736",68,314,420,74,53,C.green,true); txt(s,"Can the agent resolve real repository issues?",68,392,470,46,19,C.ink,true); txt(s,"731 / 731 terminal generations · 713 native evaluations",68,454,470,44,17,C.muted,false); txt(s,"E1 audit label",68,520,470,24,14,C.green,true);
  box(s,670,166,568,424,C.greenSoft,"roundRect",C.greenSoft); pill(s,"HARDWARE DESIGN",696,192,150,"exact"); txt(s,"VerilogEval",696,244,450,58,34,C.ink,true); txt(s,"0.41346",696,314,420,74,53,C.green,true); txt(s,"Can the agent synthesize functionally correct RTL?",696,392,470,46,19,C.ink,true); txt(s,"129 / 312 canonical raw · native full receipt",696,454,470,44,17,C.muted,false); txt(s,"E11 audit label",696,520,470,24,14,C.green,true);
  note(s,["outputs/modal_e1_e14/2026-08-16/","outputs/modal_e1_e14/2026-08-16/e11_full_receipt.json"],"These are the only results to quote as portfolio benchmark scores. State their denominators with the values.");
}

// 7 — artifact continuity
{
  const s=base("Reproducibility depends on the artifact chain", "Artifact continuity",7);
  const nodes=[{t:"Tinker sampler",d:"provider checkpoint deleted / 404",k:"blocked"},{t:"HF adapter",d:"immutable commit 64444133…",k:"exact"},{t:"Merged checkpoint",d:"26 shards · 71.9 GB pointer",k:"exact"},{t:"Native receipts",d:"grades and submissions retained",k:"info"}];
  nodes.forEach((n,i)=>{const x=42+i*300; const fills={blocked:C.redSoft,exact:C.greenSoft,info:C.blueSoft}; const colors={blocked:C.red,exact:C.green,info:C.blue}; box(s,x,212,264,210,fills[n.k],"roundRect",fills[n.k]); txt(s,String(i+1).padStart(2,"0"),x+22,232,60,28,16,colors[n.k],true); txt(s,n.t,x+22,280,220,48,25,C.ink,true); txt(s,n.d,x+22,342,220,56,17,C.muted,false); if(i<3) txt(s,"→",x+268,292,28,40,25,C.muted,true,"center");});
  box(s,42,472,1196,114,C.ink,"roundRect",C.ink); txt(s,"Consequence: deterministic replay and native-only regrade remain possible; fresh sampling from the original Tinker sampler is unavailable.",70,493,1140,70,22,C.white,true,"center");
  note(s,["outputs/E1_E14_Terminal_Status_2026-08-29.json","Hugging Face adapter commit and merged checkpoint pointer recorded in local audit artifacts"],"The deleted sampler prevents fresh generations. The surviving adapter, merged checkpoint, submissions, and receipts still support deterministic replay and regrading.");
}

// 8 — blockers
{
  const s=base("Why seven evaluations cannot be completed locally", "External completion path",8);
  const items=[
    ["SDAB · E3","private bundle · runtime · grader · license"],["WebBench · E6","live environment · reset · ground truth · scorer"],["LifeSciBench · E8","750-task package · license · native grader"],["AgentHarm · E10","private tasks · official grader"],["AppBench · E12","deployment · grading protocol · artifacts · license"],["OpenReward Games · E13","immutable held-out suite · grading contract"],["FrontierMath · E14","private hosted evaluation"]
  ];
  items.forEach((it,i)=>{const col=i<4?0:1, row=i<4?i:i-4, x=42+col*610,y=158+row*101; box(s,x,y,568,82,C.redSoft,"roundRect",C.redSoft); txt(s,it[0],x+20,y+8,190,66,17,C.red,true); line(s,x+224,y+18,x+224,y+64,C.red,2); txt(s,it[1],x+244,y+8,296,66,16,C.ink,true);});
  box(s,652,461,568,101,C.blueSoft,"roundRect",C.blueSoft); txt(s,"Access routes monitored",674,475,240,26,18,C.blue,true); txt(s,"No provider replies yet · official issues, discussions, and mail rechecked",674,507,510,38,16,C.ink,true);
  txt(s,"Official routes are filed and monitored; public or paid alternatives remain non-equivalent until a provider confirms the exact suite.",42,616,1160,42,17,C.muted,false);
  note(s,["outputs/E3_E14_EXTERNAL_FOLLOWUP_RECEIPT_2026-08-29.json","outputs/e12_appbench/external_access_recheck_2026-08-29.json","outputs/e13_openreward_games/external_access_recheck_2026-08-29.json","https://openreward.ai/arvindcr4"],"Every red row names the provider artifact required to resume. Two additional official requests were filed today; prospective alternatives remain access routes, not benchmark substitutes.");
}

// 9 — decision
{
  const s=base("From partial evidence to a publishable study", "Decision, access & venue",9);
  const rows=[
    ["1","Provider grants","Release exact task assets, license terms, native graders, and hosted endpoints."],
    ["2","Data continuity","Restore a sampler-compatible checkpoint or provide saved submissions for native-only regrade."],
    ["3","Compute budget","A full APEX-Agents pass is projected at ~$122.57 before judges—outside the current cap."],
    ["4","Evidence discipline","Publish SWE-bench Pro and VerilogEval; keep every partial aggregate at score: null."]
  ];
  rows.forEach((r,i)=>{const y=166+i*103; box(s,42,y,1196,84,i===3?C.greenSoft:C.panel,"roundRect",i===3?C.greenSoft:C.panel); txt(s,r[0],62,y,52,84,25,i===3?C.green:C.blue,true,"center"); txt(s,r[1],132,y+10,260,30,20,C.ink,true); txt(s,r[2],410,y+8,790,64,18,C.muted,false);});
  box(s,42,602,1196,52,C.ink,"roundRect",C.ink); txt(s,"Conference target: ICLR 2027 · withdraw overlapping NeurIPS paper before 18 Sep abstract",64,602,1152,52,20,C.white,true,"center");
  note(s,["outputs/E1_E14_Terminal_Status_2026-08-29.json","outputs/e1_e14_incremental_spend_ledger_2026-08-29.json","autoresearch/deli-neurips-tmlr-260802/drafts/PUBLICATION_READINESS.md","https://iclr.cc/Conferences/2027/CallForPapers","https://iclr.cc/Conferences/2027/AuthorGuidelines","https://jmlr.org/tmlr/editorial-policies.html","https://neurips.cc/Conferences/2026/MainTrackHandbook"],"Close on decisions. Publish E1 and E11 as the exact results; keep partial suites null. ICLR 2027 is the next conference target only if the overlapping NeurIPS submission is formally withdrawn before the 18 September abstract deadline; the paper deadline is 25 September AOE. Otherwise keep NeurIPS and use rolling TMLR after that review ends.");
}

await fs.mkdir(path.join(WORK,"rendered"),{recursive:true});
for (const [i,slide] of P.slides.items.entries()) {
  const stem=`slide-${String(i+1).padStart(2,"0")}`;
  const png=await P.export({slide,format:"png",scale:1});
  await fs.writeFile(path.join(WORK,"rendered",`${stem}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await slide.export({format:"layout"});
  await fs.writeFile(path.join(WORK,"rendered",`${stem}.layout.json`),await layout.text());
}
const montage=await P.export({format:"webp",montage:true,scale:1});
await fs.writeFile(path.join(WORK,"rendered","deck-montage.webp"),new Uint8Array(await montage.arrayBuffer()));
const pptx=await PresentationFile.exportPptx(P); await pptx.save(OUT);
console.log(JSON.stringify({slides:P.slides.items.length,out:OUT,montage:path.join(WORK,"rendered/deck-montage.webp")},null,2));
