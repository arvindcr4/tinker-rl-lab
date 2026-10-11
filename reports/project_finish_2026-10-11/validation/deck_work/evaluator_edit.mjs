import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const workspaceDir=path.resolve('reports/project_finish_2026-10-11');
const stage=path.join(workspaceDir,'validation/evaluator_review');
const source=path.join(stage,'prior-delivered.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const originalNotes=p.slides.items.map(slide=>slide.speakerNotes.text);
const replacements=new Map([
 ['Hold the stack fixed. Change one factor.','Research question and project scope'],
 ['If the stack also moves, the number is not an algorithm result. That rule is applied twice.','Which differences survive a recorded, controlled comparison, and which remain confounded?'],
 ['What does GRPO do?','Diagnostics and controlled studies'],
 ['Libraries, scales, group sizes, length, and a reporting standard. Only Tinker and TRL produced completed runs, and they used different base checkpoints.','P1–P7 study training signal, comparison design and reporting. P8 is a supervised synthetic-fraud side study. Framework runs completed on Tinker and TRL with different checkpoints.'],
 ['One Tinker-trained checkpoint, 40 steps, fourteen native suites. Scores describe that checkpoint. They are not a post-training delta.','One Tinker-trained checkpoint, 40 steps, fourteen evaluation lanes. Every score names its actual scope. Unavailable original contracts remain blocked.'],
 ['vs held-out outcome, n = 23','Pooled table, n = 23'],
 ['Three stored GSM8K tensors (Qwen3-8B, sampling only) reproduce 0.130, 0.190 and 0.155, pooled 0.1583. After mean reward is regressed out, r = +0.21, n = 17, and that interval crosses zero.','Three measured sampling-only GSM8K tensors reproduce ZVF 0.130, 0.190 and 0.155 (pooled 0.1583). The 23-cell association pools heterogeneous records; projection membership is unresolved.'],
 ['Four questions, at the strength of the evidence','Research questions and findings'],
 ['What the evidence does not support','Implementation: a recorded manifest comparison'],
 ['No scaling law','1. Retained inputs'],
 ['Flat fit over about 2.4 orders of magnitude. Scale is collinear with the recipe. Single-seed anchors.','Reward tensors feed ZVF telemetry. Run manifests record loss settings, model, sampler, split and missing fields.'],
 ['No length-bias finding','2. Actual DAPO records'],
 ['In the controlled comparison, completion length drifts shorter. GRPO and Dr. GRPO are indistinguishable on held-out gain.','Open record: KL coefficient = 0.0. Managed record: KL coefficient = null (unreported). Equal algorithm labels do not resolve that difference.'],
 ['No group-size optimum','3. Working CLI result'],
 ['G = 2 through 16 are not separable on the three-seed sweep. G = 16 versus G = 2 fails equivalence at ±0.02 (TOST p = 0.056).','stackdiff returns R5: a differing field is unreported. This is a reporting-risk flag, not a predicted score change.'],
 ['No decisive audit headline','4. Recorded outputs'],
 ['Seven audited headlines: none decisive, one suggestive (H7, iter-136 efficiency). Holm on that family of eight moves the p to 0.25. That is not the +5.0 pp gap.','The registry retains the comparison. The campaign ledger binds scores to scope, coverage and terminal state. Missing evidence stays visible.'],
 ['Campaign scores are checkpoint scores','Campaign scores by stated scope'],
 ['Do not average these rows. Denominators count different things. The paired base-versus-trained comparison, 5 to 151 items per lane, is inconclusive in every lane. Original contracts and replacement scopes stay in separate columns.','Each row names its original or replacement scope. Denominators count different things, so no cross-suite average is reported. Small paired base-versus-trained comparisons remain inconclusive.'],
 ['C1 draws a boundary. It does not repair the score.','C1: frozen endpoint and reference audit'],
 ['17.19%. Wilson interval 9.88–28.21%. Parser v2. Not changed after review.','After 8 parser-zero answers, at least one reference match in 24 fresh draws.\n\nDeterministic first-64 cohort; 17.19%. The interval is descriptive.'],
 ['What I will defend as the contribution','Individual contribution and inherited foundation'],
 ['A diagnostic','Signal diagnostics'],
 ['ZVF and pairwise contrast density, recomputable from stored tensors, with the association stated at the width of its interval.','I implemented and audited ZVF and contrast-density reporting from stored rewards. The group-normalization algebra is inherited.'],
 ['A minimum report','Reporting implementation'],
 ['Eight items: seven manifest fields and one evaluation item. A label without its stack is not an experimental specification.','I developed the eight-item report, machine-readable registry and stack comparison. Slide 6 shows the implemented R5 check.'],
 ['A ledger with gates','Campaign and ownership'],
 ['Every campaign lane has a terminal state and a reopen condition. Missing, simulated, and measured results stay distinct.','Semester 3: shared harness and baseline integrations. My Semester 4: research studies, campaign ledger and thesis. Results retain their evidence limits.'],
 ['The four pushes I expect','Limitations and the next controlled experiment'],
 ['No family reaches Henderson’s ten seeds. Single-seed Tinker figures are descriptive. The open-source contrasts use three to eight.','Open-source contrasts use three to eight seeds. Single-seed Tinker results are descriptive. Narrow effects need a larger, powered study.'],
 ['Closed trainer','Execution visibility'],
 ['Tinker’s server-side loss, normalisation and hardware are not visible. Those results are the platform’s GRPO, not an abstract algorithm.','Some client custom-loss loops are inspectable. Historical managed losses, server execution and hardware remain incompletely recorded.'],
 ['Power','Next comparison'],
 ['Inconclusive is the result at this n. I will not upgrade it to “no effect” in the room.','Tune and warm up the PPO critic. Match prompt exposure and token budgets, lock evaluation tasks, and increase paired seeds.'],
 ['Ninety seconds, then stop','Offline demonstration'],
 ['demo.sh --self-test\ndemo.sh\n\nOpen the HTML file directly if the port is busy.\n\nIf the SHA fails, show the failure. Do not skip it.','From the repository or ZIP root:\n\ncd submission/demo\n./demo.sh --self-test\n./demo.sh\n\nOpen output/demo_report.html.\nIf integrity fails, show the failure.'],
 ['The measurements are informative.\nThe algorithmic levers, at this budget,\nare noise-limited.','An auditable workflow for RL studies\nwith explicit evidence limits'],
 ['A label without its stack is not a result I will defend.','Delivered: recomputable diagnostics, an inspectable registry and a campaign ledger. The strongest training contrast remains baseline-specific.']
]);
const seen=new Set(), changes=[];
for(const row of (await p.inspect({kind:'textbox',maxChars:120000})).ndjson.split('\n').filter(Boolean).map(JSON.parse)) {
 if(replacements.has(row.text)) {
  const after=replacements.get(row.text);
  p.resolve(row.id).text.replace(row.text,after); seen.add(row.text);
  changes.push({slide:row.slideIndex+1,kind:'textbox',before:row.text,after});
 }
}
const missed=[...replacements.keys()].filter(s=>!seen.has(s));
if(missed.length) throw Error('Missing source text: '+JSON.stringify(missed));
const table=p.resolve('tb/bm98jid4');
const tableChanges=[
 [1,0,'VerilogEval (original)'],[1,2,'150/183 failures were extraction failures. Token-cap causality is unquantified.'],
 [2,0,'Omni-MATH (replacement)'],[3,0,'SWE-bench Pro (original)'],
 [4,0,'WebArena (replacement)'],[5,0,'AgentDojo benign (replacement)']
];
for(const [r,c,value] of tableChanges) {table.cells.set(r,c,value);changes.push({slide:7,kind:'table',row:r,column:c,after:value});}
p.slides.items[0].speakerNotes.text='Open with the problem and the concrete deliverable. I developed a workflow to measure group-relative learning signal, disclose comparison conditions and bind benchmark results to their scope. Semester 3 supplied the shared harness; Semester 4 is my individual continuation. No new training result was generated by this evaluator revision. Sources: thesis §1.4–1.6 and PROJECT_HISTORY.md.';
p.slides.items[1].speakerNotes.text='Source: thesis §1.4–1.6. P1–P7 are RL diagnosis/reporting studies; P8 is a supervised synthetic-fraud side study. Fourteen lanes do not mean fourteen completed original contracts. A common checkpoint does not eliminate serving, parser or scope differences. No complete seven-library comparison is claimed.';
p.slides.items[2].speakerNotes.text += '\nThe n=23 table pools heterogeneous records and membership of simulation-projection rows is unresolved. It is not a separately established measured-only cohort. This is a descriptive association from that table, not validated predictive performance.';
p.slides.items[7].speakerNotes.text=p.slides.items[7].speakerNotes.text.replace(/eight equal wrong numbers/gi,'eight parser-zero answers');
p.slides.items[7].speakerNotes.text += '\nFrozen event: after eight initial parser-zero answers, at least one of 24 fresh responses matches the reference under parser v2. Cohort: deterministic first 64 eligible questions. Wilson 95% interval 9.88–28.21% is descriptive, not a population recovery estimate. The assistant-reviewed three numerical witnesses are separate.';
p.slides.items[5].speakerNotes.text='Sources: thesis §5.6 and §7.2; platform_hybrid/registry/query.py; entries/colab-open_dapo_e3.json; entries/tinker_dapo_qwen3.5-4b_gsm8k.json. The exact CLI was rerun locally on 11 October 2026 and returned R5 with process exit code 5, by design. Example: reference_kl.kl_beta is 0.0 vs null. Other differences also contribute. The records describe different tasks and incomplete configurations, so this is an implemented provenance comparison, not an outcome-controlled experiment. Registry records can contain internal inconsistencies; stackdiff does not authenticate upstream execution or resolve them.';
p.slides.items[6].speakerNotes.text+='\nVisible table corrected: original vs replacement scope names are explicit and the E11 causal attribution is withdrawn. Source: thesis Chapter 8 and Appendix D. No stored score changed.';
p.slides.items[8].speakerNotes.text='Sources: PROJECT_HISTORY.md, platform_hybrid/sem 4 work/PROVENANCE.md, thesis §1.6 and §9.3. The six-person Semester 3 project supplied the shared runtime and baseline work. My individual Semester 4 contribution is the research direction, diagnostic/reporting studies, registry/controller analyses, expanded evaluation and thesis. Internal reports P1–P8 are not eight accepted publications. AI assistance is declared. The minimum reporting standard is a bounded integration contribution, not invention of group normalization.';
p.slides.items[9].speakerNotes.text='Thesis §9.2–9.4. Client-side custom-loss loops can be inspected; missing historical execution and managed internals are different limits. No fixed seed threshold guarantees power. The next comparison needs a protocol and power rationale before results are observed. These steps are future work, not measured gains.';
p.slides.items[10].speakerNotes.text='From repository or extracted ZIP root: cd submission/demo; ./demo.sh --self-test; ./demo.sh. Open output/demo_report.html. The seven standard-library tests and default mode require no network. The 68.75% dashboard mean is from 80 recorded binary rewards in one artifact. It is not benchmark accuracy. The optional registry example is documented in the evaluator guide; its expected exit code is 5 for R5. No missing fallback folder is required.';
p.slides.items[11].speakerNotes.text='Close on delivered implementation and the strongest bounded finding: +5.0 percentage points against an untuned PPO baseline in five paired seeds. The exact sign-flip p-value is 0.0625, and prompt exposure differs. The next work is a tuned, exposure-matched, larger paired study. Invite questions without adding a new result.';
for(let i=0;i<p.slides.items.length;i++) {
 if(originalNotes[i]!==p.slides.items[i].speakerNotes.text) changes.push({slide:i+1,kind:'notes',before:originalNotes[i],after:p.slides.items[i].speakerNotes.text});
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(stage,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
await fs.writeFile(path.join(stage,'deck_changes.json'),JSON.stringify(changes,null,2));
await fs.writeFile(path.join(stage,'deck_inspection.ndjson'),(await p.inspect({kind:'slide,textbox,table,notes',maxChars:150000})).ndjson);
const candidatePath=path.join(stage,'candidate.pptx'); await(await PresentationFile.exportPptx(p)).save(candidatePath);
const skill='/Users/arvind/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath:path.join(stage,'final/validated-v4.pptx'),pythonExecutable:'/Users/arvind/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12191695,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['Calibri','Georgia'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(stage,'deck_validation_v4.json')});
console.log(JSON.stringify(result));
