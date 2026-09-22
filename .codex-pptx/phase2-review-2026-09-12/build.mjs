import fs from 'node:fs/promises';import path from 'node:path';import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='/Users/arvind/Developer/tinker-rl-lab', tmp=path.join(root,'.codex-pptx/phase2-review-2026-09-12'),out=path.join(root,'outputs/PES_Phase2_Review_2026-09-12');
const source=path.join(root,'outputs/public_portfolio_2026-09-05/presentation/E1_E14_Public_Research_Update_v15.pptx');
const score=JSON.parse(await fs.readFile(path.join(out,'e14_scoring_recheck.json'))),e1=JSON.parse(await fs.readFile(path.join(out,'e1_metadata_recheck.json')));
if(score.correct!==2271||score.accepted!==4426||score.skipped!==2||score.strict_score!==null||e1.audit.verified_image_configs!==300)throw Error('Evidence mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const records=(await p.inspect({kind:'slide,textbox,table,chart,notes',maxChars:300000})).ndjson.split('\n').filter(Boolean).map(JSON.parse);
const shapes={
'sh/k3yl0zql':'PES M.Tech Phase 2 review\nE1–E14 public benchmark results',
'sh/7qp4be9c':'Arvind C R  ·  PES2PGE24DS140  ·  12 September 2026',
'sh/cza94vmx':'4/14',
'sh/d0jax03i':'native results, including E14 with exclusions',
'sh/3ah8rqlg':'3 strict completions. E14 official result excludes 2 reports.',
'sh/0ba143al':'E1: all 300 image configurations verified. Native test runs remain pending.',
'sh/mdonql4z':'Omni-MATH: 51.31% official accuracy',
'sh/0b65obm9':'2,271 correct / 4,426 accepted reports.\n2 truncated reports excluded from the original 4,428.',
'sh/nex4jq5k':'Strict full-suite score remains unavailable. Saved native scoring reproduced today.',
'sh/q50nydsj':'3 strict completions and an E14 result with 2 exclusions',
'sh/bq9orito':'E1: all 300 image configurations verified today. Native test execution remains pending.',
'sh/o3i5w3ad':'E5: 698 documents and 97 initializations ready. E13: 255 model episodes still pending.',
'sh/p4r6p8by':'E14: 51.31% official accuracy. Full archive audit needs a missing historical ZIP.',
};
for(const[id,text]of Object.entries(shapes)){const r=records.find(x=>x.id===id);if(!r)throw Error(id);if(r.text.includes("\n")){const a=r.text.split("\n"),b=text.split("\n");if(a.length!==b.length)throw Error("Line mismatch");a.forEach((v,i)=>p.resolve(id).text.replace(v,b[i]));}else{p.resolve(id).text.replace(r.text,text);}}
const cells=[['tb/g7uhcbqt',3,0,'Judge responses'],['tb/nad4jits',5,2,'746 paper tasks'],['tb/4vqpgvyl',4,1,'51.31% (2,271/4,426)¹'],['tb/fu5cvqds',1,3,'300/300 image configs verified'],['tb/nad4jits',7,3,'51.31% official¹'],['tb/g7uhcbqt',3,1,'4,428 / 4,428 judge responses'],['tb/g7uhcbqt',4,0,'Official / strict score'],['tb/g7uhcbqt',4,1,'51.31% / unavailable'],['tb/nypc3i5c',4,1,'KbsdJames/Omni-Judge, Modal A100 BF16'],['tb/nypc3i5c',5,0,'Original sampler today'],['tb/nypc3i5c',5,1,'Checkpoint not found in live metadata lookup'],['tb/j2xwnmtg',4,2,'4,426 accepted, 2 excluded']];
for(const[id,row,col,text]of cells)p.resolve(id).cells.set(row,col,text);
// Explicitly disclose the same denominator wherever the official result appears.
p.resolve('sh/cf2tcr61').text.replace(records.find(x=>x.id==='sh/cf2tcr61').text,'¹ E14: 4,426 accepted, 2 excluded. E12: 284 paper-task assets missing.\nE13: 255 model episodes pending.');
let scripts=JSON.parse(await fs.readFile(path.join(root,'.codex-run/public_deck_build/v15/content.json'))).scripts;
scripts[0][1]='Good morning sir. This is my Phase 2 progress review. I evaluated Qwen3.6-35B-A3B with the final Pavlov adapter trained through Tinker, using seed 809. The current work covers fourteen public benchmarks. I will show the results, the evaluation conditions, and the work still pending.';
scripts[2][1]='There are three strictly complete suites and one additional official result with parser exclusions. LAB-Bench is 450 out of 1,967, or 22.88 percent. AgentDojo benign utility is 88 out of 97, or 90.72 percent. VerilogEval is 129 out of 312, or 41.35 percent. Omni-MATH is 2,271 out of 4,426 accepted reports, or 51.31 percent. Two reports were excluded by the original parser, so the strict 4,428-report score remains unavailable.';
scripts[3][1]='Today I resolved the remaining six SWE-bench Multilingual image configurations. All 300 metadata records now pass the original strict audit. This completes image metadata recovery, while image execution and model patches remain pending. CORE-Bench has one of 45 original capsules prepared. MLAgentBench still needs its released evaluator and weights. BankerToolBench needs the actual banking deliverables and grading. Tau3 Banking has all 698 documents indexed and all 97 native initializations checked, but no policy episodes yet. WebArena still needs its live sites. CyberGym still needs the native task payload and verifier.';
scripts[4][1]='This table covers the remaining public suites. LAB-Bench, AgentDojo and VerilogEval have complete native results. Omni-MATH now has an official result with two parser exclusions. MLDevBench still needs fourteen original starting workspaces and all 34 runs. VisualAgentBench lacks 284 original task assets. BALROG has local environment checks but all 255 model episodes remain pending.';
scripts[8][1]='Omni-MATH has 4,428 fixed policy answers and 4,428 saved judge responses. The original Omni-Judge parser accepts 4,426 reports and excludes two truncated reports. It marks 2,271 accepted answers correct, giving 51.31 percent. Today I reran the unchanged native scorer, checked the saved file hashes, and matched every one of the 4,428 saved task dispositions. I did not retry the excluded reports or change their grades. The expanded native-input audit now verifies 42,408 files. Historical budget references resolve to byte-identical saved snapshots; unrelated temporary log archives remain explicitly labeled as non-input historical records. The strict full-suite score remains unavailable.';
scripts[9][1]='The policy is the exact Qwen3.6 base with the final Pavlov adapter. LAB-Bench, AgentDojo and Omni-MATH answers used the merged BF16 weights. VerilogEval used the original Tinker sampler. Omni-Judge is the separate math grader, with the later grading campaign on Modal A100. A fresh metadata request today still returns checkpoint not found for the original sampler. No numerical parity between serving backends is claimed.';
scripts[11][1]='Each score has its own meaning and denominator. LAB-Bench measures question accuracy. AgentDojo measures benign task completion. VerilogEval measures pass at one over two task framings. Omni-MATH uses the separate judge and its accepted-report denominator. These scores should not be averaged or interpreted as improvement over an unmeasured baseline.';
scripts[12][1]='The main update is that the original Omni-MATH scoring now reproduces locally, and SWE-bench image metadata recovery is complete for all 300 tasks. Three suites meet the full strict requirements. Omni-MATH has an official result with two exclusions. The other ten public suites still need model execution or missing native assets. Tau3 and BALROG have setup progress, but setup checks do not count as policy outcomes. My immediate next step is to restore the exact actor and native Linux runtime, then run the ready workloads under the agreed resource limit.';
const refs=[path.join(out,'finish/e14/native_input_review_v4.json'),source,path.join(out,'e14_scoring_recheck.json'),path.join(out,'e1_metadata_recheck.json'),path.join(root,'outputs/public_portfolio_2026-09-05/results.json')];
const sources=await Promise.all(refs.map(async f=>({path:f,sha256:crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex')})));
const changed=[1,3,4,5,9,10,12,13];
for(const n of changed)p.slides.items[n-1].speakerNotes.textFrame.setText(scripts[n-1][1]+'\n\nSources:\n'+sources.map(x=>x.path+' SHA256 '+x.sha256).join('\n'));
const verify=(await p.inspect({kind:'textbox',maxChars:300000})).ndjson.split('\n').filter(Boolean).map(JSON.parse);for(const[id,text]of Object.entries(shapes)){if(verify.find(r=>r.id===id)?.text!==text)throw Error('Text update failed '+id);}
await(await PresentationFile.exportPptx(p)).save(path.join(tmp,'artifact-draft.pptx'));
await fs.writeFile(path.join(tmp,'after.ndjson'),(await p.inspect({kind:'slide,textbox,table,chart,notes',maxChars:300000})).ndjson);
await fs.writeFile(path.join(tmp,'content.json'),JSON.stringify({sources,changed,scripts,score},null,2));
await fs.writeFile(path.join(out,'Talking_Script.txt'),'PES Phase 2 review — 12 September 2026\n\n'+scripts.map(([title,t],i)=>`Slide ${i+1}: ${title}\n${t}`).join('\n\n'));
console.log('Authored 13 editable slides and speaking script');
