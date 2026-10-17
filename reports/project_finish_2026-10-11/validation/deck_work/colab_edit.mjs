import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const workspaceDir=path.resolve('reports/project_finish_2026-10-11');
const stage=path.join(workspaceDir,'validation/colab_followup');
const source=path.join(stage,'source.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const replacements=new Map([
 ['Range across measured runs','Selected measured cohort means'],
 ['ρ = 0.27','9 / 17'],
 ['Pooled table, n = 23','Finite cells are synthetic'],
 ['CI crosses 0','Prediction untested'],
 ['[−0.37, 0.88]','Historical outcome mixes metrics'],
 ['Three measured sampling-only GSM8K tensors reproduce ZVF 0.130, 0.190 and 0.155 (pooled 0.1583). The 23-cell association pools heterogeneous records; projection membership is unresolved.','Three measured sampling-only GSM8K tensors reproduce ZVF 0.130, 0.190 and 0.155 (pooled 0.1583). A Colab source audit resolves the mixed provenance behind the historical correlation.'],
 ['Recomputable. Association with held-out accuracy is not distinguishable from zero.','Recomputable. The pooled correlation mixes synthetic rows and outcome definitions, so predictive validity remains unestablished.']
]);
const seen=new Set(),changes=[];
for(const row of (await p.inspect({kind:'textbox',maxChars:120000})).ndjson.split('\n').filter(Boolean).map(JSON.parse)){
 if(replacements.has(row.text)){
  const after=replacements.get(row.text);p.resolve(row.id).text.replace(row.text,after);seen.add(row.text);changes.push({slide:row.slideIndex+1,kind:'textbox',before:row.text,after});
 }
}
if(seen.size!==replacements.size)throw Error('Missing source text');
for(const [i,after] of [
 [2,'RQ1. Source: thesis §6.3.4 and colab_validation/result.json, inputs/manifest.json and execution_receipt.json. The frozen Python analysis ran through Colab CLI on a clean CPU runtime, with identical findings locally. The historical report counted 23 pooled cells, but six missing-ZVF cells were dropped: 17 effective pairs remain, nine synthetic. All 5,540 variance-mitigation rows reproduce the deterministic generator; duplicates are disclosed. The old rho is descriptive of mixed metrics, not held-out predictive validation. The residual n17 analysis uses the same mixed cohort and mixed source outcomes including late training reward. The three GSM8K tensors are sampling-only, G8,200 problems per seed; their pooled ZVF remains0.1583. Equal reward removes the group-relative advantage signal, but KL, auxiliary losses and optimizer state can still affect updates. No new model training occurred in this Colab reanalysis.'],
 [3,'Sources: thesis §9.3 and colab_validation/result.json. RQ1: algebra and diagnostic recomputation are supported; predictive validity is unestablished because the historical pooled analysis mixed synthetic rows and target metrics. This is not evidence that prediction is impossible. RQ2 remains baseline-specific and limited by prompt exposure and critic tuning. RQ3 tests one manifest pair and does not predict score changes. RQ4 evaluates a trained checkpoint without establishing a training delta.']
]){const before=p.slides.items[i].speakerNotes.text;p.slides.items[i].speakerNotes.text=after;changes.push({slide:i+1,kind:'notes',before,after});}
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(stage,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
await fs.writeFile(path.join(stage,'deck_changes.json'),JSON.stringify(changes,null,2));
await fs.writeFile(path.join(stage,'deck_inspection.ndjson'),(await p.inspect({kind:'slide,textbox,table,notes',maxChars:150000})).ndjson);
await fs.mkdir(path.join(stage,'final'),{recursive:true});
const candidatePath=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidatePath);
const skill='/Users/arvind/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir,candidatePath,finalPath:path.join(stage,'final/validated-colab-v1.pptx'),pythonExecutable:'/Users/arvind/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12191695,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['Calibri','Georgia'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},verifyArtifactToolImport:true,receiptPath:path.join(stage,'deck_validation_v1.json')});
console.log(JSON.stringify(result));
