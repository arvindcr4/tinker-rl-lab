import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const workspaceDir=path.resolve('reports/project_finish_2026-10-11');
const stage=path.join(workspaceDir,'validation/deck_work');
const source=path.resolve('reports/final_defense_2026-10-03/TinkerRL_Phase2_Defense_2026-10-03.pptx');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const inspection=await p.inspect({kind:'textbox,notes',maxChars:100000});
let changes=[];
for (const entry of inspection.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x))) {
 if (!entry.text) continue;
 let updated=entry.text.replaceAll('3 October 2026','11 October 2026');
 updated=updated.replace('Solved prompts spend compute and teach nothing','Equal rewards supply no group-relative signal');
 updated=updated.replace('One gap is suggestive. It is not a PPO result.','Same-stack comparison with an untuned PPO critic');
 updated=updated.replace('16 prompts × 8 completions for GRPO, against 128 prompts for PPO','8 prompts × 8 completions for GRPO, against 64 prompts × 1 for PPO');
 updated=updated.replace('the 150 extraction failures come from the token cap with thinking on, not format','150 extraction failures may reflect the token cap with thinking on; per-sample finish reasons were not recorded, so this cause is unproven');
 updated=updated.replace('ZVF is a diagnostic of advantage-signal starvation, not a predictor.','ZVF measures group-relative advantage-signal starvation. Predictive validity was not established. KL, auxiliary losses and optimizer state may still affect updates.');
 if (updated!==entry.text) {
  if (entry.kind==='notes') p.slides.items[entry.slideIndex].speakerNotes.text=updated;
  else p.resolve(entry.id).text.replace(entry.text,updated);
  changes.push({slide:entry.slideIndex+1,kind:entry.kind,before:entry.text,after:updated});
 }
}
p.slides.items[0].speakerNotes.text += '\nEvaluation review copy dated 11 October 2026. Source thesis: reports/project_finish_2026-10-11/thesis/Thesis_Report_ArvindCR_2026-10-11_REVIEW.pdf. No new training or benchmark evaluation was performed for this deck.';
p.slides.items[4].speakerNotes.text += '\nBudget source: platform_hybrid/experiments/results/samestack_gsm8k_cot.json and samestack_gsm8k_cot_full.json. GRPO 8 prompts times 8 completions, PPO 64 prompts times 1, 64 generations per step. This equalizes generations, not prompt exposure or actual token compute.';
for(let i=0;i<p.slides.items.length;i++) {
 const image=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(stage,`after-${i+1}.png`),new Uint8Array(await image.arrayBuffer()));
}
await fs.writeFile(path.join(stage,'changes.json'),JSON.stringify(changes,null,2));
await fs.writeFile(path.join(stage,'final_inspection.ndjson'),(await p.inspect({kind:'slide,textbox,notes',maxChars:100000})).ndjson);
const candidatePath=path.join(stage,'candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const skill='/Users/arvind/.codex/plugins/cache/openai-primary-runtime/presentations/26.1009.11318/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({
 workspaceDir,candidatePath,finalPath:path.join(workspaceDir,'defense/TinkerRL_PES_Defense_2026-10-11.pptx'),
 pythonExecutable:'/Users/arvind/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12191695,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['Calibri','Georgia'],referencePath:source,referenceSha256:crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex')},
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'validation.json')
});
console.log(JSON.stringify(result));
