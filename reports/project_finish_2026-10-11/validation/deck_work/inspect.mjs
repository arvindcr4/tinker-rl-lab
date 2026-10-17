import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
const root=path.resolve('reports/project_finish_2026-10-11/validation/deck_work');
const p=await PresentationFile.importPptx(await FileBlob.load('reports/final_defense_2026-10-03/TinkerRL_Phase2_Defense_2026-10-03.pptx'));
const info=await p.inspect({kind:'slide,textbox,notes',maxChars:100000});
await fs.writeFile(path.join(root,'source_inspection.ndjson'),info.ndjson);
for(let i=0;i<p.slides.items.length;i++) {
 const image=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(root,`before-${i+1}.png`),new Uint8Array(await image.arrayBuffer()));
}
console.log(p.slides.items.length);
