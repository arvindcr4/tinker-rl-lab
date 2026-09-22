import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/arvind/Developer/tinker-rl-lab/outputs/public_portfolio_2026-09-05/presentation/E1_E14_Public_Research_Update_v15.pptx'));
await fs.writeFile('.codex-pptx/phase2-review-2026-09-12/source-inspect.ndjson',(await p.inspect({kind:'slide,textbox,table,chart',maxChars:100000})).ndjson);
for(let i=0;i<p.slides.items.length;i++){const b=await p.slides.items[i].export({format:'png',scale:0.6});await fs.writeFile(`.codex-pptx/phase2-review-2026-09-12/source-${i+1}.png`,new Uint8Array(await b.arrayBuffer()));}
console.log('Imported and rendered',p.slides.items.length);
