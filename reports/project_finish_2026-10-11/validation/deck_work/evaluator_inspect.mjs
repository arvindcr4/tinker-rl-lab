import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('reports/project_finish_2026-10-11/defense/TinkerRL_PES_Defense_2026-10-11.pptx'));
console.log((await p.inspect({kind:'table',maxChars:20000})).ndjson);
console.log((await p.inspect({kind:'layout',maxChars:3000})).ndjson);
