import fs from "node:fs/promises";
import path from "node:path";

const root = "/Users/arvind/Developer/tinker-rl-lab/.codex-pptx/e1-e14-status-2026-08-22";
const inspectDir = path.join(root, "source-progress/template-inspect");
const lines = [];
for (let slide = 1; slide <= 11; slide += 1) {
  const stem = `source-slide-${String(slide).padStart(2, "0")}.layout.json`;
  const layout = JSON.parse(await fs.readFile(path.join(inspectDir, "layouts", stem), "utf8"));
  lines.push({ kind: "slide", id: layout.slide.id, slide, title: `Source slide ${slide}` });
  for (const element of layout.elements) {
    lines.push({
      kind: element.text == null ? "shape" : "textbox",
      id: element.aid,
      slide,
      name: element.name,
      text: element.text,
      textPreview: element.textPreview,
      bbox: element.bbox,
    });
  }
}
await fs.writeFile(
  path.join(inspectDir, "template-inspect.ndjson"),
  `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`,
  "utf8",
);

const roles = [
  "opening status thesis",
  "current progress summary",
  "execution timeline",
  "evidence metrics",
  "E1-E14 status map",
  "E1 full-suite evidence",
  "E2 and E4 evidence comparison",
  "remaining blocker analysis",
  "Tinker and Modal execution summary",
  "next-action decision flow",
  "bottom-line synthesis",
];

const outputSlides = roles.map((narrativeRole, index) => {
  const slide = index + 1;
  const sourceElementIds = lines
    .filter((record) => record.kind === "textbox" && record.slide === slide)
    .map((record) => record.id);
  return {
    outputSlide: slide,
    sourceSlide: slide,
    narrativeRole,
    reuseMode: "duplicate-slide",
    editTargets: [{ action: "rewrite", sourceElementIds }],
  };
});

await fs.writeFile(
  path.join(root, "template-frame-map.json"),
  `${JSON.stringify({ outputSlides, omittedSourceSlides: [] }, null, 2)}\n`,
  "utf8",
);
