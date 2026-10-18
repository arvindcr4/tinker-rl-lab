import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Explicit inputs must never silently fall back to historical evidence.
export function demoDataInput(env = process.env) {
  const requested = Object.hasOwn(env, "DEMO_DATA");
  if (requested && !env.DEMO_DATA?.trim()) throw new Error("DEMO_DATA must be a nonempty path");
  const historical = fileURLToPath(new URL("./src/data.json", import.meta.url));
  const source = requested ? resolve(env.DEMO_DATA) : historical;
  let applied = false;
  return {
    name: "demo-data-input", enforce: "pre",
    buildStart() { applied = false; console.info(`[demo-data-input] Embedding ${source}${requested ? "" : " (historical)"}`); },
    resolveId(id, importer) {
      if (importer && resolve(importer.split("?")[0], "..", id) === historical) {
        applied = true;
        return source;
      }
    },
    buildEnd(error) {
      if (!error && requested && !applied) this.error("Requested DEMO_DATA was not applied to the build");
    },
  };
}
export default defineConfig({ plugins: [demoDataInput(), react(), viteSingleFile()] });
