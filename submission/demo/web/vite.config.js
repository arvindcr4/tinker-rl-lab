import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";
import { resolve } from "node:path";

// DEMO_DATA allows verification of a fresh export without changing checked-in historical data.
const externalData = process.env.DEMO_DATA;
export default defineConfig({ plugins: [
  { name: "demo-data-input", enforce: "pre", resolveId(id, importer) {
    if (externalData && id === "./data.json" && importer?.endsWith("/src/data.js")) return resolve(externalData);
  } },
  react(), viteSingleFile(),
] });
