import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { viteSingleFile } from "vite-plugin-singlefile";

// Single self-contained HTML so the defense demo opens from disk with no server or network.
export default defineConfig({ plugins: [react(), viteSingleFile()] });
