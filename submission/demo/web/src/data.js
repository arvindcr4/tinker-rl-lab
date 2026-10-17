import DATA from "./data.json";
import { zvfCurve } from "./stats.js";
import { analysisModel } from "./analysis.js";

export { DATA };
export const SS = DATA.samestack;
export const SD = DATA.stackdiff;
export const MODEL = analysisModel(DATA);
export const ANALYSIS = MODEL.analysis;
export const ARM_NAME = { grpo_g8: "GRPO, G = 8", grpo_g2: "GRPO, G = 2", ppo: "PPO-style (critic)", ppo_8x8: "PPO-style, 8×8 exposure-matched" };
export const SHORT = { grpo_g8: "GRPO G8", grpo_g2: "GRPO G2", ppo: "PPO-style", ppo_8x8: "PPO-style 8×8" };
// Historical/descriptive collection: retain duplicate seed records without overwriting them.
export const byArm = {};
for (const [index, r] of (SS.runs || []).entries()) (byArm[r.arm] ||= {})[index] = r;
export const nSeeds = (a) => new Set(Object.values(byArm[a] || {}).map((r) => r.seed)).size;
export const CONTRASTS = MODEL.contrasts;
export const provisional = () => !MODEL.complete;
export function zvfStats(arm, G) {
  const pts = Object.values(byArm[arm] || {}).flatMap((r) => r.steps || [])
    .filter((p) => Number.isFinite(p.r) && Number.isFinite(p.zvf));
  return { pts, n: pts.length,
    mean: pts.length ? pts.reduce((a, p) => a + p.zvf, 0) / pts.length : null,
    above: pts.length ? pts.filter((p) => p.zvf > zvfCurve(p.r, G) + 1e-9).length / pts.length : null };
}
