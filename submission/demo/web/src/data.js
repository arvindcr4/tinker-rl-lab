import DATA from "./data.json";
import { paired, zvfCurve } from "./stats.js";

export { DATA };
export const SS = DATA.samestack;
export const SD = DATA.stackdiff;

export const ARM_NAME = { grpo_g8: "GRPO, G = 8", grpo_g2: "GRPO, G = 2", ppo: "PPO (critic)", ppo_8x8: "PPO, 8×8 exposure-matched" };
export const SHORT = { grpo_g8: "GRPO G8", grpo_g2: "GRPO G2", ppo: "PPO", ppo_8x8: "PPO 8×8" };

export const byArm = {};
for (const r of SS.runs) (byArm[r.arm] ||= {})[r.seed] = r;
export const nSeeds = (a) => Object.keys(byArm[a] || {}).length;

function contrast(a, b) {
  const A = byArm[a] || {}, B = byArm[b] || {};
  const seeds = Object.keys(A).filter((s) => s in B).map(Number).sort((x, y) => x - y);
  const diffs = seeds.map((s) => A[s].post - B[s].post);
  return { a, b, seeds, diffs, stats: seeds.length >= 2 ? paired(diffs) : null };
}

export const CONTRASTS = [["grpo_g8", "ppo"], ["grpo_g8", "ppo_8x8"], ["grpo_g8", "grpo_g2"], ["grpo_g2", "ppo"]]
  .map(([a, b]) => contrast(a, b))
  .filter((c) => c.stats);

export function zvfStats(arm, G) {
  const pts = Object.values(byArm[arm] || {}).flatMap((r) => r.steps);
  const n = pts.length || 1;
  return {
    pts,
    n: pts.length,
    mean: pts.reduce((a, p) => a + p.zvf, 0) / n,
    above: pts.filter((p) => p.zvf > zvfCurve(p.r, G) + 1e-9).length / n,
  };
}

export const provisional = () => nSeeds("ppo") < SS.seeds_planned || !nSeeds("ppo_8x8");
