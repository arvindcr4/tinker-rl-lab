import React, { useState } from "react";
import { ARM_NAME, SS, nSeeds, zvfStats } from "../data.js";
import { pct, zvfCurve } from "../stats.js";

const ARM_G = { grpo_g8: 8, grpo_g2: 2 };

function ZvfChart({ pts, G, dashed }) {
  const W = 520, H = 330, L = 44, R = 12, T = 12, B = 40;
  const x = (v) => L + (W - L - R) * v, y = (v) => T + (H - T - B) * (1 - v);
  let d = "";
  for (let i = 0; i <= 100; i++) d += `${i ? "L" : "M"}${x(i / 100).toFixed(1)} ${y(zvfCurve(i / 100, G)).toFixed(1)}`;
  const mid = (T + H - B) / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Zero-variance fraction against mean reward">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
          <text x={L - 6} y={y(v) + 4} textAnchor="end">{v}</text>
          <text x={x(v)} y={H - 22} textAnchor="middle">{v}</text>
        </g>
      ))}
      <text x={(L + W - R) / 2} y={H - 4} textAnchor="middle">mean reward in the training batch</text>
      <text x={12} y={mid} transform={`rotate(-90 12 ${mid})`} textAnchor="middle">ZVF</text>
      {pts.map((p, i) => <circle key={i} cx={x(p.r)} cy={y(p.zvf)} r={3.2} fill="var(--grpo)" fillOpacity={0.45} />)}
      <path d={d} fill="none" stroke="var(--muted)" strokeWidth={2} strokeDasharray={dashed ? "5 4" : undefined} />
    </svg>
  );
}

export default function Zvf() {
  const [arm, setArm] = useState("grpo_g8");
  const [G, setG] = useState(8);
  const Garm = ARM_G[arm], s = zvfStats(arm, Garm);
  return (
    <section className="view">
      <div className="panel">
        <h2>When the signal disappears</h2>
        <p className="small">If every answer in a group gets the same reward, every advantage is zero and that prompt contributes no gradient. The share of such groups is the zero-variance fraction (ZVF). If every prompt had the same success rate p, a group of G answers would be all-same with probability p<sup>G</sup> + (1−p)<sup>G</sup>. That identity alone produces a U-shape in accuracy, which is why the thesis no longer treats the U-shape as a finding in its own right.</p>
      </div>
      <div className="grid2">
        <div className="panel">
          <div className="row between">
            <h3 style={{ margin: 0 }}>ZVF against batch reward</h3>
            <div className="row">
              {Object.keys(ARM_G).map((a) => (
                <button key={a} type="button" aria-pressed={a === arm} onClick={() => { setArm(a); setG(ARM_G[a]); }}>G = {ARM_G[a]}</button>
              ))}
            </div>
          </div>
          <div className="chart-box"><ZvfChart pts={s.pts} G={G} dashed={G !== Garm} /></div>
          <div className="legend">
            <span><i style={{ background: "var(--muted)" }} />p<sup>G</sup> + (1−p)<sup>G</sup>, same difficulty for all prompts</span>
            <span><i style={{ background: "var(--grpo)" }} />measured training steps</span>
          </div>
        </div>
        <div className="panel finding">
          <h3>Reading the measured steps</h3>
          <dl className="stat">
            <dt>Arm</dt><dd>{ARM_NAME[arm]}</dd>
            <dt>Training steps</dt><dd>{s.n} ({nSeeds(arm)} seeds × {SS.setup.n_steps})</dd>
            <dt>Mean ZVF</dt><dd>{pct(s.mean)}</dd>
            <dt>Steps above G = {Garm} curve</dt><dd>{pct(s.above, 0)}</dd>
          </dl>
          <p className="small">
            {s.above > 0.5
              ? "Most measured steps sit above the curve. GSM8K prompts differ in difficulty, so easy prompts come back all-correct and hard ones all-wrong more often than one shared success rate would predict. "
              : "The measured steps track the curve closely. "}
            ZVF is therefore mostly set by reward level and prompt mix, which is why the thesis treats it as telemetry rather than as a predictor of final accuracy.
          </p>
          <label className="small" htmlFor="zvf-g">Curve group size G: <b className="mono">{G}</b></label>
          <input type="range" id="zvf-g" min="2" max="16" step="1" value={G} onChange={(e) => setG(+e.target.value)} />
        </div>
      </div>
    </section>
  );
}
