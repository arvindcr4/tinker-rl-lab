// Student t via the regularized incomplete beta; matches v2_analysis.py to float precision.
function lgamma(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5;
  t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (const ci of c) s += ci / ++y;
  return -t + Math.log((2.5066282746310005 * s) / x);
}

function betacf(a, b, x) {
  const tiny = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 3e-14) break;
  }
  return h;
}

function ibeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? (bt * betacf(a, b, x)) / a : 1 - (bt * betacf(b, a, 1 - x)) / b;
}

export function tcdf(t, df) {
  const tail = 0.5 * ibeta(df / (df + t * t), df / 2, 0.5);
  return t >= 0 ? 1 - tail : tail;
}

export function tppf(q, df) {
  let lo = -60, hi = 60;
  for (let i = 0; i < 200; i++) {
    const m = (lo + hi) / 2;
    if (tcdf(m, df) < q) lo = m; else hi = m;
  }
  return (lo + hi) / 2;
}

// Exact two-sided sign-flip p over all 2^n sign assignments.
export function signflip(d) {
  const n = d.length, obs = Math.abs(d.reduce((a, b) => a + b, 0));
  let hit = 0;
  for (let m = 0; m < 1 << n; m++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += ((m >> i) & 1 ? -1 : 1) * d[i];
    if (Math.abs(s) >= obs - 1e-12) hit++;
  }
  return hit / (1 << n);
}

export function paired(d, margin = 0.02) {
  const n = d.length, md = d.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(d.reduce((a, x) => a + (x - md) ** 2, 0) / (n - 1));
  const se = sd / Math.sqrt(n), df = n - 1, tc = tppf(0.975, df), t = md / se;
  return {
    n, md, sd, se, df, t,
    lo: md - tc * se, hi: md + tc * se,
    p: 2 * tcdf(-Math.abs(t), df),
    tost: Math.max(1 - tcdf((md + margin) / se, df), tcdf((md - margin) / se, df)),
    sf: signflip(d),
  };
}

export const zvfCurve = (p, G) => p ** G + (1 - p) ** G;

export const sgn = (x, d = 4) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
export const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
export const pfmt = (p) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));
