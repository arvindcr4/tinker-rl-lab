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

function validDf(df) {
  if (!Number.isFinite(df) || df <= 0) throw new RangeError("degrees of freedom must be positive and finite");
}

function finiteVector(d, minLength = 1) {
  if (!Array.isArray(d) || d.length < minLength || !Array.from(d).every(Number.isFinite)) {
    throw new TypeError(`expected at least ${minLength} finite numeric observations`);
  }
}

export function tcdf(t, df) {
  validDf(df);
  if (typeof t !== "number" || Number.isNaN(t)) throw new TypeError("t must be a number");
  const tail = 0.5 * ibeta(df / (df + t * t), df / 2, 0.5);
  return t >= 0 ? 1 - tail : tail;
}

export function tppf(q, df) {
  validDf(df);
  if (!Number.isFinite(q) || q < 0 || q > 1) throw new RangeError("q must be a probability in [0, 1]");
  if (q === 0) return -Infinity;
  if (q === 1) return Infinity;
  if (q === 0.5) return 0;
  const tail = q < 0.5 ? q : 1 - q;
  let lo = 0, hi = 1;
  while (tcdf(-hi, df) > tail && Number.isFinite(hi)) hi *= 2;
  if (!Number.isFinite(hi)) return q < 0.5 ? -Infinity : Infinity;
  for (let i = 0; i < 200; i++) {
    const m = (lo + hi) / 2;
    if (tcdf(-m, df) > tail) lo = m; else hi = m;
  }
  return (q < 0.5 ? -1 : 1) * (lo + hi) / 2;
}

export const MAX_EXACT_SIGNFLIP_N = 20;

// Exact two-sided sign-flip p over all 2^n sign assignments. Larger samples
// need a separately specified approximation, not a 32-bit shift or an unbounded loop.
export function signflip(d) {
  finiteVector(d, 2);
  if (d.length > MAX_EXACT_SIGNFLIP_N) throw new RangeError(`exact sign-flip is limited to ${MAX_EXACT_SIGNFLIP_N} observations`);
  const n = d.length, obs = Math.abs(d.reduce((a, b) => a + b, 0));
  const magnitude = d.reduce((a, b) => a + Math.abs(b), 0);
  if (!Number.isFinite(magnitude)) throw new RangeError("sign-flip sum is outside the finite numeric range");
  const tolerance = Number.EPSILON * Math.max(magnitude, Number.MIN_VALUE) * n * 4;
  const assignments = 2 ** n;
  let hit = 0;
  for (let m = 0; m < assignments; m++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += ((m >> i) & 1 ? -1 : 1) * d[i];
    if (Math.abs(s) >= obs - tolerance) hit++;
  }
  return hit / assignments;
}

export function paired(d, margin = 0.02) {
  finiteVector(d, 2);
  if (!Number.isFinite(margin) || margin <= 0) throw new RangeError("equivalence margin must be positive and finite");
  const n = d.length, constant = d.every((x) => x === d[0]);
  const md = constant ? d[0] : d.reduce((a, b) => a + b / n, 0);
  const scale = d.reduce((a, x) => Math.max(a, Math.abs(x - md)), 0);
  const sd = scale === 0 ? 0 : scale * Math.sqrt(d.reduce((a, x) => a + ((x - md) / scale) ** 2 / (n - 1), 0));
  if (!Number.isFinite(md) || !Number.isFinite(sd)) throw new RangeError("paired summary is outside the finite numeric range");
  const se = sd / Math.sqrt(n), df = n - 1;
  // The point-mass limit at a null boundary is p=.5, including both TOST boundaries.
  const standardized = (delta) => se === 0 ? (delta === 0 ? 0 : Math.sign(delta) * Infinity) : delta / se;
  const t = standardized(md), tc95 = tppf(0.975, df), tc90 = tppf(0.95, df);
  const difference_ci_95 = [md - tc95 * se, md + tc95 * se];
  const equivalence_ci_90 = [md - tc90 * se, md + tc90 * se];
  const tost_lower_p = tcdf(-standardized(md + margin), df);
  const tost_upper_p = tcdf(standardized(md - margin), df);
  const positive5ppP = tcdf(standardized(md - 0.05), df);
  const sf_available = n <= MAX_EXACT_SIGNFLIP_N;
  return {
    n, md, sd, se, df, t: Number.isFinite(t) ? t : null,
    lo: difference_ci_95[0], hi: difference_ci_95[1],
    p: 2 * tcdf(-Math.abs(t), df),
    tost: Math.max(tost_lower_p, tost_upper_p),
    sf: sf_available ? signflip(d) : null,
    sf_available,
    sf_unavailable_reason: sf_available ? null : `Exact enumeration is limited to ${MAX_EXACT_SIGNFLIP_N} observations`,
    difference_ci_95, equivalence_ci_90, equivalence_margin: margin,
    zero_variance: sd === 0,
    within_margin_unadjusted: equivalence_ci_90[0] > -margin && equivalence_ci_90[1] < margin,
    tost_lower_p, tost_upper_p,
    positive_5pp_test: {
      null_hypothesis: "mean >= +0.05",
      alternative_hypothesis: "mean < +0.05",
      p_value: positive5ppP,
      upper_bound_95: equivalence_ci_90[1],
      reject_at_0_05: positive5ppP < 0.05,
    },
  };
}

// Holm step-down adjustment over the supplied, explicitly defined family.
export function holm(ps) {
  if (ps === null || typeof ps !== "object" || Array.isArray(ps)) throw new TypeError("p-values must be a named object");
  const entries = Object.entries(ps);
  for (const [, p] of entries) {
    if (!Number.isFinite(p) || p < 0 || p > 1) throw new RangeError("Holm p-values must be finite probabilities in [0, 1]");
  }
  entries.sort((a, b) => a[1] - b[1]);
  let running = 0;
  return Object.fromEntries(entries.map(([name, p], i) => {
    running = Math.max(running, Math.min(1, (entries.length - i) * p));
    return [name, running];
  }));
}

export const zvfCurve = (p, G) => p ** G + (1 - p) ** G;

export const sgn = (x, d = 4) => Number.isFinite(x) ? (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d) : "unavailable";
export const pct = (x, d = 1) => Number.isFinite(x) ? (100 * x).toFixed(d) + "%" : "unavailable";
export const pfmt = (p) => Number.isFinite(p) ? (p < 0.001 ? p.toExponential(1) : p.toFixed(3)) : "unavailable";
