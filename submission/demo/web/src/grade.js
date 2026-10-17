// Port of the strict boxed-answer grader in
// reports/project_finish_2026-10-11/validation/samestack_v2/v2_analysis.py (last_boxed, to_num, grade).

export function lastBoxed(t) {
  let i = t.lastIndexOf("\\boxed");
  while (i !== -1) {
    const j = t.indexOf("{", i);
    if (j !== -1) {
      let depth = 0;
      for (let k = j; k < t.length; k++) {
        if (t[k] === "{") depth++;
        else if (t[k] === "}" && --depth === 0) return t.slice(j + 1, k);
      }
    }
    i = i === 0 ? -1 : t.lastIndexOf("\\boxed", i - 1);
  }
  return null;
}

// Python float(): decimal or exponent forms, digit-group underscores, inf/nan; nothing else.
const PY_FLOAT = /^[+-]?(?:(?:\d(?:_?\d)*)?\.?\d(?:_?\d)*(?:[eE][+-]?\d(?:_?\d)*)?|\d(?:_?\d)*\.(?:[eE][+-]?\d(?:_?\d)*)?|inf(?:inity)?|nan)$/i;
function pyFloat(s) {
  if (!PY_FLOAT.test(s)) return null;
  const t = s.replace(/_/g, "").toLowerCase();
  if (/inf/.test(t)) return t.startsWith("-") ? -Infinity : Infinity;
  if (/nan/.test(t)) return NaN;
  return Number(t);
}

export function toNum(s) {
  s = s.replace(/\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\[,;:! ]|\\%|%|\\\$|\$|,|\s/g, "").replace(/\.+$/, "");
  const m = s.match(/^\\d?frac\{(-?[\d.]+)\}\{([\d.]+)\}$/);
  if (m) {
    const a = pyFloat(m[1]), b = pyFloat(m[2]);
    return a === null || b === null || b === 0 ? null : a / b;
  }
  return pyFloat(s);
}

export function grade(text, gold) {
  const b = lastBoxed(text);
  if (b === null) return { correct: 0, status: "no boxed answer", boxed: null };
  const v = toNum(b);
  if (v === null) return { correct: 0, status: "unparsable", boxed: b };
  return { correct: Math.abs(v - Number(gold)) < 1e-6 ? 1 : 0, status: "ok", boxed: b };
}
