// Independent JavaScript implementation of boxed-exact-rational-v2.
// Keep the shared tests/fixtures/samestack_grading_v2.json contract in sync with
// samestack_protocol.py and the independently implemented analysis grader.
export const GRADING_SPEC_ID = "boxed-exact-rational-v2";

export function lastBoxed(text) {
  if (typeof text !== "string") return null;
  let found = null;
  for (const match of text.matchAll(/\\boxed\s*\{/g)) {
    const start = match.index + match[0].length;
    let depth = 1;
    for (let index = start; index < text.length; index++) {
      if (text[index] === "{") depth++;
      else if (text[index] === "}" && --depth === 0) {
        found = text.slice(start, index);
        break;
      }
    }
  }
  return found;
}

function rational(numerator, denominator = 1n) {
  if (denominator === 0n) return null;
  if (denominator < 0n) { numerator = -numerator; denominator = -denominator; }
  let a = numerator < 0n ? -numerator : numerator, b = denominator;
  while (b !== 0n) [a, b] = [b, a % b];
  return { numerator: numerator / a, denominator: denominator / a };
}

function divide(numerator, denominator, sign = 1n) {
  if (numerator === null || denominator === null || denominator.numerator === 0n) return null;
  return rational(sign * numerator.numerator * denominator.denominator,
    numerator.denominator * denominator.numerator);
}

// This parser returns a rational, never a rounded floating-point approximation.
// GSM8K uses the numeric coefficient: one currency prefix/wrapper, percent
// suffix, or second/seconds suffix is allowed. Other prose and decorations
// inside fraction operands are rejected; percent does not divide by 100.
export function toNum(value) {
  if (typeof value !== "string" || value.length > 4096) return null;
  value = value.replace(/\\[!,;: ]/g, "").trim();
  let decorations = 0;
  if (value.startsWith("$") && value.endsWith("$") && value.length >= 2) {
    value = value.slice(1, -1).trim();
    decorations++;
  } else if (value.startsWith("\\$") || value.startsWith("$")) {
    value = value.slice(value.startsWith("\\$") ? 2 : 1).trim();
    decorations++;
  }
  const percent = /(?:\\%|%)\s*$/.exec(value);
  if (percent) { value = value.slice(0, percent.index).trim(); decorations++; }
  const unit = /(?:\\text\{\s*(?:second|seconds)\s*\}|\s+(?:second|seconds))\s*$/.exec(value);
  if (unit) { value = value.slice(0, unit.index).trim(); decorations++; }
  return decorations > 1 ? null : numericCore(value);
}

function numericCore(value) {
  const fraction = value.match(/^([+-]?)\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}$/);
  if (fraction) return divide(numericCore(fraction[2].trim()), numericCore(fraction[3].trim()), fraction[1] === "-" ? -1n : 1n);
  if ((value.match(/\//g) || []).length === 1) {
    const [numerator, denominator] = value.split("/");
    return divide(numericCore(numerator.trim()), numericCore(denominator.trim()));
  }
  // Validate the entire atom before removing integer grouping. In particular,
  // neither decimal digits nor exponent digits may contain comma separators.
  const decimal = value.match(/^([+-]?)((?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?|\.\d+)(?:[eE]([+-]?\d+))?$/);
  if (!decimal) return null;
  const [integer, fractional = ""] = decimal[2].replaceAll(",", "").split(".");
  const digits = (integer + fractional).replace(/^0+/, "") || "0";
  const exponent = BigInt(decimal[3] || "0") - BigInt(fractional.length);
  const adjusted = exponent + BigInt(digits.length - 1);
  if (adjusted < -10000n || adjusted > 10000n) return null;
  const coefficient = BigInt(digits) * (decimal[1] === "-" ? -1n : 1n);
  return exponent < 0n ? rational(coefficient, 10n ** -exponent) : rational(coefficient * 10n ** exponent);
}

export function grade(text, gold) {
  const boxed = lastBoxed(text);
  if (boxed === null) return { correct: 0, status: "no boxed answer", boxed };
  const predicted = toNum(boxed), expected = toNum(gold);
  if (predicted === null) return { correct: 0, status: "unparsable", boxed };
  if (expected === null) return { correct: 0, status: "invalid gold", boxed };
  return {
    correct: Number(predicted.numerator === expected.numerator && predicted.denominator === expected.denominator),
    status: "ok", boxed,
  };
}
