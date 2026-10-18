"""Render summary.jsonl -> markdown table rows."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print("| model | condition | n | acc (training reward) [95% Wilson] | acc (strict boxed) [95% Wilson] | truncation | parse fail (no \\boxed) | think unclosed | mean tokens | gen wall-clock (s) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for s in rows:
    lo, hi = s["acc_campaign_wilson95"]; blo, bhi = s["acc_boxed_wilson95"]
    tu = "-" if s["think_unclosed_rate"] is None else f"{s['think_unclosed_rate']:.3f}"
    print(f"| {s['model']} | {s['cond']} | {s['n']} | {s['acc_campaign']:.3f} [{lo:.3f}, {hi:.3f}] | {s['acc_boxed']:.3f} [{blo:.3f}, {bhi:.3f}] | "
          f"{s['trunc_rate']:.3f} | {s['parse_fail_rate_no_boxed']:.3f} | {tu} | {s['mean_tokens']:.0f} | {s['gen_wall_s']:.0f} |")
