# Reviewer README

## Quick Start

1. **Extract the bundle:**
   ```bash
   tar -xzf code.tar.gz
   ```

2. **Run claim verification:**
   ```bash
   cd tinker-rl-lab-anon
   python3 scripts/verify_claims_offline.py
   ```

3. **Read the paper:** Start with `paper_anon.pdf`

## Key Documents

| Document | Purpose |
|----------|---------|
| `paper_anon.pdf` | Compact diagnostic-audit paper |
| `REVIEWER_VERIFICATION.md` | Claim → Evidence → Command mapping |
| `EVAL_PROTOCOL.md` | Dataset splits, reward parsers, claim status |
| `FIGURE_PROVENANCE.md` | Figure generation scripts and inputs |
| `SOURCE_PRECEDENCE.md` | Discrepant values explained |
| `VERSION.json` | Bundle version and checksums |

## Verification Commands

```bash
# All offline claim checks
python3 scripts/verify_claims_offline.py

# Specific claim check
python3 scripts/verify_claims_offline.py --claim qwen3_8b_headline_reward

# List all available checks
python3 scripts/verify_claims_offline.py --list
```

## What This Artifact Shows

1. **ZVF/GU as triage diagnostics** for whether GRPO has a learning signal
2. **Training reward ≠ held-out capability** (GSM8K held-out: 82.0% → 83.3%, p=0.26)
3. **Algorithm labels are under-specified** (PPO/GRPO ordering reverses across model families)
4. **Stack identifiability** (same nominal config gives different results)

## What This Artifact Does NOT Claim

- GRPO universally improves reasoning
- ZVF predicts final performance
- PPO is inferior to GRPO (or vice versa)
- Held-out MATH or HumanEval improvement

See `REVIEWER_VERIFICATION.md` for the full disclaimer table.

## Reproducibility Notes

- This is a blind-review package (anonymous)
- Full author-identified report is NOT included
- Tinker API runs cannot be reproduced (closed-source backend)
- Code and supporting data archives are included for inspection

For checksum verification:
```bash
sha256sum -c checksums.sha256
```
