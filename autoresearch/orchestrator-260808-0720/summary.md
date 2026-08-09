# Autoresearch summary: Pavlov domain campaign

The local design contract now maps all 53 Pavlov's List companies into all 16
published domain families. The fail-closed validator reports 12 training suites
and 14 held-out primary evaluation suites. GSM8K is calibration-only and math is
capped at 5% of the training mixture.

The first validation pass exposed missing finance training and missing primary
math evaluation. The second pass added API-Bank RLVR, OpenR1 Math, and private
FrontierMath evaluation, after which all eight contract tests passed.

The final implementation adds a non-launching deterministic campaign-manifest
builder. All 12 contract and manifest tests pass; the preview covers all 53
companies and remains blocked without launching any job.

No scientific or product-use result is claimed. Paid jobs remain blocked until
the user provides an explicit maximum dollar cap, licenses are checked, and
immutable train/eval revisions are frozen.
