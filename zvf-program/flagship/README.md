# Variance-starvation flagship experiment

This directory contains the frozen, staged protocol for deciding whether the
completed E1 audit supports a main-track mechanism-plus-controller paper.

The design is intentionally fail-closed. It first proves that TRL and verl
execute the intended objective and controller actions on identical fixtures.
It then screens six policies on the cheaper 1.7B model. The 8B confirmatory
matrix and secondary-stack replication are forbidden unless the preceding
gate passes. Screening and confirmatory seeds are disjoint.

The three honest publication outcomes are:

1. mechanism plus controller, if every gate passes;
2. mechanism-only or negative controller result, if prediction generalizes but
   the controller does not beat static and naive policies;
3. stop, if neither prospective prediction nor control value survives.

`preregistration.json` is the source of truth. A later implementation-freeze
manifest must pin the Git commit, container digest, CUDA/PyTorch versions,
objective-test digest, prompt splits, and reward parsers before any GPU run.
No existing E1 record may be modified or pooled into the new confirmatory
analysis.

The frozen protocol digest is recorded in `preregistration.sha256`; verify it
from this directory with `shasum -a 256 -c preregistration.sha256`.
