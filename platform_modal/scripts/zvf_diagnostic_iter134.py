#!/usr/bin/env python3
"""Iter134 review28_v2: provenance-gated retrospective risk analysis.

Historical mixed/synthetic or mismatched-score outputs are superseded.
Use --measured-input and optional --heldout-input; see _review28_risk.py.
No input is silently imputed. Stored historical artifacts are untouched.
"""
from _review28_risk import score, evaluate
from _review28_statistics import auc as auroc


def main():
    from _review28_risk import main as run
    run(134)


if __name__ == "__main__":
    main()
