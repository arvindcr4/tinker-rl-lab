# Colab source audit

On 11 October 2026, the installed Colab CLI ran the frozen analysis in an ephemeral
CPU runtime. The remote result matched the local result within 1e-12. The CLI
confirmed that no runtime remained active after the job. This is an exploratory
reanalysis of existing numeric records, not new model training or benchmark scoring.

## What it improved

| Historical claim | Reproduced source evidence | Consequence |
|---|---|---|
| Correlation over 23 cells | 23 candidate cells, six missing ZVF, 17 effective pairs | The displayed sample size was wrong. |
| Measured-only association | Nine of the 17 finite cells pool 45 synthetic trajectories | The mixed coefficient does not validate measured prediction. |
| Held-out outcome | `last10_avg` combines source-specific metrics; the measured arithmetic sources use late training reward | It is not a common held-out endpoint. |
| Independent residual evidence | The n17 residual uses the same mixed cohort and residualises only the outcome | It adds no measured-only predictive validation. |
| Projection membership uncertain | All 5,540 retained projection rows exactly reproduce the deterministic recipe; 4,500 unique keys and 1,040 excess duplicates | Projection provenance is resolved and duplicates are disclosed. |

The analysis also retains all 22 raw measured arithmetic runs from two fixed
studies and recomputes their logged metrics. It reports six separate, tiny strata
rather than pooling study or algorithm differences. Recorded post-training
`heldout_acc` remains separate from training reward; the fresh arithmetic draws
are not certified disjoint from the training domain. No predictive or collapse
validation follows from these near-ceiling runs.

## Replay locally

Python, NumPy and SciPy are required. From this directory:

```sh
python3 analyze.py --input-dir inputs --output replay.json
python3 -m unittest discover -s . -p test_analyze.py
```

`PROTOCOL.md` specifies fixed inclusion, exclusions and acceptance before the Colab
run. The analyzer pins the input manifest and protocol hashes. `result.json` is the
actual Colab result, not a hand-edited summary. `execution_receipt.json` records the
Linux/Python/NumPy/SciPy environment, source hashes, local comparison and cleanup.
Original source snapshots are mapped in `inputs/manifest.json`; remote validation
checks the transferred selected data, not original execution authenticity.

## Replay through Colab CLI

With authenticated Colab CLI, first create the local comparison file from the ZIP
or repository root, then run the included launcher:

```sh
python3 reports/project_finish_2026-10-11/colab_validation/analyze.py \
  --input-dir reports/project_finish_2026-10-11/colab_validation/inputs \
  --output reports/project_finish_2026-10-11/validation/colab_followup/local_result.json
python3 reports/project_finish_2026-10-11/colab_validation/run_colab.py
```

The launcher transfers only the explicit numeric inputs, source excerpts and
analyzer. It requests CPU, sets a 180-second execution limit, and uses an isolated
session-state file. `colab run` releases its runtime; the launcher checks cleanup.
A replay writes new results and receipts, so use a copy if preserving delivered
archive hashes. No Drive mount, access token, model checkpoint or private text is
required. CLI provisioning time is separate from the sub-second analysis time.
