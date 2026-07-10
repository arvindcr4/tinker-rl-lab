# Research Engineering

This context describes the repository’s research-engineering machinery: experiment runners, paper audits, figure generation, and reproducibility checks for TinkerRL.

## Language

**GRPO run module**:
The deepened module that owns the GRPO seed loop, sampling, loss call, optimizer step, and checkpoint cadence for an experiment.
_Avoid_: experiment script, runner copy.

**Dataset adapter**:
A small adapter that supplies GRPO training examples as `(prompt, target_tool, arguments)` pairs or equivalent task records.
_Avoid_: dataset loader copy, raw data helper.

**Reward adapter**:
A small adapter that scores sampled completions against the task target and returns a numeric reward.
_Avoid_: scorer helper, metric function.

**Audit runner**:
The module that imports audit functions, collects audit results, formats the `METRIC` lines, and sets the suite exit code.
_Avoid_: audit shell script, stdout parser.

**Audit result**:
The structured value returned by an audit: a name plus a collection of issue records.
_Avoid_: METRIC string, printed report.

**Grouped audit**:
An audit that returns multiple grouped check results, used for large scientific audits such as `platform_local/scientific_audit.py`.
_Avoid_: monolithic audit, giant checker.

**Figure module**:
The consolidated figure-generation module that owns figure rendering and writes the paper’s figure outputs.
_Avoid_: figure script family, plot script.

**Results adapter**:
An adapter that turns a data source such as `experiments/master_results.json`, measured artifacts, or missing-figure sources into the records the figure module consumes.
_Avoid_: data loader copy, provenance helper.

**Fallback adapter**:
A results adapter that explicitly supplies placeholder or canonical fallback data when measured artifacts are missing.
_Avoid_: hardcoded plot data, placeholder script.
