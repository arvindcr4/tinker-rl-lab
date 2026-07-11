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

**Experiment preset**:
A named GRPO configuration that selects the run parameters, dataset adapter, reward adapter, and evaluation behavior for a reproducible experiment family.
_Avoid_: copied constants, script variant.

**Run checkpoint manifest**:
The local structured record that binds a partial GRPO run to its configuration, optimizer state, completed step, sampler path, and reward trace.
_Avoid_: resume file, progress JSON.

**Audit runner**:
The module that imports audit functions, collects audit results, formats the `METRIC` lines, and sets the suite exit code.
_Avoid_: audit shell script, stdout parser.

**Audit result**:
The structured value returned by an audit: a name plus a collection of issue records.
_Avoid_: METRIC string, printed report.

**Audit issue**:
A stable machine-readable finding code contained by an audit result.
_Avoid_: print line, warning text.

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

**Figure manifest**:
The provenance record emitted by the figure module, naming the renderer, results adapter source, profile, and generated outputs.
_Avoid_: output log, image list.
