# Experiment Design Stage Prompts

Extracted from AI-Scientist-v2 (agent_manager.py) and AI-Researcher (exp_analyser.py).

## 4-Stage Progressive Experiment Framework (AI-Scientist-v2)

### Stage 1: Initial Implementation
**Goal**: Get a working baseline on a simple dataset.

```
Stage 1 - Initial Implementation:
- Implement the core method on the simplest dataset
- Ensure training converges (check training curves)
- Establish baseline metrics
- Verify code runs without errors

Completion criteria:
- Training behavior measured and failures diagnosed
- Validation metrics recorded, including chance-level outcomes
- Code executes end-to-end without errors
```

### Stage 2: Baseline Tuning
**Goal**: Optimize hyperparameters and test on multiple datasets.

```
Stage 2 - Baseline Tuning:
- Tune learning rate, batch size, and key hyperparameters
- Test on at least 2 datasets
- Compare against published baselines
- Run the prespecified seed set; report uncertainty descriptively

Completion criteria:
- Planned comparisons completed, regardless of direction
- Variation across the planned seeds reported
- Training convergence or nonconvergence documented
```

### Stage 3: Creative Research
**Goal**: Novel improvements and comprehensive evaluation.

```
Stage 3 - Creative Research:
- Implement novel improvements to the method
- Test on 3+ datasets
- Compare against 3+ baselines
- Ablation of key design choices
- Generate publication-quality figures

Completion criteria:
- Planned baseline comparisons complete, including null or negative results
- Planned ablations complete; claims reflect supported or unsupported outcomes
- Figures are informative and well-designed
```

### Stage 4: Ablation Studies
**Goal**: Systematic component analysis.

```
Stage 4 - Ablation Studies:
- Remove/modify each key component one at a time
- Measure impact on performance
- Sensitivity analysis for key hyperparameters
- Report mean ± std descriptively; inferential claims require the prespecified analysis

Completion criteria:
- Every planned ablation completed; unsupported claims removed
- Hyperparameter sensitivity reported
- Results table is complete with all comparisons
```

## VLM-Based Stage Completion Check (AI-Scientist-v2)

```
Examine the training curves and results:
1. Is the training loss decreasing?
2. Is validation performance improving?
3. Has the model converged or does it need more epochs?
4. Are there signs of overfitting?
5. Is the performance competitive with baselines?

Based on this analysis, determine if the current stage is complete
or if more experiments are needed.
```

## Best-Node Selection (AI-Scientist-v2)

```
Given the following experiment results and their training curves,
holistically select the best experiment considering:
1. Development/validation performance (primary metric)
2. Training stability (smooth loss curves)
3. Consistency across seeds
4. Generalization (train-validation gap)

Experiment results:
{results_json}

Select the best experiment and justify your choice.
```

## Ablation Study Design (AI-Researcher)

```
Given the experimental results:
{results}

Design an ablation study to verify each component's contribution:
1. List all key components of the method
2. For each component, propose a variant where it is removed/replaced
3. Predict expected impact of each removal
4. Prioritize: test the most impactful ablations first
```

## Sensitivity Analysis (AI-Researcher)

```
Design a sensitivity analysis for these hyperparameters:
{hyperparameters}

For each hyperparameter:
1. Define a reasonable range to test
2. Specify the number of values to try
3. Identify which metrics to track
4. Note any interactions between hyperparameters
```

## Research integrity contract v2 (mandatory)

Select candidates on development/validation outcomes only. Confirmatory test outcomes must remain unavailable to adaptive selection. Freeze method, configuration, split, seed set, endpoint, and analysis before confirmation. Previously inspected test sets are development evidence and must be recorded as such. Completion means all planned valid runs and analyses are finished, including negative and inconclusive outcomes. Mean and standard deviation are descriptive; inference requires a prespecified effect, interval/test, experimental unit, and multiplicity correction. Seed counts follow power or precision targets.
