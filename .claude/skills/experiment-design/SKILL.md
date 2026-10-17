---
name: experiment-design
description: Design experiment plans with progressive stages — initial implementation, baseline tuning, creative research, and ablation studies. Plan baselines, datasets, hyperparameter sweeps, and evaluation metrics. Use when planning experiments for a research paper.
argument-hint: [idea-or-plan]
---

# Experiment Design

Design structured, progressive experiment plans for research papers.

## Input

- `$0` — Research idea, plan, or method description

## References

- 4-stage progressive experiment prompts: `~/.claude/skills/experiment-design/references/stage-prompts.md`

## Scripts

### Generate experiment design
```bash
python ~/.claude/skills/experiment-design/scripts/design_experiments.py --plan research_plan.json --output experiment_design.json
python ~/.claude/skills/experiment-design/scripts/design_experiments.py --method "contrastive learning" --task classification --format markdown
```

Generates baselines, ablation matrix, hyperparameter grid, metric selection. Stdlib-only.

## 4-Stage Progressive Framework (from AI-Scientist-v2)

### Stage 1: Initial Implementation
- Focus on getting a basic working implementation
- Use a simple dataset
- Aim for basic functional correctness
- Completion: at least one working (non-buggy) implementation

### Stage 2: Baseline Tuning
- Tune hyperparameters (learning rate, epochs, batch size)
- Do NOT change model architecture
- Test on at least TWO datasets
- Completion: planned runs and analyses finished, including negative results

### Stage 3: Creative Research
- Explore novel improvements and insights
- Be creative and think outside the box
- Test on at least THREE datasets
- Completion: planned valid evaluation finished, regardless of improvement

### Stage 4: Ablation Studies
- Systematic component analysis
- Each ablation tests a different aspect
- Use same datasets as Stage 3
- Completion: all planned ablations done

## Output Format

```json
{
  "stages": [
    {
      "name": "initial_implementation",
      "goals": ["Basic working baseline", "Simple dataset"],
      "max_iterations": 5,
      "completion_criteria": "Working implementation with non-zero accuracy"
    }
  ],
  "baselines": ["Method A", "Method B"],
  "datasets": ["Dataset1", "Dataset2", "Dataset3"],
  "metrics": ["accuracy", "F1", "inference_time"],
  "ablation_components": ["component_A", "component_B"],
  "hyperparameter_grid": {
    "lr": [1e-4, 1e-3, 1e-2],
    "batch_size": [32, 64, 128]
  },
  "num_seeds": 3
}
```

## Rules

- Always start simple (Stage 1) before complex experiments
- Each stage builds on the best result from the previous stage
- Multi-seed evaluation under the prespecified power/precision plan
- Document every experiment run in notes.txt
- Generate figures for training curves and comparisons

## Related Skills

## Research integrity contract v2 (mandatory)

Select candidates on development/validation outcomes only. Confirmatory test outcomes must remain unavailable to adaptive selection. Freeze method, configuration, split, seed set, endpoint, and analysis before confirmation. Previously inspected test sets are development evidence and must be recorded as such. Completion means all planned valid runs and analyses are finished, including negative and inconclusive outcomes. Mean and standard deviation are descriptive; inference requires a prespecified effect, interval/test, experimental unit, and multiplicity correction. Seed counts follow power or precision targets.
