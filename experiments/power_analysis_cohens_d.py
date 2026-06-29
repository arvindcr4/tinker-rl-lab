import json
import numpy as np
from pathlib import Path
from statsmodels.stats.power import TTestIndPower

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d for two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return float('nan')
    
    var1 = np.var(group1, ddof=1) if n1 > 1 else 0
    var2 = np.var(group2, ddof=1) if n2 > 1 else 0
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
        
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def run_power_analysis():
    """Run formal power analysis to determine minimum detectable effect size."""
    analysis = TTestIndPower()
    # 5 seeds per condition for Modal H100 runs (matches the released design)
    min_effect = analysis.solve_power(nobs1=5, alpha=0.05, power=0.80)
    print(f"Power Analysis: For N=5, minimum detectable Cohen's d is {min_effect:.3f}")

if __name__ == '__main__':
    print("--- Paper 1: Statistical Rigor & Power Analysis ---")
    run_power_analysis()
    print("Scaffolding Cohen's d extraction for cross-library and scaling runs.")
