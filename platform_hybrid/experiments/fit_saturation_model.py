import json
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

# Ignore optimize warnings for poorly fitting data
warnings.filterwarnings('ignore')

def saturation_model(t, R_max, lam, R_0):
    return R_0 + (R_max - R_0) * (1 - np.exp(-lam * t))

def analyze_saturation(results_file):
    path = Path(results_file)
    if not path.exists():
        print(f"File {results_file} not found.")
        return
    
    with open(path) as f:
        data = json.load(f)
    
    print("--- Paper 3: Scaling Laws & Saturation Dynamics ---")
    for exp in data.get('experiments', []):
        trace = exp.get('reward_trace', [])
        if not trace or len(trace) < 5:
            continue
        t_data = np.arange(len(trace))
        r_data = np.array(trace)
        try:
            popt, _ = curve_fit(saturation_model, t_data, r_data, bounds=([0, 0, 0], [1.0, 1.0, 1.0]))
            R_max, lam, R_0 = popt
            if lam > 0.01: # Filter out flatlines
                print(f"Model: {exp.get('model_short', 'unknown')} | R_max: {R_max:.3f} | lambda: {lam:.3f}")
        except Exception:
            pass

if __name__ == '__main__':
    analyze_saturation('/Users/arvind/research/paper/tinker-rl-lab/experiments/master_results.json')
