import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def saturation_model(t, R_max, lam, R_0):
    return R_0 + (R_max - R_0) * (1 - np.exp(-lam * t))

def plot_saturation():
    results_file = Path('experiments/master_results.json')
    if not results_file.exists():
        results_file = Path('master_results.json')
    with open(results_file) as f:
        data = json.load(f)
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    models_plotted = 0
    
    for exp in data.get('experiments', []):
        trace = exp.get('reward_trace', [])
        if not trace or len(trace) < 5:
            continue
        t_data = np.arange(len(trace))
        r_data = np.array(trace)
        model_name = exp.get('model_short', 'unknown')
        
        try:
            popt, _ = curve_fit(saturation_model, t_data, r_data, bounds=([0, 0, 0], [1.0, 1.0, 1.0]))
            R_max, lam, R_0 = popt
            if lam > 0.01 and R_max > 0.05 and models_plotted < 10:
                t_fit = np.linspace(0, len(trace), 100)
                r_fit = saturation_model(t_fit, *popt)
                plt.scatter(t_data, r_data, alpha=0.3, color=colors[models_plotted])
                plt.plot(t_fit, r_fit, label=f"{model_name} (Rmax={R_max:.2f}, $\\lambda$={lam:.2f})", color=colors[models_plotted])
                models_plotted += 1
        except Exception:
            pass
            
    plt.title('Reward Saturation Dynamics Across Frameworks and Models')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/figures/figure_saturation_curves.pdf')
    print("Saved paper/figures/figure_saturation_curves.pdf")

if __name__ == '__main__':
    plot_saturation()
