import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower

def plot_power():
    analysis = TTestIndPower()
    effect_sizes = np.linspace(0.1, 2.5, 50)
    powers_n10 = [analysis.solve_power(effect_size=es, nobs1=10, alpha=0.05) for es in effect_sizes]
    powers_n5 = [analysis.solve_power(effect_size=es, nobs1=5, alpha=0.05) for es in effect_sizes]
    powers_n20 = [analysis.solve_power(effect_size=es, nobs1=20, alpha=0.05) for es in effect_sizes]
    
    plt.figure(figsize=(8, 6))
    plt.plot(effect_sizes, powers_n5, label='N=5', linestyle='--')
    plt.plot(effect_sizes, powers_n10, label='N=10 (Our Protocol)', linewidth=3, color='blue')
    plt.plot(effect_sizes, powers_n20, label='N=20', linestyle='-.')
    
    plt.axhline(0.8, color='red', linestyle=':', label='80% Power Threshold')
    plt.axvline(1.325, color='gray', linestyle=':', label="Detects d=1.325 at N=10")
    
    plt.title('Statistical Power Curve for RL Framework Benchmarking')
    plt.xlabel("Effect Size (Cohen's d)")
    plt.ylabel('Statistical Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/figures/figure_stat_rigor.pdf')
    print("Saved paper/figures/figure_stat_rigor.pdf")

if __name__ == '__main__':
    plot_power()
