with open("reports/final/capstone_final_report.tex", "r") as f:
    content = f.read()

fig3 = """\\begin{figure}[h]
  \\centering
  \\includegraphics[width=\\textwidth]{fig3_synthetic_vs_real.png}
  \\caption{Comparison of synthetic versus real data environments, illustrating that GRPO is highly sensitive to the structural properties and ambiguity of the task.}
  \\label{fig:synthetic-vs-real}
\\end{figure}

"""
content = content.replace("\\section{The Zero-Shot Hit Rate Threshold: Why GRPO Fails on Semantics}\n\\label{sec:structure-semantics}\n", "\\section{The Zero-Shot Hit Rate Threshold: Why GRPO Fails on Semantics}\n\\label{sec:structure-semantics}\n\n" + fig3)


fig1 = """\\begin{figure}[h]
  \\centering
  \\includegraphics[width=\\textwidth]{fig1_capacity_threshold.png}
  \\caption{Capacity threshold demonstrating how model scale and instruction tuning interact with GRPO gradient availability. Models below the capability threshold suffer immediate ZVF saturation.}
  \\label{fig:capacity-threshold}
\\end{figure}

"""
content = content.replace("\\section{Instruction Tuning Dominates Small-Model GRPO}\n\\label{sec:instruction-tuning}\n", "\\section{Instruction Tuning Dominates Small-Model GRPO}\n\\label{sec:instruction-tuning}\n\n" + fig1)

fig2 = """\\begin{figure}[h]
  \\centering
  \\includegraphics[width=\\textwidth]{fig2_diagnostics.png}
  \\caption{Reward and diagnostic trajectories across key ablation runs. ZVF and GU provide an early indicator of whether a run has useful learning signal or is merely wasting compute.}
  \\label{fig:diagnostics-curves}
\\end{figure}

"""
content = content.replace("\\section{ZVF Predicts Failure and Wasted Compute}\n\\label{sec:zvf-results}\n", "\\section{ZVF Predicts Failure and Wasted Compute}\n\\label{sec:zvf-results}\n\n" + fig2)

with open("reports/final/capstone_final_report.tex", "w") as f:
    f.write(content)
