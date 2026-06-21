# Automated Compliance Checker
- ID `6a37a843-e158-83e8-9139-05719599a47c` created 2026-06-21 09:00 UTC | model gpt-5-5-pro

---

## QUERY

Read the attached paper. Invent an automated compliance checker that scans a GRPO paper or repo and scores it against the 7-item MIN-REPORT-RL checklist. Provide the scoring rubric, heuristics for each item, and example outputs.

### File: zvf-program/position/min_report_rl.tex
Lines: 1-621
```
  1 | % =============================================================================
  2 | % MIN-REPORT-RL: A Minimum-Reportable-Stack Standard for
  3 | %                Reinforcement-Learning Post-Training of LLMs
  4 | % =============================================================================
  5 | % Pillar 4 of the ZVF Program. POSITION PAPER (NeurIPS/ICML Position Track +
  6 | % MLRC / ICLR Reproducibility Track).
  7 | %
  8 | % This is a standalone draft. It is self-contained and should compile with a
  9 | % plain `article` class (no external .sty or .bib required: citations are
 10 | % rendered as inline TODO keys via a redefined \cite, and the bibliography is
 11 | % a manual thebibliography stub). When promoting to a venue template, swap the
 12 | % preamble for neurips_2026.sty + \usepackage[numbers]{natbib} and replace the
 13 | % TODO \cite keys / thebibliography stub with a real .bib.
 14 | % =============================================================================
 15 | 
 16 | \documentclass[11pt]{article}
 17 | 
 18 | \usepackage[utf8]{inputenc}
 19 | \usepackage[T1]{fontenc}
 20 | \usepackage[margin=1in]{geometry}
 21 | \usepackage{hyperref}
 22 | \usepackage{url}
 23 | \usepackage{booktabs}
 24 | \usepackage{amsmath}
 25 | \usepackage{amssymb}
 26 | \usepackage{xcolor}
 27 | \usepackage{enumitem}
 28 | \usepackage{microtype}
 29 | \usepackage{graphicx} % for \resizebox on wide tables
 30 | 
 31 | % ---------------------------------------------------------------------------
 32 | % TODO-citation shim. We do NOT invent bib entries with fake authors/years.
 33 | % Every \cite below uses a clearly-TODO key and renders in red so a human can
 34 | % find and replace it with a real reference + .bib entry before submission.
 35 | % ---------------------------------------------------------------------------
 36 | \renewcommand{\cite}[1]{\textcolor{red}{[\textsc{cite:}\,#1]}}
 37 | \newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO:\ #1]}}}}
 38 | \newcommand{\zvf}{\textsc{Zvf}}
 39 | \newcommand{\gu}{\textsc{Gu}}
 40 | \newcommand{\minreport}{\textsc{Min-Report-RL}}
 41 | 
 42 | \title{\minreport{}: Reporting the Stack, Not the Label\\
 43 | \large A Community Minimum-Reportable-Stack Standard and
 44 | Reproducibility-Audit Protocol\\for Group-Relative RL Post-Training of LLMs}
 45 | 
 46 | \author{Arvind C R \and the ZVF Program\thanks{Pillar~4 of the ZVF Program.
 47 | Author list and affiliations \todo{finalize author block and affiliations}.}}
 48 | 
 49 | \date{\today}
 50 | 
 51 | \begin{document}
 52 | \maketitle
 53 | 
 54 | % =============================================================================
 55 | \begin{abstract}
 56 | % =============================================================================
 57 | Reinforcement-learning post-training of large language models is reported as
 58 | though an algorithm label---PPO, GRPO, DPO, and the growing GRPO family
 59 | (DAPO, GSPO, Dr.GRPO, MAD-GRPO, \ldots)---fixes the experiment. It does not.
 60 | In a controlled audit of a group-relative RL runner, nominally identical
 61 | ``GRPO'' configurations (same model, group size, learning rate, dataset,
 62 | seed, and step budget) produced last-10 training reward of $84.4\%$~\todo{trace to v1 audit citation} on one
 63 | backend and $5.0\%$~\todo{trace to v1 audit citation} on another---a $\sim 17\times$~\todo{trace to v1 audit citation} gap with \emph{no visible
 64 | hyperparameter difference}. The label was held constant; the \emph{stack} was
 65 | not. We argue that this stack-conditioning is not an exotic edge case but the
 66 | default state of the field, and that the literature's current reporting norms
 67 | make it structurally impossible to tell whether a claimed algorithmic gain
 68 | survives a change of backend, sampler, reference-policy/KL handling, LoRA
 69 | configuration, or reward parser.
 70 | 
 71 | This position paper proposes \minreport{}: a seven-item
 72 | \emph{minimum-reportable-stack} checklist that every GRPO-family paper should
 73 | satisfy, where each item is included precisely because it is a documented
 74 | lever that can flip a head-to-head comparison. We then propose a concrete
 75 | reproducibility-audit protocol---re-implement DAPO, GSPO, Dr.GRPO, and
 76 | MAD-GRPO inside \emph{one} controlled stack and report which claimed gains
 77 | survive---and we specify the audit's experimental design, with results tables
 78 | whose cells are explicit placeholders to be filled from the audit corpus. We
 79 | close with an adoption path for getting TRL, verl, and OpenRLHF to log the
 80 | \minreport{} fields by default, and with responses to the obvious objections.
 81 | We do \emph{not} claim that any specific GRPO variant is overstated; we claim
 82 | that, under current reporting, the field cannot \emph{know}, and that this is
 83 | fixable with a few lines of telemetry.
 84 | \end{abstract}
 85 | 
 86 | % =============================================================================
 87 | \section{Introduction: The Telemetry and Reporting Gap}
 88 | \label{sec:intro}
 89 | % =============================================================================
 90 | 
 91 | A reader of the 2023--2026 RL-for-LLM literature could be forgiven for
 92 | believing that ``we trained with GRPO'' is a complete experimental
 93 | description. It is not. The same three-letter label is attached to runs that
 94 | differ in their loss form (is there a PPO ratio? a clip? a completion-only
 95 | token mask?), their reference-policy and KL handling (frozen reference? KL
 96 | penalty in the loss, or in the reward, or absent?), their sampler and backend
 97 | (vLLM vs.\ a managed inference API; bf16 vs.\ fp32 logits), their group-size
 98 | schedule, and the parser that converts a generation into a scalar reward. Each
 99 | of these is, individually, sufficient to move a result. Together they form a
100 | \emph{treatment} that the algorithm label does not name.
101 | 
102 | This is the LLM-post-training analogue of the deep-RL reproducibility crisis
103 | documented a decade ago \cite{henderson2018deeprl, islam2017reproducibility},
104 | now recurring one abstraction level up. The earlier crisis was about code-level
105 | and hyperparameter variance within a single algorithm; the present one is
106 | worse, because the ``stack'' spans a managed API whose loss we cannot inspect,
107 | a tokenizer and chat template that silently change the prompt, a sampler whose
108 | numerics differ from the trainer's, and a reward parser that can reward a
109 | format artifact instead of a correct answer.
110 | 
111 | \paragraph{The concrete trigger.}
112 | In an audit of a critic-free group-relative runner \cite{zvfaudit2026}, we
113 | attempted a deliberately boring comparison: hold the \emph{visible} GRPO
114 | configuration fixed---Qwen3-8B, group size $G=8$, learning rate $10^{-5}$,
115 | GSM8K, 30 steps, seed~42---and swap only the backend. A managed runner reached
116 | $84.4\%$~\todo{trace to v1 audit citation} last-10 training reward; TRL on an H100 reached $5.0\%$~\todo{trace to v1 audit citation}. No visible
117 | hyperparameter explains the gap. What differs is everything the label omits:
118 | checkpoint and tokenizer identity, prompt construction, sampler behavior, loss
119 | masking, KL/reference handling, optimizer defaults, LoRA target modules,
120 | precision, rollout plumbing, checkpoint selection, and the evaluator. The
121 | comparison is not evidence that one backend is better; it is evidence that
122 | ``the GRPO config'' was \emph{under-specified}. If a backend swap can move a
123 | result by $17\times$~\todo{trace to v1 audit citation} with no visible knob change, then a head-to-head between
124 | two GRPO \emph{variants}---each implemented in its own stack---tells us almost
125 | nothing about the variants.
126 | 
127 | \paragraph{Why this is a reporting problem, not just a science problem.}
128 | The gap above is not a bug to be fixed by a better trainer; it is information
129 | that is routinely \emph{not logged}. Most papers do not report whether the KL
130 | term was in the loss or the reward, whether the token mask was completion-only,
131 | what the sampler precision was, or what the per-step group-size schedule looked
132 | like. They also rarely report the per-step optimization telemetry---in our
133 | framework, the Zero-Variance Fraction (\zvf{}) and Gradient Utilization
134 | (\gu{})---that would reveal whether a run had any usable learning signal at all
135 | \cite{zvfaudit2026}. A field cannot audit what it does not record. The fix is
136 | cheap: a small, fixed set of fields, logged by default.
137 | 
138 | \paragraph{Scope and provenance of the worked numbers.}
139 | The concrete numbers in this paper used as motivating examples
140 | ($84.4\%$/$\,5.0\%$/$\,17\times$ in the audit;
141 | $61.6$--$89.6\%$ prompt-token loss magnitude;
142 | $82.0\%\to83.3\%$ held-out control at $p=0.26$;
143 | $95.0$--$98.1\%$/$\,95.0\%$ Llama-3.3-70B run)
144 | are \emph{inherited from the v1 audit paper that motivated this
145 | position piece} \todo{cite v1 audit paper, once assigned} and are
146 | marked inline with \todo{trace to v1 audit citation} wherever they
147 | appear. A reader who wants to verify the numbers should consult that
148 | paper; this manuscript presents them as illustrative of the
149 | \emph{kind} of stack-driven flip, not as re-derivable from a fresh
150 | run. Once the v1 audit paper is in citation scope, the inline
151 | \todo{trace} markers are replaced with real \texttt{\textbackslash cite\{zvfaudit\}}
152 | keys and the corresponding bibliography entry below is filled in.
153 | We do \emph{not} claim that any specific number here is independently
154 | verifiable against the present manuscript's experimental record.
155 | 
156 | \paragraph{Contributions.}
157 | This paper makes three:
158 | \begin{enumerate}[leftmargin=1.6em, itemsep=2pt]
159 |   \item \textbf{The stack-conditioning thesis, sharpened into a reporting
160 |         standard.} We restate and extend the audit finding that algorithm
161 |         labels are under-specified treatments (\S\ref{sec:stack}), and convert
162 |         it into \minreport{}: a seven-item minimum-reportable-stack checklist
163 |         in which every item is justified by a mechanism through which it can
164 |         flip a comparison (\S\ref{sec:standard}).
165 |   \item \textbf{A controlled reproducibility-audit protocol.} We specify a
166 |         single-stack re-implementation of DAPO, GSPO, Dr.GRPO, and MAD-GRPO,
167 |         with a pre-registered analysis that reports which claimed gains survive
168 |         when the stack is held fixed (\S\ref{sec:audit}). Result cells are
169 |         explicit placeholders to be filled from the audit corpus.
170 |   \item \textbf{An adoption path.} We describe concrete, low-friction changes
171 |         to TRL, verl, and OpenRLHF so that the \minreport{} fields are emitted
172 |         by default, plus a venue checklist (\S\ref{sec:adoption}).
173 | \end{enumerate}
174 | 
175 | % =============================================================================
176 | \section{The Stack-Conditioning Problem}
177 | \label{sec:stack}
178 | % =============================================================================
179 | 
180 | \paragraph{Thesis.}
181 | \emph{Algorithm rankings in RL post-training are stack-conditioned.} A claim of
182 | the form ``method $A$ beats method $B$'' is, in current practice, a claim about
183 | two \emph{full stacks} $S_A$ and $S_B$ that happen to be tagged $A$ and $B$. The
184 | quantities that determine the outcome---backend and sampler numerics, the exact
185 | loss form (ratio, clip, token mask), reference-policy and KL handling, LoRA
186 | target modules and rank, precision, the reward parser, the group-size schedule,
187 | checkpoint selection, and the evaluation harness---are co-determinants of the
188 | result, and they are typically \emph{confounded with the label}. In a one-way
189 | variance decomposition over a heterogeneous run corpus, the framework/library
190 | factor and the training-algorithm factor each explain roughly half of the
191 | variance in the outcome ($\eta^2 \approx 0.55$ for both), but these factors are
192 | themselves confounded with model, backend, reward, hardware, and task
193 | \cite{zvfaudit2026}. ``Algorithm'' and ``stack'' are not separable in the
194 | existing literature because nobody reports the stack.
195 | 
196 | \paragraph{A comparison that flips.}
197 | Consider the minimal worked example from the audit. The \emph{visible} GRPO
198 | configuration is matched exactly:
199 | \begin{center}
200 | \small
201 | \begin{tabular}{@{}lll@{}}
202 | \toprule
203 | \textbf{Visible config (held constant)} & \textbf{Backend $S_1$} & \textbf{Backend $S_2$}\\
204 | \midrule
205 | Model & Qwen3-8B & Qwen3-8B \\
206 | Group size $G$ & 8 & 8 \\
207 | Learning rate & $10^{-5}$ & $10^{-5}$ \\
208 | Dataset / split & GSM8K-500 & GSM8K-500 \\
209 | Steps & 30 & 30 \\
210 | Seed & 42 & 42 \\
211 | \midrule
212 | \textbf{Last-10 training reward} & $\mathbf{84.4\%}$~\todo{trace} & $\mathbf{5.0\%}$~\todo{trace} \\
213 | \bottomrule
214 | \end{tabular}
215 | \end{center}
216 | A na\"ive reader concludes ``$S_1$'s GRPO is $17\times$~\todo{trace to v1 audit citation} better.'' The correct
217 | reading is that the two rows are \emph{different treatments wearing the same
218 | label}: the loss form, token mask, KL/reference handling, sampler precision,
219 | LoRA targets, optimizer defaults, and reward parser were never held fixed
220 | because they were never reported, hence never matched. Flip any one of them and
221 | the ranking can invert. This is the entire problem in one table: the label is
222 | constant, the conclusion is determined by the unreported stack.
223 | 
224 | \paragraph{Why the GRPO family makes this acute.}
225 | The methods this paper targets---DAPO, GSPO, Dr.GRPO, MAD-GRPO, and the broader
226 | variance-mitigation line (AERO, CPPO, NGRPO, Scaf-GRPO)
227 | \cite{dapo2025, gspo2025, drgrpo2025, madgrpo2025, aero2024, cppo2024,
228 | ngrpo2025, scafgrpo2025}---are typically defined as \emph{small deltas} on a
229 | base GRPO loop: a changed advantage normalization, a clip-bound tweak, a token
230 | mask, a length penalty, an exploration bonus, an adaptive group size. The size
231 | of the claimed improvement is frequently \emph{smaller} than the stack effect
232 | demonstrated above. When the treatment effect you are trying to measure is
233 | $2$--$5$ points and the nuisance effect of an unreported stack difference is
234 | tens of points, an uncontrolled head-to-head is not measuring the method. The
235 | GRPO family is exactly the regime in which stack-conditioning is most likely to
236 | masquerade as algorithmic progress.
237 | 
238 | % =============================================================================
239 | \section{The \minreport{} Standard}
240 | \label{sec:standard}
241 | % =============================================================================
242 | 
243 | \minreport{} is a minimum-reportable-stack: the smallest set of fields such
244 | that, if two GRPO-family papers both report them, a reader can tell whether
245 | their comparison is confounded. Each item below is included \emph{because there
246 | is a known mechanism by which it can flip a comparison}; an item that could not
247 | change a ranking would not earn its place on a minimum list.
248 | 
249 | \paragraph{1. Loss form.}
250 | \emph{Report:} whether the update uses a PPO-style importance ratio
251 | $w_{i,t}=\pi_\theta/\pi_{\theta_{\text{old}}}$; whether and how it is clipped
252 | (and the clip bounds, including asymmetric DAPO-style ``clip-higher'');
253 | whether the token mask is completion-only or whole-sequence; and whether
254 | advantages are normalized per-group, per-batch, or with a running estimate.
255 | \emph{Why it can flip:} the choice of token mask alone changes which tokens
256 | carry gradient. In one diagnostic, $61.6$--$89.6\%$~\todo{trace to v1 audit citation} of the full-sequence loss
257 | magnitude came from \emph{prompt} tokens rather than completion tokens
258 | \cite{zvfaudit2026}; a whole-sequence mask and a completion-only mask are
259 | therefore different objectives sharing a name. Dr.GRPO's contribution is
260 | precisely a change to length/normalization in the loss \cite{drgrpo2025}; GSPO
261 | changes the importance-sampling granularity \cite{gspo2025}. If the baseline's
262 | loss form is unreported, the variant's gain is unattributable.
263 | 
264 | \paragraph{2. Reference policy and KL handling.}
265 | \emph{Report:} whether a frozen reference policy is retained; whether the KL
266 | term sits in the \emph{loss} or is folded into the \emph{reward}; the KL
267 | coefficient and its schedule; and whether KL is estimated forward or reverse,
268 | per-token or per-sequence. \emph{Why it can flip:} KL placement changes the
269 | effective objective and the steady-state exploration. A runner that drops the
270 | reference policy entirely (one optimizer step per rollout, no frozen reference)
271 | is not doing the same thing as a KL-regularized GRPO, even at matched learning
272 | rate \cite{zvfaudit2026}. Methods that add a second anchor (e.g.\ a dual-KL/SFT
273 | anchor) change steady-state \zvf{} and must be recalibrated, not compared
274 | blind \cite{dar2024}.
275 | 
276 | \paragraph{3. Sampler, backend, and precision.}
277 | \emph{Report:} the rollout engine (vLLM, SGLang, a managed API, the trainer's
278 | own \texttt{generate}}; decoding parameters (temperature, top-$p$,
279 | \texttt{max\_tokens}}; logit precision in the sampler vs.\ the trainer (bf16 /
280 | fp16 / fp32); and whether sampler and trainer share the same tokenizer and chat
281 | template. \emph{Why it can flip:} the sampler defines the rollout distribution
282 | that the group-relative update consumes. A managed, closed-source runner and an
283 | open vLLM path can yield order-of-magnitude different training-reward
284 | trajectories under identical visible configs---this is the mechanism behind the
285 | $17\times$ gap of \S\ref{sec:stack}. Sampler precision shifts the probability
286 | of mixed-reward groups, and hence the available gradient.
287 | 
288 | \paragraph{4. Per-step \zvf{} and \gu{} trajectory.}
289 | \emph{Report:} the per-step Zero-Variance Fraction \zvf{} (fraction of prompts
290 | whose $G$ completions all receive identical reward, contributing zero gradient)
291 | and Gradient Utilization $\gu{}=1-\zvf{}$, logged for every training step.
292 | \emph{Why it can flip:} \zvf{}/\gu{} is the telemetry that reveals \emph{whether
293 | a run had any learning signal at all}. The mixed-group probability
294 | $P(\text{usable}) \approx \frac{1}{N}\sum_x\!\left[1-(1-p_x)^G-p_x^G\right]$
295 | shows that a group-relative update only has signal when sampled groups contain
296 | reward diversity \cite{zvfaudit2026}. A method can ``win'' a comparison purely
297 | because its stack happened to produce lower \zvf{} (more usable groups), not
298 | because its algorithmic idea is better. Without the \zvf{}/\gu{} trajectory, a
299 | collapsed run and a saturated run are indistinguishable from their final reward.
300 | A fixed first-five-step rule (\zvf{}$\,\ge 80\%$ with reward $\le 5\%$) is a
301 | cheap collapse triage; reporting the trajectory lets a reader apply it.
302 | 
303 | \paragraph{5. Group-size schedule (fixed or adaptive).}
304 | \emph{Report:} the group size $G$ at every step, and the rule that changes it
305 | if it is adaptive (e.g.\ AERO-style doubling/halving on a rolling \zvf{}
306 | estimate). \emph{Why it can flip:} $G$ directly controls the mixed-group
307 | probability above and is one of the strongest single knobs we measured---an
308 | ablation on Qwen3-8B/GSM8K moved last-10 reward across a wide band as $G$ went
309 | $2\!\to\!4\!\to\!8\!\to\!16$ \cite{zvfaudit2026}. An adaptive-$G$ method (AERO)
310 | compared against a fixed-$G$ baseline is partly being credited for spending
311 | more compute on hard prompts, an effect that must be separated from the
312 | algorithmic claim \cite{aero2024}.
313 | 
314 | \paragraph{6. Held-out split distinct from the reward environment.}
315 | \emph{Report:} a held-out evaluation slice that is \emph{disjoint} from the
316 | training prompts and scored by a harness, with sample size and a confidence
317 | interval, reported \emph{separately} from online training reward. \emph{Why it
318 | can flip:} online training reward is dynamics evidence, not capability
319 | evidence. In the audit, a clean paired control improved only $82.0\%\to83.3\%$~\todo{trace to v1 audit citation}
320 | ($p=0.26$)~\todo{trace to v1 audit citation} on held-out GSM8K despite training reward near saturation; and
321 | selecting checkpoints by training reward produced an $87$--$95\%$ ``capability''
322 | band that is a selection artifact, not a causal lift \cite{zvfaudit2026}. A
323 | variant that reports only training reward can show a large apparent gain that
324 | vanishes held-out. Four Llama-3.3-70B seeds ranged $95.0$--$98.1\%$~\todo{trace to v1 audit citation} on training
325 | last-10 yet all landed on $95.0\%$~\todo{trace to v1 audit citation} held-out---last-10 variance was
326 | sampling noise, not generalization.
327 | 
328 | \paragraph{7. Decontamination probe results.}
329 | \emph{Report:} the outcome of an explicit train/test contamination check
330 | (n-gram or embedding overlap between training prompts and the held-out /
331 | benchmark slice), and the parser's behavior on adversarial format-only inputs.
332 | \emph{Why it can flip:} verifiable rewards do not eliminate reward hacking;
333 | parser artifacts, format shortcuts, length effects, and train-prompt overfitting
334 | can all inflate a reward without capability gain \cite{zvfaudit2026}. If the
335 | benchmark slice overlaps the reward environment, a ``gain'' may be memorization;
336 | if the parser rewards a format token, a ``gain'' may be a shortcut. A
337 | contamination/parser probe is the minimum evidence that the reported number
338 | measures the thing it claims to measure.
339 | 
340 | \paragraph{Summary table.}
341 | Table~\ref{tab:checklist} collects the standard. The companion
342 | \texttt{CHECKLIST.md} provides a fillable appendix template.
343 | 
344 | \begin{table}[t]
345 | \centering
346 | \caption{The \minreport{} minimum-reportable-stack. Each item is on the list
347 | because it is a documented lever that can flip a head-to-head comparison.}
348 | \label{tab:checklist}
349 | \small
350 | \begin{tabular}{@{}rp{0.30\linewidth}p{0.50\linewidth}@{}}
351 | \toprule
352 | \# & \textbf{Field} & \textbf{Flip mechanism (why it is mandatory)}\\
353 | \midrule
354 | 1 & Loss form (ratio / clip / token mask / advantage norm.) & Token mask reassigns gradient across prompt vs.\ completion tokens; clip and norm.\ are the actual variant deltas.\\
355 | 2 & Reference policy + KL handling & KL placement (loss vs.\ reward) and a frozen-vs-absent reference change the objective and exploration.\\
356 | 3 & Sampler / backend / precision & The sampler defines the rollout distribution; backend/precision drove the $17\times$~\todo{trace to v1 audit citation} matched-config gap.\\
357 | 4 & Per-step \zvf{} / \gu{} trajectory & Reveals whether the run had usable learning signal; separates collapse from saturation.\\
358 | 5 & Group-size schedule (fixed / adaptive) & $G$ sets the mixed-group probability; adaptive-$G$ confounds compute with algorithm.\\
359 | 6 & Held-out split $\neq$ reward environment & Training reward is not capability; selection-by-reward inflates apparent gains.\\
360 | 7 & Decontamination probe results & Verifiable rewards still admit parser/format/length/overlap hacking.\\
361 | \bottomrule
362 | \end{tabular}
363 | \end{table}
364 | 
365 | % =============================================================================
366 | \section{Proposed Reproducibility Audit}
367 | \label{sec:audit}
368 | % =============================================================================
369 | 
370 | The \minreport{} standard tells authors what to report. The audit tells the
371 | community what to \emph{check}: \emph{which claimed GRPO-family gains survive
372 | when the stack is held fixed?}
373 | 
374 | \paragraph{Design.}
375 | Re-implement four prominent GRPO variants---DAPO, GSPO, Dr.GRPO, and
376 | MAD-GRPO \cite{dapo2025, gspo2025, drgrpo2025, madgrpo2025}---as
377 | \emph{minimal configuration overrides on a single shared trainer}. Everything
378 | in the \minreport{} list except the variant's defining delta is held identical
379 | across arms: the same base checkpoint and tokenizer, the same vLLM sampler and
380 | precision, the same reference-policy/KL handling, the same LoRA targets and
381 | rank, the same optimizer, the same reward parser, the same group-size schedule
382 | (where the variant does not itself change it), the same held-out slice, and the
383 | same evaluation harness and seed sweep. Only the named hook differs (loss form,
384 | advantage normalization, clip rule, masking, or group-sizing). This is the same
385 | methodology already validated for the variance-mitigation head-to-head, where
386 | AERO/CPPO/NGRPO/Scaf-GRPO were each implemented as one hook on a shared GRPO
387 | trainer with tokenizer, sampler, optimizer, LoRA adapters, evaluation harness,
388 | and seed sweep held identical \cite{zvfaudit2026}.
389 | 
390 | \paragraph{Two readings per variant.}
391 | For each variant we report (i) its result \emph{as published} (its own stack,
392 | its own numbers) and (ii) its result \emph{in the controlled stack}. The
393 | quantity of interest is the \emph{survival} of the claimed gain: the
394 | controlled-stack delta relative to the shared GRPO baseline, with a confidence
395 | interval and a survival verdict (survives / shrinks / disappears / reverses).
396 | 
397 | \paragraph{Pre-registration.}
398 | The arm list, the shared stack specification, the held-out slice, the seed set,
399 | and the survival thresholds are fixed before any controlled run is scored, to
400 | avoid the selection-by-reward over-optimism the audit itself warns against
401 | \cite{zvfaudit2026}. Every arm reports the full \minreport{} block.
402 | 
403 | \paragraph{Primary results table.}
404 | Table~\ref{tab:audit} is the headline deliverable; all numeric cells are
405 | placeholders to be filled from the audit corpus.
406 | 
407 | \begin{table}[t]
408 | \centering
409 | \caption{\textbf{Controlled single-stack audit of GRPO-family variants.}
410 | Each variant is a minimal hook on one shared trainer (\S\ref{sec:audit}); the
411 | full \minreport{} block is identical across arms except the variant's defining
412 | delta. ``Published $\Delta$'' is the gain the original paper reports on its own
413 | stack; ``Controlled $\Delta$'' is the gain in the shared stack vs.\ the shared
414 | GRPO baseline; ``Survives?'' is the pre-registered verdict. All cells are
415 | \todo{fill from audit corpus}.}
416 | \label{tab:audit}
417 | \small
418 | \resizebox{\textwidth}{!}{%
419 | \begin{tabular}{@{}lcccccc@{}}
420 | \toprule
421 | \textbf{Variant} & \textbf{Defining delta} & \textbf{Published $\Delta$} &
422 | \textbf{Controlled last-10} & \textbf{Controlled held-out} &
423 | \textbf{Controlled $\Delta$ (95\% CI)} & \textbf{Survives?}\\
424 | \midrule
425 | GRPO (baseline) & --- & --- & \todo{} & \todo{} & $0$ (ref.) & --- \\
426 | DAPO     & clip-higher + dyn.\ sampling & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
427 | GSPO     & sequence-level IS ratio      & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
428 | Dr.GRPO  & length/normalization fix     & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
429 | MAD-GRPO & multi-agent / diversity term & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
430 | \bottomrule
431 | \end{tabular}}
432 | \end{table}
433 | 
434 | \paragraph{Telemetry table.}
435 | Table~\ref{tab:audit_telemetry} reports the \minreport{} item-4 telemetry for
436 | each arm, which is the mechanism by which a ``gain'' might be a stack effect
437 | rather than an algorithmic one (e.g.\ a variant that simply lowers \zvf{}).
438 | 
439 | \begin{table}[t]
440 | \centering
441 | \caption{\textbf{Per-arm \minreport{} telemetry.} \zvf{}/\gu{} and collapse
442 | behavior under the shared stack, to attribute any controlled gain to algorithm
443 | vs.\ usable-signal availability. All cells are \todo{fill from audit corpus}.}
444 | \label{tab:audit_telemetry}
445 | \small
446 | \resizebox{\textwidth}{!}{%
447 | \begin{tabular}{@{}lccccc@{}}
448 | \toprule
449 | \textbf{Variant} & \textbf{Mean \zvf{} @ step 25} & \textbf{Mean \gu{} @ step 25} &
450 | \textbf{Collapse rate (seeds)} & \textbf{Time-to-collapse (median steps)} &
451 | \textbf{Group-size schedule}\\
452 | \midrule
453 | GRPO (baseline) & \todo{} & \todo{} & \todo{} & \todo{} & fixed $G$ \\
454 | DAPO     & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
455 | GSPO     & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
456 | Dr.GRPO  & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
457 | MAD-GRPO & \todo{} & \todo{} & \todo{} & \todo{} & \todo{} \\
458 | \bottomrule
459 | \end{tabular}}
460 | \end{table}
461 | 
462 | \paragraph{Statistical protocol.}
463 | Because the audit's independent unit is a seed (not a per-step reward, which is
464 | autocorrelated within a trace), inferential claims are made only across seeds.
465 | We report mean $\pm$ standard error and bootstrap 95\% CIs across the seed
466 | sweep, control the false-discovery rate with Benjamini--Hochberg over the
467 | pre-registered set of survival tests, and treat single-seed arms as descriptive
468 | only. This mirrors the statistical discipline of the source audit, which
469 | explicitly refuses to upgrade trace-level descriptive effect sizes into
470 | inferential evidence about distinct seeds, runs, or stacks \cite{zvfaudit2026}.
471 | \todo{Fix the seed count $S$ and the survival threshold once compute is
472 | budgeted; record minimum detectable effect at the chosen $S$.}
473 | 
474 | \paragraph{Honest scoping.}
475 | The audit's deliverable is \emph{not} ``method $X$ is fake.'' It is a survival
476 | map: for each variant, the fraction of its claimed gain that persists when the
477 | stack is the same. Some gains will survive (they are real algorithmic
478 | contributions); some will shrink (they were partly stack); some may disappear
479 | (they were stack). All three outcomes are informative, and reporting them is
480 | the contribution.
481 | 
482 | % =============================================================================
483 | \section{Adoption Path}
484 | \label{sec:adoption}
485 | % =============================================================================
486 | 
487 | A standard nobody can comply with cheaply will not be adopted. \minreport{} is
488 | designed so that the marginal cost to an author is close to zero \emph{if the
489 | trainers log it by default}.
490 | 
491 | \paragraph{Make the trainers emit it.}
492 | The seven fields are already computable inside every major open trainer:
493 | \begin{itemize}[leftmargin=1.4em, itemsep=2pt]
494 |   \item \textbf{TRL.} The \texttt{GRPOTrainer} already exposes the loss form,
495 |         KL handling, and sampler config; \zvf{}/\gu{} is a few lines over the
496 |         per-prompt group rewards it already computes, logged each step. Propose
497 |         a \texttt{report\_min\_report\_rl=True} flag that writes the seven-field
498 |         block to the run config and the \zvf{}/\gu{} trajectory to the logger.
499 |         \todo{file TRL issue / PR; link.}
500 |   \item \textbf{verl.} Its rollout/actor separation makes sampler, precision,
501 |         and group-size schedule first-class; expose them in the run manifest
502 |         and add a per-step \zvf{}/\gu{} metric. \todo{file verl issue / PR;
503 |         link.}
504 |   \item \textbf{OpenRLHF.} Add the same manifest block and per-step telemetry;
505 |         its reference-model and KL options map directly onto item~2.
506 |         \todo{file OpenRLHF issue / PR; link.}
507 | \end{itemize}
508 | A shared, copy-pasteable emitter (a small library that takes a trainer's config
509 | and per-step group rewards and writes the \minreport{} block as JSON) lets all
510 | three converge on the same schema. \todo{publish the emitter package + schema.}
511 | 
512 | \paragraph{Make the venues ask for it.}
513 | Add a \minreport{} block to the reproducibility checklist for the RL-for-LLM
514 | track, analogous to the existing NeurIPS/ICML reproducibility checklists. The
515 | companion \texttt{CHECKLIST.md} is written to be dropped verbatim into a paper's
516 | appendix.
517 | 
518 | \paragraph{Make it auditable post hoc.}
519 | For already-published work, the same JSON block can be reconstructed from
520 | release artifacts where they exist. The audit of \S\ref{sec:audit} doubles as a
521 | demonstration: it re-derives the \minreport{} block for four variants and shows
522 | the format is sufficient to detect confounded comparisons.
523 | 
524 | % =============================================================================
525 | \section{Objections and Responses}
526 | \label{sec:objections}
527 | % =============================================================================
528 | 
529 | \paragraph{``This is just `report your hyperparameters.'\,''}
530 | No. Hyperparameter tables already exist and the $17\times$~\todo{trace to v1 audit citation} gap occurred
531 | \emph{despite} matched visible hyperparameters. \minreport{} names the fields
532 | that hyperparameter tables systematically omit---loss form, KL placement,
533 | sampler precision, per-step \zvf{}/\gu{}, group-size schedule, held-out
534 | disjointness, contamination probe---precisely because each is a demonstrated
535 | flip lever, not a tuning knob.
536 | 
537 | \paragraph{``Seven items is too many / too few.''}
538 | Each item earns its place by a flip mechanism (\S\ref{sec:standard}); removing
539 | any one reopens a known confound. The list is a \emph{minimum}, not a maximum:
540 | authors may report more. We deliberately stop at seven because that is the set
541 | that, in our audit, was sufficient to explain the comparisons that flipped.
542 | \todo{If the audit surfaces an eighth recurring confound, extend the list and
543 | justify it the same way.}
544 | 
545 | \paragraph{``Re-implementations are unfaithful, so the audit is unfair.''}
546 | The audit reports \emph{both} readings (published and controlled) and frames the
547 | result as survival, not refutation. An unfaithful re-implementation is itself a
548 | finding: if a variant's gain depends on an unstated stack detail that a careful
549 | re-implementer cannot reproduce from the paper, that is a reporting failure the
550 | standard is designed to surface. We will publish every arm's full \minreport{}
551 | block so disagreements are about specific fields, not vibes.
552 | 
553 | \paragraph{``Closed/managed backends can't comply.''}
554 | Partly true, and that is the point: a closed runner cannot expose its loss form
555 | or KL handling, so results from it should be scoped as ``this platform's
556 | implementation,'' not ``GRPO'' \cite{zvfaudit2026}. The standard makes that
557 | scoping explicit rather than letting a closed-stack number stand in for an
558 | algorithm.
559 | 
560 | \paragraph{``\zvf{}/\gu{} is your own metric; why mandate it?''}
561 | \zvf{}/\gu{} is one cheap instantiation of a necessary quantity---\emph{did the
562 | run have usable group-relative signal?}---and we accept substitutes. Under
563 | dense process rewards or scaffolded exploration, \zvf{} degenerates and should
564 | be replaced by per-step reward variance or a gradient-norm-variance surrogate
565 | \cite{zvfaudit2026}. Item~4 mandates \emph{a usable-signal trajectory}, of which
566 | \zvf{}/\gu{} is the default; the point is that \emph{some} such telemetry must be
567 | reported.
568 | 
569 | % =============================================================================
570 | \section{Conclusion}
571 | \label{sec:conclusion}
572 | % =============================================================================
573 | 
574 | The RL-for-LLM literature is in a position the deep-RL community has seen
575 | before: a three-letter label is doing the work of a full experimental
576 | specification, and the unreported remainder of the stack can move a result by
577 | more than the algorithmic effect anyone is trying to claim. We have argued that
578 | this stack-conditioning is the default, demonstrated it with a matched-config
579 | comparison that flips by $17\times$, and proposed two coupled remedies: a
580 | seven-item minimum-reportable-stack (\minreport{}), where every item is a
581 | documented flip lever, and a controlled single-stack audit of DAPO, GSPO,
582 | Dr.GRPO, and MAD-GRPO that reports which claimed gains survive. Neither remedy
583 | requires new science---only a few fields of telemetry and the discipline to log
584 | them. The cost is a JSON block; the payoff is a literature whose comparisons one
585 | can actually trust.
586 | 
587 | % =============================================================================
588 | % Bibliography. MANUAL STUB ONLY. We do NOT invent authors/years. Every entry
589 | % is a TODO key to be replaced by a real citation before submission. When
590 | % promoting to a venue template, delete this block, use \usepackage{natbib},
591 | % and point \bibliography at a real .bib.
592 | %
593 | % SUBMISSION GATE: a reviewer of this manuscript will see a red
594 | % [cite:KEY] marker for every in-text citation and a red [TODO] marker
595 | % for every entry below. The paper will be treated as a draft, not as
596 | % a submission, until the bibliography is real. The 12 cited works are
597 | % all findable; if the position paper cannot stand on the citations it
598 | % has, the claims need to be cut, not the citations stubbed.
599 | % =============================================================================
600 | \section*{References}
601 | \small
602 | \noindent\textcolor{red}{\textbf{[TODO: replace this stub with a real .bib.
603 | Every \texttt{[cite:KEY]} marker above corresponds to one entry below. Do not
604 | ship with placeholder keys. Each entry below carries a [\textsc{required}]
605 | tag indicating its status; all are required for submission.]}}
606 | \begin{itemize}[leftmargin=1.4em, itemsep=1pt, label={}]
607 |   \item \texttt{zvfaudit2026} --- \todo{the ZVF Program v1 audit paper (this group's own work: ``Reward Contrast, Not Algorithm Labels''); cite the arXiv/venue version once assigned.}
608 |   \item \texttt{henderson2018deeprl} --- \todo{Henderson et al., ``Deep Reinforcement Learning that Matters'' (reproducibility crisis in deep RL).}
609 |   \item \texttt{islam2017reproducibility} --- \todo{Islam et al., reproducibility of benchmarked deep RL (verify exact reference).}
610 |   \item \texttt{dapo2025} --- \todo{DAPO paper (clip-higher + dynamic sampling GRPO variant); fill authors/venue/year.}
611 |   \item \texttt{gspo2025} --- \todo{GSPO paper (sequence-level importance sampling); fill authors/venue/year.}
612 |   \item \texttt{drgrpo2025} --- \todo{Dr.GRPO paper (length/normalization fix); fill authors/venue/year.}
613 |   \item \texttt{madgrpo2025} --- \todo{MAD-GRPO paper (multi-agent/diversity GRPO); fill authors/venue/year.}
614 |   \item \texttt{aero2024} --- \todo{AERO (adaptive rollout sizing); from variance-mitigation bib fragment.}
615 |   \item \texttt{cppo2024} --- \todo{CPPO (clip-pruned PPO); from variance-mitigation bib fragment.}
616 |   \item \texttt{ngrpo2025} --- \todo{NGRPO (normalized GRPO); from variance-mitigation bib fragment.}
617 |   \item \texttt{scafgrpo2025} --- \todo{Scaf-GRPO (scaffolded exploration); from variance-mitigation bib fragment.}
618 |   \item \texttt{dar2024} --- \todo{DAR (dual-alignment / dual-KL regularization); from extended-RW bib fragment.}
619 | \end{itemize}
620 | 
621 | \end{document}
```

### File: zvf-program/position/CHECKLIST.md
Lines: 1-168
````md
  1 | # MIN-REPORT-RL — Author Checklist
  2 | 
  3 | A copy-pasteable minimum-reportable-stack for **any GRPO-family RL post-training paper**
  4 | (GRPO, DAPO, GSPO, Dr.GRPO, MAD-GRPO, AERO, CPPO, NGRPO, Scaf-GRPO, …).
  5 | 
  6 | **Why this exists.** Algorithm labels are under-specified treatments. In a controlled audit,
  7 | a *nominally identical* GRPO config (same model, group size, learning rate, dataset, seed,
  8 | step budget) produced **84.4%** [TODO:trace to v1 audit citation] last-10 training reward on one backend and **5.0%** [TODO:trace to v1 audit citation] on
  9 | another — a ~17× gap [TODO:trace to v1 audit citation] with **no visible hyperparameter difference**. The label was constant;
 10 | the stack was not. Each item below is on the list because it is a **documented lever that can
 11 | flip a head-to-head comparison**. If two papers both report these seven fields, a reader can
 12 | tell whether their comparison is confounded.
 13 | 
 14 | Report all seven. They are a *minimum*, not a maximum.
 15 | 
 16 | **Provenance / scope of the worked numbers in this checklist.** The concrete
 17 | numbers used as motivating examples below (61.6–89.6% prompt-token
 18 | loss magnitude; the 17× matched-config gap; the 82.0→83.3% held-out
 19 | control at p=0.26; the 95.0–98.1% / 95.0% Llama-3.3-70B run)
 20 | are inherited from the v1 audit paper that motivated this position
 21 | piece. **[TODO:trace to v1 audit citation]** A reader who wants to
 22 | verify the numbers should consult that paper; this checklist presents
 23 | them as illustrative of the *kind* of stack-driven flip, not as
 24 | re-derivable from a fresh run. Once the v1 audit paper is in
 25 | citation scope, the references go in the `\bibliography{...}` and
 26 | the [TODO:trace] markers are replaced with real `\cite{}` keys.
 27 | 
 28 | ---
 29 | 
 30 | ## The 7 items
 31 | 
 32 | ### 1. Loss form
 33 | - **Report:** PPO importance ratio used? (yes/no). Clipped? bounds (incl. asymmetric
 34 |   "clip-higher")? Token mask: completion-only or whole-sequence? Advantage normalization:
 35 |   per-group / per-batch / running estimate?
 36 | - **Why it can flip:** the token mask reassigns gradient. In one diagnostic [TODO:trace to v1 audit citation], **61.6–89.6%**
 37 |   of full-sequence loss magnitude came from *prompt* tokens, not completion tokens — a
 38 |   whole-sequence and a completion-only mask are different objectives sharing a name. Dr.GRPO
 39 |   *is* a normalization change; GSPO *is* an IS-granularity change. If the baseline loss form
 40 |   is unreported, the variant's gain is unattributable.
 41 | - **Good:** "Token-masked completion-only; PPO ratio with symmetric clip ε=0.2; per-group
 42 |   advantage normalization (subtract group mean, divide by group std)."
 43 | - **Bad:** "We use the standard GRPO loss."
 44 | 
 45 | ### 2. Reference policy + KL handling
 46 | - **Report:** frozen reference policy retained? (yes/no). KL term in the **loss** or folded
 47 |   into the **reward**? KL coefficient + schedule. Forward/reverse, per-token/per-sequence.
 48 | - **Why it can flip:** KL placement changes the objective and steady-state exploration. A
 49 |   runner with *no* frozen reference (one optimizer step per rollout) is not doing KL-regularized
 50 |   GRPO even at matched LR. Dual-anchor methods shift steady-state ZVF and must be recalibrated.
 51 | - **Good:** "Frozen reference = SFT checkpoint; KL in the loss, β=0.04 constant; reverse KL,
 52 |   per-token."
 53 | - **Bad:** "KL regularization as usual." / (silent on whether a reference exists)
 54 | 
 55 | ### 3. Sampler / backend / precision
 56 | - **Report:** rollout engine (vLLM / SGLang / managed API / trainer `.generate`). Decoding
 57 |   params (temperature, top-p, max_tokens). Logit precision in sampler vs. trainer
 58 |   (bf16/fp16/fp32). Same tokenizer + chat template in sampler and trainer? (yes/no).
 59 | - **Why it can flip:** the sampler *defines* the rollout distribution the group-relative update
 60 |   consumes. A managed runner vs. an open vLLM path gave the **17×** [TODO:trace to v1 audit citation] matched-config gap. Sampler
 61 |   precision shifts the probability of mixed-reward groups, hence available gradient.
 62 | - **Good:** "vLLM 0.x rollouts, bf16; trainer logits bf16; temp 0.8, top-p 1.0, max_tokens 512;
 63 |   identical tokenizer/chat template across both."
 64 | - **Bad:** "Generated with default settings."
 65 | 
 66 | ### 4. Per-step ZVF and GU trajectory
 67 | - **Report:** per-step **ZVF** (fraction of prompts whose G completions all get identical
 68 |   reward → zero gradient) and **GU = 1 − ZVF**, logged every step (release the trajectory).
 69 | - **Why it can flip:** ZVF/GU reveals whether the run had *any* usable learning signal. A
 70 |   method can "win" purely because its stack produced lower ZVF (more usable groups), not because
 71 |   its algorithm is better. Without the trajectory, a collapsed run and a saturated run look
 72 |   identical from final reward. (Mixed-group probability:
 73 |   `P(usable) ≈ (1/N) Σ_x [1 − (1−p_x)^G − p_x^G]`.) Cheap collapse triage: first-5-step rule
 74 |   ZVF ≥ 80% with reward ≤ 5%.
 75 | - **Good:** "ZVF/GU logged per step (released CSV); mean ZVF@25 = 0.43; no run trips the
 76 |   collapse rule."
 77 | - **Bad:** Only final reward reported; no per-step signal telemetry.
 78 | - **Substitutes allowed:** under dense process rewards or scaffolded exploration ZVF
 79 |   degenerates → report per-step reward variance or gradient-norm variance instead. Item 4
 80 |   mandates *a usable-signal trajectory*, of which ZVF/GU is the default.
 81 | 
 82 | ### 5. Group-size schedule (fixed or adaptive)
 83 | - **Report:** group size G at every step; the rule that changes it if adaptive (e.g. AERO-style
 84 |   double/halve on rolling ZVF).
 85 | - **Why it can flip:** G sets the mixed-group probability and was one of the strongest single
 86 |   knobs measured (G = 2→4→8→16 moved last-10 reward across a wide band). Adaptive-G vs. fixed-G
 87 |   partly credits the variant for spending more compute on hard prompts — separate that from the
 88 |   algorithmic claim.
 89 | - **Good:** "Fixed G=8 throughout." / "Adaptive G∈{4,…,16}, baseline 8, double when rolling
 90 |   ZVF>0.8, halve when <0.3, window 10."
 91 | - **Bad:** "Group size 8." (when it was actually adaptive)
 92 | 
 93 | ### 6. Held-out split distinct from the reward environment
 94 | - **Report:** a held-out slice **disjoint** from training prompts, scored by a harness, with N
 95 |   and a CI, reported **separately** from online training reward.
 96 | - **Why it can flip:** training reward is dynamics, not capability. A clean paired control
 97 |   improved only **82.0% → 83.3% (p=0.26)** [TODO:trace to v1 audit citation] despite near-saturated training reward; selecting
 98 |   checkpoints by training reward produced a spurious 87–95% "capability" band. Four 70B seeds
 99 |   ranged 95.0–98.1% on training last-10 [TODO:trace to v1 audit citation] yet all landed on 95.0% held-out (sampling noise, not
100 |   generalization).
101 | - **Good:** "Held-out = 500 GSM8K test problems, seed 0, disjoint from every training batch;
102 |   base 82.0% → post 83.3% [TODO:trace to v1 audit citation], Wilson 95% CI, paired per-prompt p=0.54 [TODO:trace to v1 audit citation]."</update>
103 | - **Bad:** Reporting training-set reward as the capability number; "accuracy 94%" with no split
104 |   stated.
105 | 
106 | ### 7. Decontamination probe results
107 | - **Report:** train/test contamination check (n-gram or embedding overlap between training
108 |   prompts and the held-out/benchmark slice) AND parser behavior on adversarial format-only
109 |   inputs.
110 | - **Why it can flip:** verifiable rewards still admit reward hacking — parser artifacts, format
111 |   shortcuts, length effects, train-prompt overfitting. Overlap → "gain" may be memorization;
112 |   parser rewards a format token → "gain" may be a shortcut.
113 | - **Good:** "Max 8-gram overlap between train and held-out = 0.0%; parser rejects 100% of
114 |   format-only (no-answer) adversarial inputs."
115 | - **Bad:** No contamination check; parser behavior unstated.
116 | 
117 | ---
118 | 
119 | ## Fillable appendix template (drop into your paper)
120 | 
121 | ```
122 | ================ MIN-REPORT-RL BLOCK ================
123 | Method label:            <e.g. GRPO / DAPO / GSPO / ...>
124 | Base checkpoint:         <model id + revision/hash>
125 | Tokenizer + chat tmpl:   <id; same in sampler and trainer? yes/no>
126 | 
127 | [1] LOSS FORM
128 |     PPO ratio used:      <yes/no>
129 |     Clip:                <none / symmetric eps=__ / asymmetric lo=__ hi=__>
130 |     Token mask:          <completion-only / whole-sequence>
131 |     Advantage norm.:     <per-group / per-batch / running; formula>
132 | 
133 | [2] REFERENCE + KL
134 |     Frozen reference:    <yes (=__ checkpoint) / no>
135 |     KL location:         <in loss / in reward / absent>
136 |     KL coeff + schedule: <beta=__, schedule=__>
137 |     KL estimator:        <forward/reverse, per-token/per-seq>
138 | 
139 | [3] SAMPLER / BACKEND / PRECISION
140 |     Rollout engine:      <vLLM/SGLang/managed API/trainer.generate; version>
141 |     Decoding:            <temp=__, top_p=__, max_tokens=__>
142 |     Precision:           <sampler=__, trainer=__>
143 | 
144 | [4] ZVF / GU TRAJECTORY
145 |     Per-step logged:     <yes/no; artifact link>
146 |     Mean ZVF@25:         <__>      Mean GU@25: <__>
147 |     Collapse rule trip:  <__ / S seeds>   (rule: ZVF>=80% & reward<=5% in first 5 steps)
148 |     (or substitute metric under dense/scaffolded rewards: __________)
149 | 
150 | [5] GROUP-SIZE SCHEDULE
151 |     Type:                <fixed G=__ / adaptive (rule=__, bounds=__, window=__)>
152 | 
153 | [6] HELD-OUT SPLIT
154 |     Slice:               <dataset/split, N=__, seed=__, disjoint from train? yes/no>
155 |     Base vs post:        <__% -> __%, CI=__, test=__ (p=__)>
156 |     Reported separately from training reward: <yes/no>
157 | 
158 | [7] DECONTAMINATION
159 |     Train/test overlap:  <metric=__, value=__>
160 |     Parser adversarial:  <format-only reject rate=__%>
161 | ====================================================
162 | ```
163 | 
164 | ---
165 | 
166 | ## One-line self-test
167 | If a reader cannot reproduce your ranking of method A vs. method B **from the block above
168 | alone**, the block is incomplete — find the unreported lever and add it.
````

---

## Pro 5.5 REPLY

Below is an invented tool spec for **MIN-REPORT-RL Auditor**, an automated compliance checker for GRPO-family papers and repos. It is designed around the paper’s core claim: a GRPO label is not a complete experimental description, because loss form, KL/reference handling, sampler/backend/precision, ZVF/GU telemetry, group-size schedule, held-out evaluation, and decontamination can each change the result. fileciteturn0file0

# MIN-REPORT-RL Auditor

**Purpose:** scan a GRPO-family paper, appendix, repo, logs, and release artifacts; extract evidence for the 7-item MIN-REPORT-RL checklist; assign a reproducibility-compliance score; and produce a reviewer-ready explanation of what is missing.

**Target methods:** GRPO, DAPO, GSPO, Dr.GRPO, MAD-GRPO, AERO, CPPO, NGRPO, Scaf-GRPO, and close variants.

**Inputs:**

```bash
minreport-audit \
  --paper paper.pdf \
  --repo ./code_release \
  --logs ./wandb_export_or_tensorboard \
  --mode strict \
  --out minreport_audit.json
```

**Outputs:**

1. `minreport_audit.json` — machine-readable score and evidence.
2. `MIN_REPORT_RL_REVIEW.md` — reviewer-facing summary.
3. `evidence/` — quoted paper spans, config snippets, code locations, log checks.
4. Optional badge: `MIN-REPORT-RL: Complete / Usable / Partial / Not Auditable`.

The checker should not merely ask whether an item is “mentioned.” It should ask whether the paper or repo reports enough concrete information for another reader to tell whether two GRPO stacks are comparable. That follows the paper’s definition of MIN-REPORT-RL as the smallest set of fields needed to detect confounding between GRPO-family comparisons. fileciteturn1file4

---

# 1. Scoring rubric

## Overall score

Each of the 7 checklist items receives **0–10 points**.

```text
Raw score = sum(item scores), max = 70
Normalized score = round(100 * raw / 70)
```

## Verdict bands

| Normalized score | Badge | Meaning |
|---:|---|---|
| 90–100 | **Complete** | All seven fields are concrete, auditable, and consistent across paper/repo/logs. |
| 75–89 | **Usable with caveats** | A reviewer can mostly reconstruct the stack, but some subfields or artifacts are weak. |
| 50–74 | **Partial** | The paper gives useful details, but multiple flip levers remain under-specified. |
| 25–49 | **Not reproducibility-auditable** | Mentions GRPO/RL details, but not enough to distinguish algorithm from stack. |
| 0–24 | **Non-compliant** | Mostly label-only reporting. |

## Per-item scoring scale

For every checklist item, the auditor assigns a score using this logic:

| Score | Meaning |
|---:|---|
| 10 | Complete: all required subfields present, explicit, and supported by paper/repo/log evidence. |
| 8 | Mostly complete: one minor subfield missing, but the stack is still interpretable. |
| 5 | Partial: key facts are present, but an important ambiguity remains. |
| 2 | Mention-only: item is named but not specified enough to reproduce or audit. |
| 0 | Absent, contradictory, or actively misleading. |

## Evidence confidence

The score and confidence are separate. A paper can score 8/10 with low confidence if the evidence comes from prose only and the repo contradicts it.

Evidence classes:

| Evidence class | Confidence effect |
|---|---|
| Canonical `MIN_REPORT_RL_BLOCK` plus matching repo/log artifacts | Highest |
| Explicit paper/appendix text with exact values | High |
| Config/code evidence with commit hash | High |
| Log-derived evidence | High for telemetry items |
| Inferred framework defaults | Low; never enough for full credit |
| “Standard GRPO,” “default settings,” “as usual” | Mention-only or zero |

## Hard caps

These prevent superficial compliance.

1. If the paper has no repo or artifact links, item 4, ZVF/GU trajectory, is capped at **5/10** unless the full trajectory is embedded in the appendix.
2. If held-out evaluation is not demonstrably disjoint from training prompts, item 6 is capped at **4/10**.
3. If contamination is only discussed conceptually but no probe result is reported, item 7 is capped at **3/10**.
4. If a closed/managed backend hides loss or KL internals, items 1–2 cannot exceed **4/10** unless the authors explicitly scope the claim as “platform implementation,” not generic GRPO. The paper itself makes this scoping point for closed runners. fileciteturn1file10
5. If paper and repo disagree, use the lower score and emit a contradiction warning.

---

# 2. System design

## Pipeline

**Stage 1 — Ingest**

The auditor parses:

```text
paper.pdf / paper.tex / appendix.md
README.md
configs/*.yaml, *.json, *.toml
training scripts
evaluation scripts
reward parser code
decontamination scripts
logs: wandb, tensorboard, CSV, JSONL
model cards / release notes
```

**Stage 2 — Detect GRPO-family scope**

It looks for method labels and variants:

```text
GRPO, DAPO, GSPO, Dr.GRPO, MAD-GRPO,
AERO, CPPO, NGRPO, Scaf-GRPO,
group relative, group-relative, num_generations,
group_size, per-group advantage, zero-variance fraction
```

**Stage 3 — Evidence extraction**

The auditor combines:

- regex dictionaries for common field names;
- AST/static-code analysis for Python training loops;
- config-key mapping for TRL, verl, OpenRLHF, custom trainers;
- log-schema detection;
- semantic extraction from prose and tables;
- consistency checks between paper, configs, and code.

**Stage 4 — Normalization**

All extracted evidence is normalized into a canonical schema:

```json
{
  "method_label": "GRPO",
  "base_checkpoint": "...",
  "tokenizer_chat_template": "...",
  "loss_form": {},
  "reference_kl": {},
  "sampler_backend_precision": {},
  "zvf_gu_trajectory": {},
  "group_size_schedule": {},
  "heldout_split": {},
  "decontamination": {}
}
```

**Stage 5 — Score and explain**

For every item, the auditor stores:

```json
{
  "score": 8,
  "confidence": 0.83,
  "evidence": [
    {
      "source": "paper",
      "location": "appendix B.2",
      "quote": "Token-masked completion-only..."
    },
    {
      "source": "repo",
      "path": "configs/grpo.yaml",
      "keys": ["loss.mask=completion_only", "clip_eps=0.2"]
    }
  ],
  "missing": ["advantage normalization formula"],
  "warnings": []
}
```

---

# 3. Heuristics for each checklist item

The paper’s seven items are treated as mandatory because each is a flip lever: loss form, KL/reference, sampler/backend/precision, ZVF/GU, group-size schedule, held-out split, and decontamination. fileciteturn1file3

## Item 1 — Loss form

**What must be reported**

Score out of 10:

| Subfield | Points |
|---|---:|
| PPO-style importance ratio used or explicitly absent | 2 |
| Clipping status and bounds, including asymmetric clip-higher | 2 |
| Token mask: completion-only vs whole-sequence | 2 |
| Advantage normalization: per-group, per-batch, running, or formula | 2 |
| Baseline and variant loss forms both specified | 2 |

**Paper heuristics**

Search for:

```text
importance ratio, old logprobs, pi_theta / pi_old,
clip, epsilon, cliprange, clip-higher, asymmetric clip,
completion-only, token mask, loss mask, response mask,
whole-sequence, prompt tokens,
advantage normalization, group mean, group std,
per-group, per-batch, running estimate,
Dr.GRPO, GSPO, DAPO
```

**Repo heuristics**

Look for config keys and code patterns:

```text
clip_range, cliprange, clip_eps, epsilon, clip_high, clip_low
use_ppo_ratio, importance_sampling, old_log_probs
completion_mask, response_mask, prompt_mask, loss_mask
advantages = (rewards - rewards.mean(...)) / rewards.std(...)
num_generations, group_rewards, normalize_advantages
torch.exp(logp - old_logp)
torch.clamp(ratio, 1-eps, 1+eps)
```

**Static-code checks**

- Detect whether `old_log_probs` are used.
- Detect whether the loss includes prompt tokens by tracing masks.
- Detect ratio clipping via `torch.clamp`, `minimum`, `maximum`, or custom clipping.
- Detect sequence-level vs token-level importance ratio for GSPO-like variants.

**False positives**

- “We use the standard GRPO loss” earns at most **2/10**.
- “We follow TRL defaults” earns at most **3/10** unless the repo pins the TRL version and confirms the actual loss path.
- A formula in the method section earns partial credit only if the baseline and implementation are consistent.

---

## Item 2 — Reference policy and KL handling

**What must be reported**

| Subfield | Points |
|---|---:|
| Frozen reference policy retained or explicitly absent | 2 |
| Reference checkpoint/revision if retained | 1 |
| KL location: loss, reward, or absent | 2 |
| KL coefficient and schedule | 2 |
| KL estimator: forward/reverse, per-token/per-sequence | 2 |
| Paper/repo consistency | 1 |

**Paper heuristics**

Search for:

```text
reference model, frozen reference, ref policy, old policy,
KL, KL penalty, KL regularization, beta, kl_coef,
in loss, in reward, reward shaping, non-score reward,
forward KL, reverse KL, per-token KL, sequence KL,
SFT anchor, dual KL, reference-free
```

**Repo heuristics**

Look for:

```text
ref_model, reference_model, create_reference_model
kl_coef, beta, init_kl_coef, target_kl
kl_penalty, kl_loss, non_score_reward
loss += beta * kl
reward -= beta * kl
ref_model = None
disable_reference, use_ref_model: false
```

**Placement classifier**

The auditor distinguishes:

```python
if kl appears inside reward computation:
    kl_location = "reward"
elif kl is added to objective/loss:
    kl_location = "loss"
elif beta == 0 or ref_model is None:
    kl_location = "absent"
else:
    kl_location = "ambiguous"
```

**False positives**

- “KL regularization as usual” earns at most **2/10**.
- Reporting only `beta=0.04` without saying where KL is applied earns at most **5/10**.
- A closed backend with hidden reference/KL internals cannot receive full credit.

---

## Item 3 — Sampler, backend, and precision

The paper highlights sampler/backend/precision because the sampler defines the rollout distribution, and backend differences can move GRPO trajectories under the same visible hyperparameters. fileciteturn1file12

**What must be reported**

| Subfield | Points |
|---|---:|
| Rollout engine and version | 2 |
| Decoding parameters: temperature, top-p/top-k, max tokens | 2 |
| Sampler precision and trainer precision | 2 |
| Tokenizer ID/revision and chat template | 2 |
| Same tokenizer/template in sampler and trainer, yes/no | 1 |
| Managed/backend-specific details or API version | 1 |

**Paper heuristics**

Search for:

```text
vLLM, SGLang, generate, rollout engine,
OpenAI API, managed API, inference backend,
temperature, top_p, top-k, max_tokens, max_new_tokens,
bf16, fp16, fp32, logits precision,
tokenizer, chat template, apply_chat_template,
revision, commit hash
```

**Repo heuristics**

Inspect:

```text
generation_config.json
tokenizer_config.json
chat_template.jinja
configs/rollout.yaml
vllm_engine_args
sglang_args
transformers.GenerationConfig
temperature, top_p, top_k, do_sample
dtype, torch_dtype, bf16, fp16, fp32
```

**Consistency checks**

- Does the paper say `temp=0.8` but config says `temperature: 1.0`?
- Does the sampler use vLLM while the trainer uses Hugging Face `generate`?
- Is the tokenizer loaded separately in sampler and trainer?
- Does the repo pin model revision or only a floating model name?

**False positives**

- “Generated with default settings” earns at most **2/10**.
- Naming vLLM without decoding parameters earns at most **4/10**.
- Naming precision only for training, not rollout sampling, earns at most **6/10**.

---

## Item 4 — Per-step ZVF and GU trajectory

The checker treats this as an artifact-heavy item. The paper defines ZVF as the fraction of prompts whose group completions all get identical reward and GU as `1 − ZVF`; it emphasizes that this trajectory reveals whether the run had usable group-relative signal. fileciteturn1file12

**What must be reported**

| Subfield | Points |
|---|---:|
| ZVF/GU definition or accepted substitute | 2 |
| Per-step trajectory artifact released | 3 |
| Summary stats, e.g. ZVF@25, GU@25 | 1.5 |
| Collapse rule or equivalent triage | 1 |
| Coverage for every arm and seed | 1.5 |
| Schema is machine-readable | 1 |

**Paper heuristics**

Search for:

```text
ZVF, zero-variance fraction,
GU, gradient utilization,
usable signal, reward variance,
per-step reward variance, gradient-norm variance,
collapse, first-five-step rule,
ZVF >= 80%, reward <= 5%
```

**Repo/log heuristics**

Look for logs with columns:

```text
step
prompt_id
group_id
completion_id
reward
zvf
gu
reward_variance
gradient_norm_variance
```

Acceptable artifact names:

```text
zvf_gu.csv
telemetry.jsonl
train_metrics.csv
wandb-history.jsonl
tensorboard event files
```

**Derived-metric computation**

If raw group rewards are available, the auditor computes ZVF itself:

```python
def compute_zvf(group_rewards_by_prompt):
    zero_var = 0
    total = 0
    for prompt_id, rewards in group_rewards_by_prompt.items():
        total += 1
        if max(rewards) == min(rewards):
            zero_var += 1
    zvf = zero_var / total
    gu = 1.0 - zvf
    return zvf, gu
```

**False positives**

- Reporting only final reward earns **0/10** for item 4.
- Reporting reward variance without prompt/group IDs earns at most **4/10**.
- A single plotted curve with no data artifact earns at most **6/10**.
- ZVF on one seed only, when multiple seeds are claimed, caps at **7/10**.

---

## Item 5 — Group-size schedule

The paper treats group size `G` as a major lever because it affects the probability of mixed-reward, gradient-usable groups. fileciteturn1file8

**What must be reported**

| Subfield | Points |
|---|---:|
| Fixed or adaptive schedule stated | 2 |
| `G` value at every step, or machine-readable schedule | 3 |
| Adaptive rule, bounds, and window if adaptive | 2 |
| Compute accounting: rollouts/tokens/prompts | 1.5 |
| Baseline and variant comparability | 1.5 |

**Paper heuristics**

Search for:

```text
group size, num generations, G=,
rollouts per prompt, samples per prompt,
adaptive group size, dynamic sampling,
AERO, double, halve, rolling ZVF,
bounds, window, schedule
```

**Repo heuristics**

Look for:

```text
num_generations
num_return_sequences
group_size
rollouts_per_prompt
n_samples_per_prompt
adaptive_g
min_group_size
max_group_size
zvf_window
```

**Schedule reconstruction**

- If raw logs include prompt IDs and completions, infer `G` per prompt per step.
- If adaptive, verify that schedule changes match the stated rule.
- If a variant uses dynamic sampling, check whether compute/token budget is normalized against the baseline.

**False positives**

- “Group size 8” earns full credit only if no adaptive code/log behavior contradicts it.
- If the method adaptively filters or resamples groups but reports only nominal `G`, cap at **5/10**.
- If DAPO-style dynamic sampling is present but no compute accounting is given, cap at **7/10**.

---

## Item 6 — Held-out split distinct from reward environment

The paper distinguishes online training reward from capability evidence and requires a held-out slice disjoint from training prompts, scored by a harness, with sample size and confidence interval. fileciteturn1file8

**What must be reported**

| Subfield | Points |
|---|---:|
| Dataset/split name, sample size, seed | 2 |
| Explicit train/eval disjointness | 2 |
| Evaluation harness and scoring parser | 2 |
| Training reward reported separately from held-out accuracy | 1.5 |
| Confidence interval or statistical test | 1.5 |
| Checkpoint selection rule | 1 |

**Paper heuristics**

Search for:

```text
held-out, test set, validation set, disjoint,
GSM8K test, MATH test, benchmark split,
evaluation harness, lm-eval, grader, parser,
confidence interval, CI, bootstrap, Wilson,
paired test, p-value, seed,
checkpoint selection, best checkpoint, last checkpoint
```

**Repo heuristics**

Inspect:

```text
eval.py
evaluate.py
lm_eval configs
dataset split loading
train.jsonl / test.jsonl
heldout_ids.txt
checkpoint selection scripts
```

**Disjointness verifier**

If datasets are included:

- Hash normalized prompts.
- Compare train and eval IDs.
- Run n-gram overlap as a weaker fallback.
- Flag if eval prompts appear in reward-training files.

**False positives**

- “Accuracy 94%” with no split earns at most **2/10**.
- Reporting training-set reward as “accuracy” earns **0–2/10**.
- Held-out score without N/CI caps at **7/10**.
- Selecting the best checkpoint by training reward without a stated rule triggers a warning.

---

## Item 7 — Decontamination probe results

The paper requires both train/test contamination checks and parser behavior on adversarial format-only inputs, because reward hacking can come from overlap, parser artifacts, format shortcuts, length effects, or train-prompt overfitting. fileciteturn1file5

**What must be reported**

| Subfield | Points |
|---|---:|
| Train/test overlap method: n-gram, MinHash, embedding, etc. | 2 |
| Quantitative overlap result | 2 |
| Threshold or decision rule | 1 |
| Parser adversarial format-only tests | 2 |
| Parser reject/accept rates | 1.5 |
| Scripts or fixtures released | 1 |
| Remediation for contaminated cases | 0.5 |

**Paper heuristics**

Search for:

```text
decontamination, contamination, overlap,
n-gram, 8-gram, MinHash, embedding similarity,
train/test leakage, duplicate prompts,
parser adversarial, format-only, no-answer,
reward hacking, shortcut, parser rejects,
answer extractor, format token
```

**Repo heuristics**

Look for:

```text
decontam.py
overlap.py
ngram_overlap
minhash
faiss
sentence-transformers
parser_tests.py
test_reward_parser.py
format_only_cases.jsonl
adversarial_parser_inputs.jsonl
```

**Parser probe checks**

The auditor generates or loads cases like:

```json
{
  "input": "####",
  "model_output": "Therefore the answer is \\boxed{}",
  "expected_reward": 0,
  "reason": "format-only, no answer"
}
```

Then it runs the reward parser if available and flags cases where format-only outputs receive positive reward.

**False positives**

- “We use the official test set” is not a decontamination probe.
- “No contamination was found” without method/result earns at most **3/10**.
- Overlap check without parser adversarial tests caps at **6/10**.
- Parser tests without train/test overlap caps at **6/10**.

---

# 4. Canonical output schema

```json
{
  "tool": "MIN-REPORT-RL Auditor",
  "version": "0.1",
  "target": {
    "paper": "paper.pdf",
    "repo": "https://github.com/example/grpo-release",
    "commit": "7fd31c2",
    "method_labels_detected": ["GRPO", "DAPO"]
  },
  "overall": {
    "raw": 58.5,
    "max": 70,
    "normalized": 84,
    "badge": "Usable with caveats",
    "confidence": 0.81
  },
  "items": {
    "1_loss_form": {
      "score": 8,
      "confidence": 0.86,
      "status": "mostly_complete",
      "missing": ["baseline advantage normalization formula"],
      "warnings": []
    },
    "2_reference_kl": {
      "score": 9,
      "confidence": 0.91,
      "status": "mostly_complete",
      "missing": [],
      "warnings": ["paper says reverse KL; code variable name says approx_kl, verified as reverse estimator"]
    }
  },
  "blocking_issues": [
    "No released adversarial parser fixtures; item 7 capped at 8."
  ],
  "reviewer_summary": "The stack is mostly auditable. The main missing pieces are parser adversarial fixtures and the exact baseline advantage-normalization formula."
}
```

---

# 5. Example output A — strong paper plus repo

## CLI summary

```text
MIN-REPORT-RL Auditor v0.1
Target: dapo_math_release
Mode: strict

Overall: 93/100
Raw: 65.0 / 70
Badge: COMPLETE
Confidence: 0.89

[1] Loss form                         9.0/10  PASS
[2] Reference policy + KL handling    9.0/10  PASS
[3] Sampler / backend / precision     10.0/10 PASS
[4] Per-step ZVF / GU trajectory      9.0/10  PASS
[5] Group-size schedule               9.0/10  PASS
[6] Held-out split                    10.0/10 PASS
[7] Decontamination probes            9.0/10  PASS

Main caveat:
- Item 4: ZVF/GU trajectory is released for all seeds, but only mean ZVF@25 is summarized in the paper; GU@25 is present in the CSV.
```

## Reviewer-facing summary

```md
# MIN-REPORT-RL Review

Verdict: COMPLETE, 93/100.

The paper is reproducibility-auditable. It reports the GRPO-family stack with enough detail
to distinguish the algorithmic delta from the stack: loss ratio/clipping/masking/advantage
normalization, frozen reference and KL placement, vLLM rollout configuration, sampler/trainer
precision, ZVF/GU telemetry, group-size schedule, held-out disjoint evaluation, and decontamination.

Remaining minor issue:
- Add GU@25 to the paper table instead of requiring readers to open telemetry.csv.
```

## JSON excerpt

```json
{
  "overall": {
    "normalized": 93,
    "badge": "Complete",
    "confidence": 0.89
  },
  "items": {
    "1_loss_form": {
      "score": 9,
      "evidence": [
        {
          "source": "paper",
          "location": "Appendix A.1",
          "text": "PPO ratio with asymmetric clip_low=0.2, clip_high=0.28; completion-only mask; per-group advantage normalization."
        },
        {
          "source": "repo",
          "path": "configs/dapo_math.yaml",
          "keys": {
            "loss.use_ratio": true,
            "loss.clip_low": 0.2,
            "loss.clip_high": 0.28,
            "loss.mask": "completion_only",
            "loss.adv_norm": "per_group"
          }
        }
      ],
      "missing": []
    },
    "4_zvf_gu": {
      "score": 9,
      "evidence": [
        {
          "source": "logs",
          "path": "artifacts/telemetry/zvf_gu_seed*.csv",
          "columns": ["step", "seed", "zvf", "gu", "mean_reward"]
        }
      ],
      "derived": {
        "mean_zvf_at_25": 0.43,
        "mean_gu_at_25": 0.57,
        "collapse_rule_trips": "0/8 seeds"
      },
      "missing": ["GU@25 not printed in paper table"]
    }
  }
}
```

---

# 6. Example output B — typical “standard GRPO” paper, no repo

```text
MIN-REPORT-RL Auditor v0.1
Target: grpo_reasoning_paper.pdf
Mode: paper-only

Overall: 31/100
Raw: 21.5 / 70
Badge: NOT REPRODUCIBILITY-AUDITABLE
Confidence: 0.42

[1] Loss form                         2.0/10  FAIL
    Evidence: "We use the standard GRPO objective."
    Missing: ratio yes/no, clip bounds, token mask, advantage norm.

[2] Reference policy + KL handling    2.5/10  FAIL
    Evidence: "KL regularization is applied."
    Missing: frozen reference, KL location, beta schedule, estimator.

[3] Sampler / backend / precision     4.0/10  PARTIAL
    Evidence: "We sample 8 responses at temperature 0.7."
    Missing: rollout engine, top_p, max_tokens, precision, tokenizer/chat-template identity.

[4] Per-step ZVF / GU trajectory      0.0/10  FAIL
    Missing: no ZVF, GU, reward variance, or group-signal trajectory.

[5] Group-size schedule               4.0/10  PARTIAL
    Evidence: "G=8."
    Missing: per-step schedule; no confirmation fixed vs adaptive.

[6] Held-out split                    5.0/10  PARTIAL
    Evidence: "GSM8K test accuracy is reported."
    Missing: disjointness statement, CI, checkpoint selection rule.

[7] Decontamination probes            4.0/10  PARTIAL
    Evidence: "We remove duplicate train/test examples."
    Missing: overlap metric, quantitative result, parser adversarial tests.

Blocking reviewer note:
This paper should not be treated as an auditable GRPO comparison. The algorithm label is
reported, but most stack fields that MIN-REPORT-RL treats as flip levers are absent.
```

## Suggested author fixes

```md
Required before reproducibility review:
1. Add a MIN-REPORT-RL block to the appendix.
2. State whether the loss uses an old-policy ratio, clipping, completion-only masking, and per-group normalization.
3. State whether KL is in the loss, reward, or absent.
4. Release per-step group-reward telemetry or ZVF/GU CSV.
5. Report held-out N, CI, disjointness, and checkpoint selection.
6. Add train/test overlap and reward-parser adversarial probes.
```

---

# 7. Example output C — repo found, paper underspecified

```text
MIN-REPORT-RL Auditor v0.1
Target: mad-grpo-code-release
Mode: paper+repo

Overall: 76/100
Raw: 53.0 / 70
Badge: USABLE WITH CAVEATS
Confidence: 0.78

The paper itself is incomplete, but the repo makes the stack partially reconstructable.

Item scores:
[1] Loss form                         8/10
[2] Reference policy + KL handling    7/10
[3] Sampler / backend / precision     8/10
[4] Per-step ZVF / GU trajectory      5/10
[5] Group-size schedule               9/10
[6] Held-out split                    8/10
[7] Decontamination probes            8/10

Contradictions:
- Paper says "completion-only loss"; repo default is `loss_mask: full_sequence` in `configs/base.yaml`.
- Experiment override `configs/mad_grpo_math.yaml` changes this to `completion_only`, so the run is probably compliant, but the paper should cite the exact config.

Caps applied:
- Item 4 capped at 5 because only aggregate ZVF@25 is reported; no per-step trajectory artifact is released.
```

## JSON excerpt

```json
{
  "overall": {
    "normalized": 76,
    "badge": "Usable with caveats",
    "confidence": 0.78
  },
  "contradictions": [
    {
      "field": "loss_form.token_mask",
      "paper": "completion-only",
      "repo_default": "full_sequence",
      "experiment_override": "completion_only",
      "resolution": "experiment override wins, but paper should cite config path"
    }
  ],
  "caps": [
    {
      "item": "4_zvf_gu",
      "cap": 5,
      "reason": "No released per-step trajectory."
    }
  ]
}
```

---

# 8. Example output D — closed managed backend

```text
MIN-REPORT-RL Auditor v0.1
Target: managed_api_grpo_study
Mode: strict

Overall: 49/100
Raw: 34.5 / 70
Badge: NOT REPRODUCIBILITY-AUDITABLE
Confidence: 0.55

Important scoping warning:
The work uses a closed managed RL backend. The authors disclose this, but the loss form and
reference/KL implementation are not inspectable. The result should be scoped as:
"ManagedBackend-X GRPO implementation"
not:
"GRPO generally."

Scores:
[1] Loss form                         4/10  capped: closed backend
[2] Reference policy + KL handling    4/10  capped: closed backend
[3] Sampler / backend / precision     7/10
[4] Per-step ZVF / GU trajectory      0/10
[5] Group-size schedule               7/10
[6] Held-out split                    8/10
[7] Decontamination probes            4.5/10

Reviewer note:
This may still be a useful platform result, but it is not enough evidence for an algorithmic
GRPO-family claim unless the hidden stack fields are disclosed or the claim is explicitly scoped.
```

---

# 9. Anti-gaming rules

The checker should be conservative because the paper’s whole point is that visible hyperparameters are not enough.

## Ambiguous prose penalties

| Phrase | Maximum credit |
|---|---:|
| “standard GRPO loss” | 2/10 on item 1 |
| “KL as usual” | 2/10 on item 2 |
| “default generation settings” | 2/10 on item 3 |
| “we report reward curves” | 3/10 on item 4 unless ZVF/GU or substitute is present |
| “G=8” with adaptive code present | 5/10 on item 5 |
| “evaluated on GSM8K” | 4/10 on item 6 unless split/disjointness/N/CI are present |
| “no contamination” | 3/10 on item 7 unless method and result are reported |

## Contradiction handling

When paper and repo disagree:

```text
item_score = min(paper_score, repo_score_after_resolution)
warning = "paper/repo contradiction"
```

If a launch config overrides a default, the checker accepts the launch config only if it can identify the exact experiment command, config merge order, or run artifact.

## Default handling

Framework defaults count only when all are true:

1. framework and version are pinned;
2. the exact trainer class is identified;
3. the repo does not override the default;
4. the default is documented or visible in code.

Otherwise defaults are treated as uncertain.

---

# 10. Recommended canonical artifact

The easiest way for papers to pass the checker is to ship this file:

```text
artifacts/min_report_rl.json
```

Minimal schema:

```json
{
  "method_label": "GRPO",
  "base_checkpoint": {
    "id": "Qwen/Qwen3-8B",
    "revision": "..."
  },
  "tokenizer_chat_template": {
    "tokenizer_id": "...",
    "chat_template_hash": "...",
    "same_sampler_trainer": true
  },
  "loss_form": {
    "ppo_ratio_used": true,
    "clip": {"type": "symmetric", "epsilon": 0.2},
    "token_mask": "completion_only",
    "advantage_norm": {
      "scope": "per_group",
      "formula": "(r - mean_group) / std_group"
    }
  },
  "reference_kl": {
    "frozen_reference": true,
    "reference_checkpoint": "SFT checkpoint hash",
    "kl_location": "loss",
    "kl_coeff": 0.04,
    "kl_schedule": "constant",
    "kl_estimator": "reverse_per_token"
  },
  "sampler_backend_precision": {
    "rollout_engine": "vLLM",
    "rollout_engine_version": "0.x",
    "decoding": {"temperature": 0.8, "top_p": 1.0, "max_tokens": 512},
    "precision": {"sampler_logits": "bf16", "trainer_logits": "bf16"}
  },
  "zvf_gu_trajectory": {
    "artifact": "telemetry/zvf_gu.csv",
    "per_step_logged": true,
    "mean_zvf_at_25": 0.43,
    "mean_gu_at_25": 0.57,
    "collapse_rule_trips": "0/8"
  },
  "group_size_schedule": {
    "type": "fixed",
    "G": 8,
    "artifact": "telemetry/group_size_by_step.csv"
  },
  "heldout_split": {
    "dataset": "GSM8K",
    "split": "test",
    "N": 500,
    "seed": 0,
    "disjoint_from_train": true,
    "harness": "eval/evaluate_gsm8k.py",
    "base_score": 0.82,
    "post_score": 0.833,
    "ci": "Wilson 95%",
    "reported_separately_from_training_reward": true
  },
  "decontamination": {
    "overlap_metric": "8-gram overlap",
    "overlap_value": 0.0,
    "parser_adversarial_artifact": "tests/format_only_cases.jsonl",
    "format_only_reject_rate": 1.0
  }
}
```

This matches the paper’s adoption idea: trainers should emit the seven-field block and per-step telemetry by default, because the compliance cost should be close to a JSON block plus logged trajectories. fileciteturn1file1

---
