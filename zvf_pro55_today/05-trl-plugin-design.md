# TRL plugin design
- ID `6a379a9c-0c40-83ee-b032-422b1df0c4fe` created 2026-06-21 08:02 UTC | model gpt-5-5-pro

---

## QUERY

Read the attached MIN-REPORT-RL position paper. Invent a TRL plugin that automatically emits the 7-item minimum-reportable-stack block as JSON at the start of every GRPO run and logs per-step ZVF/GU. Specify the hooks, config flags, and output schema.

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

---

## Pro 5.5 REPLY

## Plugin: `trl-min-report-rl`

I’d implement this as a small TRL extension named **`MinReportRLPlugin`**: a `TrainerCallback` for start/end/logging events plus one GRPO-specific reward hook that sees the per-prompt reward matrix before advantages are normalized. That split matters because standard trainer callbacks can emit a manifest, but they do not reliably receive the group reward tensor needed to compute ZVF/GU.

The attached paper’s natural contract is: **write the seven-item minimum-reportable-stack block once at run start, then write item-4 telemetry every step**. The page-6 checklist defines the seven mandatory stack fields, and the page-7 telemetry table makes ZVF/GU the per-arm/per-step signal to log. fileciteturn0file0 Current TRL is already close to this: `GRPOTrainer` supports callbacks and custom rollout plumbing, and `GRPOConfig` exposes the relevant knobs such as `num_generations`, vLLM settings, loss type, reward scaling, KL coefficient, and generation parameters. citeturn295757view1turn965512view1turn965512view0

---

## User-facing API

```python
from trl import GRPOConfig
from trl_min_report_rl import MinReportGRPOTrainer, MinReportRLConfig

training_args = GRPOConfig(
    output_dir="runs/qwen3_gsm8k_grpo",
    num_generations=8,
    loss_type="dapo",
    beta=0.0,
    use_vllm=True,
    vllm_mode="colocate",
    temperature=1.0,
    top_p=1.0,
    report_to=["wandb", "tensorboard"],
)

min_report = MinReportRLConfig(
    enabled=True,
    emit_manifest_on_train_begin=True,
    log_zvf_gu_every_step=True,
    reward_equal_atol=0.0,
    reward_equal_rtol=0.0,
    fail_on_missing_required=False,
    heldout=MinReportRLConfig.Heldout(
        name="gsm8k_test_500",
        split="test",
        reward_environment_disjoint=True,
        evaluator="lm-eval-harness:gsm8k_exact_match",
        confidence_interval="bootstrap_95",
    ),
    decontamination=MinReportRLConfig.Decontamination(
        ngram_overlap_path="artifacts/decontam/ngram_overlap.json",
        parser_probe_path="artifacts/decontam/parser_adversarial.json",
    ),
)

trainer = MinReportGRPOTrainer(
    model="Qwen/Qwen3-8B",
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=heldout_ds,
    reward_funcs=[gsm8k_reward],
    min_report_rl=min_report,
)

trainer.train()
```

It writes:

```text
output_dir/
  min_report_rl/
    min_report_rl.json
    min_report_rl.sha256
    zvf_gu.jsonl
    final_summary.json
```

It also logs scalar metrics through TRL’s existing logging path. TRL already supports experiment trackers through `report_to`, including Trackio, W&B, and TensorBoard, so the plugin should reuse that instead of inventing a separate tracker interface. citeturn295757view2

---

## Hooks

| Hook | Location | Purpose |
|---|---|---|
| `GRPOTrainer.__init__` wrapper | Plugin subclass | Capture `GRPOConfig`, model/tokenizer identity, `peft_config`, reward funcs, rollout backend, dataset fingerprints, and attach the emitter. |
| `TrainerCallback.on_train_begin` | Standard HF callback | Emit `min_report_rl.json` **before the first rollout**; log `min_report_rl/manifest_sha256`. Transformers callbacks expose `on_train_begin`, `on_step_end`, `on_log`, `on_save`, and `on_train_end` lifecycle hooks. citeturn447619search1 |
| `on_grpo_rewards_ready` | New GRPO-specific hook | Called immediately after reward aggregation and before advantage normalization/loss. Receives `prompt_ids`, `completion_ids`, `reward_matrix`, `reward_components`, `group_size`, `split`, and `global_step`. |
| `TrainerCallback.on_log` | Standard callback | Adds latest ZVF/GU scalars to normal TRL logs when tracker logging fires. |
| `TrainerCallback.on_save` | Standard callback | Copies the manifest into each checkpoint folder and snapshots telemetry summary. |
| `TrainerCallback.on_train_end` | Standard callback | Writes `final_summary.json`, including mean ZVF/GU, first-five collapse triage, and missing-field warnings. |

The one new hook should be upstreamed into `GRPOTrainer` as:

```python
def on_grpo_rewards_ready(
    self,
    *,
    split: str,
    prompt_ids: list[str],
    completion_ids: list[str],
    rewards: "torch.Tensor",        # shape [num_prompts, G]
    reward_components: dict | None, # optional raw per reward function
    group_size: int,
) -> None:
    if self.min_report_rl_emitter is not None:
        self.min_report_rl_emitter.log_step(
            trainer=self,
            split=split,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            rewards=rewards,
            reward_components=reward_components,
            group_size=group_size,
        )
```

For a no-fork package, `MinReportGRPOTrainer` subclasses `GRPOTrainer` and inserts that call at the reward-scoring site. For an upstream TRL PR, this becomes a native `GRPOConfig` feature flag.

---

## Config flags

```python
@dataclass
class MinReportRLConfig:
    enabled: bool = True
    emit_manifest_on_train_begin: bool = True
    manifest_filename: str = "min_report_rl.json"
    telemetry_filename: str = "zvf_gu.jsonl"
    final_summary_filename: str = "final_summary.json"

    emit_to_file: bool = True
    emit_to_trainer_log: bool = True
    emit_to_trackers: bool = True
    log_zvf_gu_every_step: bool = True
    tracker_metric_prefix: str = "min_report_rl/"

    reward_source: Literal["post_aggregation", "per_reward_func", "both"] = "post_aggregation"
    reward_equal_atol: float = 0.0
    reward_equal_rtol: float = 0.0
    nan_policy: Literal["exclude_group", "zero_variance", "error"] = "exclude_group"
    group_key: Literal["prompt_hash", "dataset_index", "batch_index"] = "prompt_hash"

    store_prompt_text: bool = False
    hash_prompt_text: bool = True
    hash_completion_text: bool = False

    collapse_zvf_threshold: float = 0.80
    collapse_reward_threshold: float = 0.05
    collapse_window_steps: int = 5

    fail_on_missing_required: bool = False
    missing_policy: Literal["emit_unknown", "warn", "error"] = "warn"

    heldout: Heldout | None = None
    decontamination: Decontamination | None = None
```

Suggested `GRPOConfig` flags after upstreaming:

```python
report_min_report_rl: bool = True
min_report_rl_log_zvf_gu_every_step: bool = True
min_report_rl_fail_on_missing_required: bool = False
min_report_rl_reward_equal_atol: float = 0.0
min_report_rl_reward_equal_rtol: float = 0.0
min_report_rl_store_prompt_text: bool = False
min_report_rl_decontamination_config: str | None = None
min_report_rl_heldout_config: str | None = None
```

CLI equivalents:

```bash
--report_min_report_rl true
--min_report_rl_log_zvf_gu_every_step true
--min_report_rl_fail_on_missing_required false
--min_report_rl_reward_equal_atol 0.0
--min_report_rl_store_prompt_text false
--min_report_rl_heldout_config configs/heldout.json
--min_report_rl_decontamination_config artifacts/decontam/probes.json
```

---

## ZVF/GU computation

For each prompt \(x\), TRL samples \(G\) completions and scores them with reward functions before forming the group-relative advantage; that is exactly the matrix the plugin needs. citeturn295757view0

```python
def compute_zvf_gu(reward_matrix, *, atol=0.0, rtol=0.0, nan_policy="exclude_group"):
    """
    reward_matrix: shape [num_prompts, G]
    ZVF = fraction of prompt groups whose G rewards are all identical.
    GU  = 1 - ZVF.
    """
    valid_groups = []
    zero_variance = []

    for rewards in reward_matrix:
        finite = rewards[torch.isfinite(rewards)]

        if len(finite) != len(rewards):
            if nan_policy == "error":
                raise ValueError("NaN/Inf reward in GRPO group")
            if nan_policy == "exclude_group":
                continue

        if len(finite) < 2:
            continue

        same = torch.isclose(finite, finite[0], atol=atol, rtol=rtol).all()
        valid_groups.append(True)
        zero_variance.append(bool(same))

    n = len(valid_groups)
    z = sum(zero_variance)
    zvf = z / n if n else None
    gu = 1.0 - zvf if zvf is not None else None

    return {
        "zvf": zvf,
        "gu": gu,
        "zero_variance_groups": z,
        "usable_groups": None if n == 0 else n - z,
        "valid_groups": n,
    }
```

For binary verifiable rewards, `atol=0` is correct. For dense process rewards, set a tolerance or use `reward_std`/gradient-variance fallback; the manifest records that choice rather than pretending ZVF means the same thing in every reward regime.

In distributed training, the hook must compute ZVF/GU after gathering the full reward groups across processes, not on local shards, because a group can be split across devices. TRL’s GRPO source already has custom batching/sampling logic for generation batches, so the plugin should attach at the post-gather reward point, not inside the reward function itself. citeturn470275view2

---

## Run-start manifest schema: `min_report_rl.json`

The manifest uses explicit `unknown`, `not_applicable`, or `not_configured` statuses. Missing fields are never silently omitted.

```json
{
  "schema_version": "min-report-rl.run.v0.1",
  "run": {
    "run_id": "string",
    "created_at_utc": "string",
    "output_dir": "string",
    "seed": "integer|null",
    "global_step_at_emit": 0,
    "manifest_sha256": "string"
  },
  "emitter": {
    "package": "trl-min-report-rl",
    "version": "string",
    "mode": "subclass|upstream|monkeypatch",
    "missing_policy": "emit_unknown|warn|error"
  },
  "framework": {
    "trainer": "trl.GRPOTrainer",
    "trl_version": "string",
    "transformers_version": "string",
    "accelerate_version": "string",
    "torch_version": "string",
    "python_version": "string",
    "report_to": ["string"]
  },
  "model_stack": {
    "policy_model": {
      "id_or_path": "string",
      "revision": "string|unknown",
      "model_class": "string",
      "config_sha256": "string|unknown"
    },
    "tokenizer_or_processor": {
      "id_or_path": "string",
      "revision": "string|unknown",
      "class": "string",
      "padding_side": "left|right|unknown",
      "chat_template_sha256": "string|unknown"
    },
    "peft": {
      "enabled": "boolean",
      "type": "lora|qlora|none|unknown",
      "rank": "integer|null",
      "alpha": "number|null",
      "dropout": "number|null",
      "target_modules": ["string"],
      "target_parameters": ["string"],
      "bias": "string|unknown"
    },
    "hardware_precision": {
      "num_processes": "integer",
      "distributed_type": "string",
      "device_names": ["string"],
      "bf16": "boolean",
      "fp16": "boolean",
      "tf32": "boolean|unknown",
      "trainer_compute_dtype": "bf16|fp16|fp32|unknown",
      "sampler_logit_dtype": "bf16|fp16|fp32|unknown",
      "cast_lm_head_to_fp32": "boolean"
    }
  },
  "minimum_reportable_stack": {
    "1_loss_form": {
      "algorithm_label": "GRPO",
      "loss_type": "grpo|dapo|dr_grpo|bnpo|cispo|sapo|luspo|vespo|custom|unknown",
      "policy_ratio": {
        "uses_importance_ratio": "boolean|unknown",
        "importance_sampling_level": "token|sequence|none|unknown",
        "num_iterations": "integer"
      },
      "clip": {
        "enabled": "boolean|unknown",
        "epsilon_low": "number|null",
        "epsilon_high": "number|null",
        "delta": "number|null",
        "vllm_importance_sampling_correction": "boolean|null",
        "vllm_importance_sampling_mode": "string|null",
        "vllm_importance_sampling_clip_min": "number|null",
        "vllm_importance_sampling_clip_max": "number|null"
      },
      "token_mask": {
        "scope": "completion_only|whole_sequence|custom|unknown",
        "mask_truncated_completions": "boolean",
        "top_entropy_quantile": "number|null"
      },
      "advantage_normalization": {
        "scale_rewards": "group|batch|none|true|false|unknown",
        "multi_objective_aggregation": "sum_then_normalize|normalize_then_sum|custom|unknown",
        "reward_weights": ["number"]
      }
    },
    "2_reference_policy_kl": {
      "reference_policy": {
        "present": "boolean",
        "frozen": "boolean|unknown",
        "id_or_path": "string|none|unknown",
        "sync_ref_model": "boolean",
        "ref_model_sync_steps": "integer|null",
        "ref_model_mixup_alpha": "number|null"
      },
      "kl": {
        "enabled": "boolean",
        "placement": "loss|reward|absent|custom|unknown",
        "beta": "number",
        "schedule": "constant|custom|none|unknown",
        "estimator": "forward_approx|reverse|custom|unknown",
        "granularity": "per_token|per_sequence|unknown",
        "use_bias_correction_kl": "boolean"
      }
    },
    "3_sampler_backend_precision": {
      "rollout_engine": "vllm|transformers.generate|custom_rollout_func|managed_api|unknown",
      "backend": {
        "use_vllm": "boolean",
        "vllm_mode": "server|colocate|null",
        "vllm_model_impl": "vllm|transformers|null",
        "vllm_server_base_url": "string|null",
        "vllm_tensor_parallel_size": "integer|null",
        "vllm_gpu_memory_utilization": "number|null",
        "custom_rollout_func": "string|null"
      },
      "decoding": {
        "temperature": "number|null",
        "top_p": "number|null",
        "top_k": "integer|null",
        "min_p": "number|null",
        "max_completion_length": "integer|null",
        "repetition_penalty": "number|null",
        "generation_kwargs": {}
      },
      "tokenizer_template_shared": {
        "sampler_trainer_same_tokenizer": "boolean|unknown",
        "sampler_trainer_same_chat_template": "boolean|unknown"
      }
    },
    "4_zvf_gu_trajectory": {
      "enabled": "boolean",
      "formula": "zvf = zero_variance_groups / valid_groups; gu = 1 - zvf",
      "reward_source": "post_aggregation|per_reward_func|both",
      "group_key": "prompt_hash|dataset_index|batch_index",
      "reward_equal_atol": "number",
      "reward_equal_rtol": "number",
      "nan_policy": "exclude_group|zero_variance|error",
      "telemetry_path": "string",
      "tracker_metric_prefix": "string",
      "collapse_rule": {
        "window_steps": "integer",
        "zvf_threshold": "number",
        "reward_threshold": "number"
      }
    },
    "5_group_size_schedule": {
      "initial_group_size": "integer",
      "num_generations": "integer",
      "num_generations_eval": "integer|null",
      "schedule_type": "fixed|adaptive|custom|unknown",
      "adaptive_rule": "string|null",
      "per_step_group_size_logged": "boolean"
    },
    "6_heldout_split": {
      "status": "configured_verified|configured_unverified|not_configured|unknown",
      "train_dataset": {
        "name": "string|unknown",
        "split": "string|unknown",
        "fingerprint": "string|unknown",
        "num_rows": "integer|null"
      },
      "heldout_dataset": {
        "name": "string|unknown",
        "split": "string|unknown",
        "fingerprint": "string|unknown",
        "num_rows": "integer|null"
      },
      "reward_environment_disjoint": "boolean|unknown",
      "evaluator": "string|unknown",
      "confidence_interval": "bootstrap_95|wald_95|none|unknown"
    },
    "7_decontamination_probe": {
      "status": "passed|failed|not_run|not_configured|unknown",
      "ngram_overlap": {
        "path": "string|null",
        "max_overlap": "number|null",
        "threshold": "number|null",
        "status": "passed|failed|not_run|unknown"
      },
      "embedding_overlap": {
        "path": "string|null",
        "max_similarity": "number|null",
        "threshold": "number|null",
        "status": "passed|failed|not_run|unknown"
      },
      "parser_adversarial_probe": {
        "path": "string|null",
        "format_only_reward_rate": "number|null",
        "status": "passed|failed|not_run|unknown"
      }
    }
  }
}
```

---

## Per-step telemetry schema: `zvf_gu.jsonl`

Each line is one JSON object. It is written every optimizer step, even when normal `logging_steps` is larger.

```json
{
  "schema_version": "min-report-rl.telemetry.v0.1",
  "run_id": "string",
  "split": "train|eval",
  "global_step": "integer",
  "optimizer_step": "integer",
  "generation_step": "integer",
  "time_utc": "string",
  "group_size": "integer",
  "prompt_groups": {
    "valid_groups": "integer",
    "invalid_groups": "integer",
    "zero_variance_groups": "integer",
    "usable_groups": "integer",
    "all_correct_groups": "integer|null",
    "all_incorrect_groups": "integer|null"
  },
  "metrics": {
    "zvf": "number|null",
    "gu": "number|null",
    "reward_mean": "number|null",
    "reward_std": "number|null",
    "reward_min": "number|null",
    "reward_max": "number|null",
    "nonfinite_reward_count": "integer"
  },
  "reward_source": {
    "aggregation": "post_aggregation|per_reward_func|both",
    "reward_func_names": ["string"],
    "reward_weights": ["number"]
  },
  "collapse_triage": {
    "within_first_window": "boolean",
    "collapse_flag": "boolean",
    "rule": "zvf >= threshold and reward_mean <= threshold"
  },
  "privacy": {
    "prompt_text_stored": false,
    "prompt_hashes_sha256": ["string"]
  }
}
```

Tracker scalar keys:

```text
min_report_rl/zvf
min_report_rl/gu
min_report_rl/group_size
min_report_rl/zero_variance_groups
min_report_rl/usable_groups
min_report_rl/reward_mean
min_report_rl/reward_std
min_report_rl/collapse_flag
```

---

## Minimal upstream patch shape

```python
# trl/trainer/grpo_config.py
@dataclass
class GRPOConfig(...):
    report_min_report_rl: bool = True
    min_report_rl_log_zvf_gu_every_step: bool = True
    min_report_rl_fail_on_missing_required: bool = False
    min_report_rl_reward_equal_atol: float = 0.0
    min_report_rl_reward_equal_rtol: float = 0.0
    min_report_rl_store_prompt_text: bool = False
    min_report_rl_heldout_config: str | None = None
    min_report_rl_decontamination_config: str | None = None
```

```python
# trl/trainer/grpo_trainer.py
class GRPOTrainer(...):
    def __init__(..., min_report_rl: MinReportRLConfig | None = None, ...):
        ...
        if args.report_min_report_rl:
            self.min_report_rl_emitter = MinReportRLEmitter.from_trainer(
                trainer=self,
                config=min_report_rl,
            )
            self.add_callback(MinReportRLCallback(self.min_report_rl_emitter))
        else:
            self.min_report_rl_emitter = None

    def _after_rewards_aggregated(...):
        ...
        if self.min_report_rl_emitter is not None:
            self.min_report_rl_emitter.log_step(
                trainer=self,
                split="train" if self.model.training else "eval",
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                rewards=aggregated_rewards.reshape(num_prompts, self.num_generations),
                reward_components=raw_rewards_by_func,
                group_size=self.num_generations,
            )
```

That PR is small because TRL already owns the necessary stack fields: `GRPOConfig` records generation settings, vLLM mode, KL coefficient, reward scaling, clipping/loss knobs, and group size; the trainer receives callbacks and reward/rollout functions; and existing `report_to` handles external logging. citeturn965512view1turn351594view0turn295757view2

---

## Important honesty behavior

The plugin should **not** fake compliance. If a closed sampler does not expose logit precision, the field is emitted as:

```json
"sampler_logit_dtype": "unknown"
```

If no held-out split is configured:

```json
"6_heldout_split": {
  "status": "not_configured",
  "reward_environment_disjoint": "unknown"
}
```

If no decontamination probe is provided:

```json
"7_decontamination_probe": {
  "status": "not_configured"
}
```

With `min_report_rl_fail_on_missing_required=true`, those become hard errors before the first rollout. That makes the paper’s adoption claim concrete: the default path costs one JSON block and one JSONL telemetry stream; strict mode turns the standard into an auditable gate.

---
