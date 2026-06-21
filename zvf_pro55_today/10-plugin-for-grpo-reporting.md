# Plugin for GRPO Reporting
- ID `6a37a4ab-33e4-83ee-9e3e-a89e36a0cbde` created 2026-06-21 08:45 UTC | model gpt-5-5-pro

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

## Proposed TRL plugin: `trl-min-report-rl`

`trl-min-report-rl` is a small TRL observer/callback package for `GRPOTrainer`. It does **not** change the loss, sampler, optimizer, reward function, or training loop semantics. Its only job is to make the paper’s Min-Report-RL block automatic: emit the seven-item stack manifest as JSON before the first rollout, then log per-step `ZVF` and `GU` from the grouped rewards that GRPO already needs to compute advantages. The attached paper’s Section 3 defines the seven mandatory fields, and Section 5 explicitly proposes a TRL flag like `report_min_report_rl=True` plus a shared JSON emitter for TRL, verl, and OpenRLHF. fileciteturn0file0

Current TRL already has most of the necessary surfaces: `GRPOTrainer` accepts `callbacks` and `rollout_func`, reward functions are summed across reward functions, and `GRPOConfig` exposes key stack levers such as `num_generations`, vLLM settings, `beta`, `epsilon`, `importance_sampling_level`, `scale_rewards`, and `loss_type`. citeturn256404view0turn256404view1turn256404view2 TRL also already logs a closely related reward-diversity metric, `frac_reward_zero_std`, but the plugin should still emit canonical `min_report_rl/zvf` and `min_report_rl/gu` because the standard needs a pinned definition, JSONL trace, tolerance, grouping key, reward stage, and manifest schema. citeturn256404view3

---

## 1. User-facing API

### Native upstream form

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="runs/qwen-grpo",
    learning_rate=1e-6,
    num_generations=8,
    use_vllm=True,
    vllm_mode="server",

    # New plugin flags
    report_min_report_rl=True,
    min_report_rl_strict="warn",
    min_report_rl_output_dir="runs/qwen-grpo/min_report_rl",
    min_report_rl_zvf_atol=1e-12,
    min_report_rl_emit_to=["file", "trainer_log", "tracker"],

    # User-supplied fields TRL cannot infer reliably
    min_report_rl_reward_env_id="gsm8k/train[:500]",
    min_report_rl_heldout_id="gsm8k/test[:500]",
    min_report_rl_eval_harness="lm-eval-harness:gsm8k@<commit>",
    min_report_rl_decontam_report="artifacts/decontam_gsm8k.json",
    min_report_rl_parser_probe_report="artifacts/reward_parser_probe.json",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-7B-Instruct",
    args=training_args,
    reward_funcs=[reward_fn],
    train_dataset=train_dataset,
    eval_dataset=heldout_dataset,
)

trainer.train()
```

### External package form before upstream merge

```python
from trl import GRPOTrainer, GRPOConfig
from trl_min_report_rl import MinReportRLConfig, attach_min_report_rl

args = GRPOConfig(...)
trainer = GRPOTrainer(..., args=args)

attach_min_report_rl(
    trainer,
    MinReportRLConfig(
        enabled=True,
        strict="warn",
        output_dir=f"{args.output_dir}/min_report_rl",
        reward_env_id="gsm8k/train[:500]",
        heldout_id="gsm8k/test[:500]",
        eval_harness="lm-eval-harness:gsm8k@<commit>",
        decontam_report="artifacts/decontam.json",
        parser_probe_report="artifacts/parser_probe.json",
    ),
)

trainer.train()
```

---

## 2. Hooks

The plugin should be implemented as a **hybrid**: ordinary `TrainerCallback` hooks for lifecycle events, plus one small GRPO-specific reward hook because normal callbacks do not see the grouped reward tensor at the right time.

| Hook | Existing or new | When it fires | What it does |
|---|---:|---|---|
| `GRPOTrainer.__init__` config hook | new TRL patch | After `GRPOConfig` is resolved | If `report_min_report_rl=True`, attach `MinReportRLCallback` and `GRPORewardTelemetryObserver`. |
| `on_init_end` | existing callback | End of trainer init | Snapshot static trainer/model/tokenizer/PEFT/config fields, but do not write yet. |
| `on_train_begin` | existing callback | Before first training step | Emit `min_report_rl.manifest.v1.json`; log manifest URI and schema version. |
| `on_grpo_rollout_begin(payload)` | new GRPO observer hook | Before completions are generated | Allocate `rollout_id`; record sampler/backend/decoding params and actual group-size plan. |
| `on_grpo_rewards_computed(payload)` | **new required GRPO observer hook** | Immediately after raw reward aggregation, before reward scaling / advantage normalization | Compute ZVF/GU from per-prompt grouped rewards; write JSONL step record; push scalar logs. |
| `on_grpo_advantages_computed(payload)` | optional new hook | After advantage normalization | Sanity-check that zero-variance groups produce zero usable group-relative signal. |
| `on_grpo_group_size_update(payload)` | optional new hook | Whenever adaptive `G` changes | Append the schedule event and reason. |
| `on_log` | existing callback | TRL logging event | Merge `min_report_rl/zvf`, `min_report_rl/gu`, and counts into tracker logs. |
| `on_save` | existing callback | Checkpoint save | Copy manifest and telemetry tail into checkpoint artifact directory. |
| `on_train_end` | existing callback | End of training | Emit `min_report_rl.summary.v1.json`, including telemetry checksum and collapse triage. |

The critical hook is `on_grpo_rewards_computed`. It must run **after** all reward functions have been weighted and aggregated into the scalar reward used by GRPO, but **before** `scale_rewards`, advantage normalization, clipping, or loss computation. TRL’s current docs state that rewards can come from one or more reward functions and are summed, so the plugin should compute ZVF on the same aggregated scalar reward that will drive the group-relative update. citeturn256404view0

### Required reward hook payload

```python
@dataclass(frozen=True)
class GRPORewardTelemetryPayload:
    run_id: str
    phase: Literal["train", "eval"]
    global_step: int
    rollout_id: int
    optimizer_step: int | None
    process_rank: int
    world_size: int

    # Grouping
    prompt_group_ids: list[str]          # length = num_completions
    prompt_hashes: list[str]             # same length; no raw prompts by default
    completion_indices: list[int]        # 0..G_i-1 within prompt
    group_sizes: dict[str, int]          # prompt_group_id -> actual G_i

    # Rewards
    raw_aggregated_rewards: list[float | None]
    reward_components: dict[str, list[float | None]]
    reward_weights: dict[str, float]
    reward_stage: Literal["raw_aggregated_before_scaling"]

    # Optional diagnostics
    completion_lengths: list[int] | None
    truncated: list[bool] | None
    sampler_logprob_dtype: str | None
```

---

## 3. Config flags

Use one top-level enable flag and a nested config object. In CLI form, every nested field becomes `--min_report_rl_*`.

| Flag | Default | Meaning |
|---|---:|---|
| `report_min_report_rl` | `True` for `GRPOTrainer` | Emit the manifest and per-step telemetry. |
| `min_report_rl_output_dir` | `"{output_dir}/min_report_rl"` | Directory for JSON manifest, JSONL telemetry, and summary. |
| `min_report_rl_schema_version` | `"min-report-rl.trl.v1"` | Manifest schema version. |
| `min_report_rl_strict` | `"warn"` | `"off"`, `"warn"`, or `"error"` for missing/unknown mandatory fields. |
| `min_report_rl_emit_to` | `["file", "trainer_log"]` | Destinations: `file`, `stdout`, `trainer_log`, `tracker`, `hub_artifact`. |
| `min_report_rl_metric_prefix` | `"min_report_rl"` | Prefix for scalar logs. |
| `min_report_rl_zvf_atol` | `1e-12` | Absolute reward equality tolerance. |
| `min_report_rl_zvf_rtol` | `0.0` | Relative reward equality tolerance. |
| `min_report_rl_hash_prompts` | `True` | Hash prompts instead of logging raw prompt text. |
| `min_report_rl_prompt_hash_salt_env` | `"MIN_REPORT_RL_SALT"` | Optional salt env var for prompt hashes. |
| `min_report_rl_log_reward_components` | `True` | Also log component-level ZVF/GU, not only aggregated reward ZVF/GU. |
| `min_report_rl_collapse_window_steps` | `5` | Window for first-five-step collapse triage. |
| `min_report_rl_collapse_zvf_threshold` | `0.80` | Collapse warning threshold. |
| `min_report_rl_collapse_reward_threshold` | `0.05` | Collapse warning threshold for low reward. |
| `min_report_rl_reward_env_id` | `null` | Training/reward environment identifier. |
| `min_report_rl_heldout_id` | `null` | Held-out dataset/split identifier. |
| `min_report_rl_heldout_fingerprint` | inferred if possible | Dataset fingerprint or prompt-hash manifest. |
| `min_report_rl_eval_harness` | `null` | Harness name/version/commit for held-out scoring. |
| `min_report_rl_ci_method` | `null` | Confidence-interval method used for held-out evaluation. |
| `min_report_rl_decontam_report` | `null` | Path/URI to train/test overlap report. |
| `min_report_rl_parser_probe_report` | `null` | Path/URI to adversarial parser probe report. |
| `min_report_rl_fail_on_closed_stack_unknowns` | `False` | If true, fail when a backend cannot expose mandatory fields. |

Fields that TRL can infer should be filled automatically: `loss_type`, `epsilon`, `epsilon_high`, `importance_sampling_level`, `scale_rewards`, `beta`, reference-model state, `num_generations`, vLLM settings, decoding parameters, tokenizer/chat-template hashes, precision flags, PEFT/LoRA config if present, optimizer class, package versions, and seed. Fields that TRL cannot prove, especially held-out disjointness and decontamination probes, must be supplied or marked `unknown`.

---

## 4. ZVF/GU computation

For each training step, the plugin groups rewards by prompt. For prompt group `x` with completions `r_{x,1..G}`, the group is zero-variance if:

```text
max(finite_rewards_x) - min(finite_rewards_x)
    <= min_report_rl_zvf_atol
       + min_report_rl_zvf_rtol * max(1, max(abs(finite_rewards_x)))
```

Then:

```text
ZVF = zero_variance_group_count / eligible_group_count
GU  = 1.0 - ZVF
```

Important details:

* The denominator is **prompt groups**, not completions.
* The reward stage is `raw_aggregated_before_scaling`, so ZVF reflects the scalar signal entering GRPO before normalization.
* Groups with fewer than two finite rewards are excluded and counted in `ineligible_group_count`.
* For variable/adaptive `G`, each prompt uses its actual `G_i`; the JSONL record logs min/mean/max group size.
* For multiple reward functions, the canonical ZVF uses the final weighted scalar reward; optional component-level metrics are logged as `min_report_rl/zvf_by_reward/<reward_name>`.
* In distributed training, all ranks gather reward telemetry first; rank zero computes the global ZVF/GU, and all ranks log the same metric keys to avoid tracker divergence.

Scalar names emitted to TRL/tracker logs:

```text
min_report_rl/zvf
min_report_rl/gu
min_report_rl/zero_variance_groups
min_report_rl/mixed_groups
min_report_rl/eligible_groups
min_report_rl/ineligible_groups
min_report_rl/group_size_mean
min_report_rl/group_size_min
min_report_rl/group_size_max
min_report_rl/collapse_candidate
```

---

## 5. Run-start manifest schema

The start-of-run artifact is a single JSON file:

```text
{output_dir}/min_report_rl/min_report_rl.manifest.v1.json
```

Schema outline:

```json
{
  "$schema": "https://zvf.dev/schemas/min-report-rl/trl-manifest-v1.json",
  "schema_version": "min-report-rl.trl.v1",
  "emitted_at_utc": "2026-06-21T00:00:00Z",
  "run": {
    "run_id": "uuid",
    "trainer": "trl.GRPOTrainer",
    "algorithm_label": "GRPO",
    "trl_version": "string",
    "transformers_version": "string",
    "accelerate_version": "string",
    "torch_version": "string",
    "python_version": "string",
    "git": {
      "repo": "string|null",
      "commit": "string|null",
      "dirty": "boolean|null"
    },
    "seed": 42,
    "distributed": {
      "world_size": 8,
      "mixed_precision": "bf16|fp16|fp32|unknown"
    }
  },
  "minimum_reportable_stack": {
    "1_loss_form": {
      "loss_type": "grpo|dapo|dr_grpo|bnpo|cispo|sapo|luspo|vespo|custom",
      "importance_ratio": {
        "enabled": true,
        "level": "token|sequence|none|unknown",
        "old_policy_source": "rollout_logprobs|policy_snapshot|unknown"
      },
      "clip": {
        "enabled": true,
        "epsilon_low": 0.2,
        "epsilon_high": 0.2,
        "asymmetric": false
      },
      "token_mask": {
        "scope": "completion_only|whole_sequence|custom|unknown",
        "mask_truncated_completions": false,
        "top_entropy_quantile": 1.0
      },
      "advantage_normalization": {
        "scale_rewards": "group|batch|none|custom",
        "multi_objective_aggregation": "sum_then_normalize|normalize_then_sum|custom",
        "reward_weights": {
          "reward_fn_name": 1.0
        }
      }
    },
    "2_reference_policy_kl": {
      "reference_policy": {
        "retained": true,
        "source": "separate_ref_model|adapter_disabled|none|closed|unknown",
        "sync_ref_model": false,
        "ref_model_sync_steps": null
      },
      "kl": {
        "coefficient_beta": 0.0,
        "placement": "loss|reward|none|closed|unknown",
        "schedule": "constant|linear|custom|none|unknown",
        "estimator": "per_token_forward|per_sequence|reverse|unknown",
        "logged_metric": "kl"
      }
    },
    "3_sampler_backend_precision": {
      "rollout_engine": {
        "name": "vllm|transformers.generate|custom_rollout_func|managed_api|unknown",
        "mode": "server|colocate|local|closed|unknown",
        "version": "string|null",
        "model_impl": "vllm|transformers|unknown"
      },
      "decoding": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": null,
        "max_completion_length": 256,
        "repetition_penalty": 1.0,
        "generation_kwargs": {}
      },
      "precision": {
        "trainer_dtype": "bf16|fp16|fp32|mixed|unknown",
        "sampler_dtype": "bf16|fp16|fp32|closed|unknown",
        "logprob_dtype": "bf16|fp16|fp32|closed|unknown",
        "lm_head_cast_to_fp32": false
      },
      "tokenizer_and_template": {
        "tokenizer_name_or_path": "string",
        "tokenizer_revision": "string|null",
        "tokenizer_hash": "sha256|null",
        "chat_template_hash": "sha256|null",
        "sampler_trainer_template_shared": true
      }
    },
    "4_zvf_gu_trajectory": {
      "enabled": true,
      "definition": "ZVF = fraction of prompt groups whose raw aggregated rewards have zero variance; GU = 1 - ZVF",
      "reward_stage": "raw_aggregated_before_scaling",
      "tolerance": {
        "atol": 1e-12,
        "rtol": 0.0
      },
      "logged_every": "rollout_step",
      "telemetry_jsonl": "min_report_rl.steps.v1.jsonl",
      "metric_prefix": "min_report_rl"
    },
    "5_group_size_schedule": {
      "initial_num_generations": 8,
      "schedule_type": "fixed|adaptive|dynamic_sampling|custom|unknown",
      "per_step_logged": true,
      "adaptive_rule": null,
      "compute_accounting": {
        "log_completion_count": true,
        "log_token_count": true
      }
    },
    "6_heldout_split": {
      "reward_environment": {
        "dataset_id": "gsm8k/train[:500]",
        "split": "train",
        "fingerprint": "string|null",
        "prompt_hash_manifest": "path|null"
      },
      "heldout": {
        "dataset_id": "gsm8k/test[:500]",
        "split": "test",
        "fingerprint": "string|null",
        "n": 500,
        "disjoint_from_reward_environment": "true|false|unknown",
        "eval_harness": "lm-eval-harness:gsm8k@<commit>",
        "confidence_interval": {
          "method": "bootstrap|wilson|normal|unknown",
          "level": 0.95
        }
      }
    },
    "7_decontamination_probes": {
      "train_test_overlap": {
        "status": "passed|failed|unknown",
        "methods": ["ngram", "embedding"],
        "report_uri": "artifacts/decontam_gsm8k.json",
        "max_ngram_overlap": 0.0,
        "max_embedding_similarity": null
      },
      "reward_parser": {
        "name": "string",
        "version": "string|null",
        "source_digest": "sha256|null"
      },
      "parser_adversarial_probe": {
        "status": "passed|failed|unknown",
        "report_uri": "artifacts/reward_parser_probe.json",
        "format_only_reward_rate": 0.0,
        "length_only_reward_rate": 0.0
      }
    }
  },
  "implementation_context": {
    "model": {
      "name_or_path": "string",
      "revision": "string|null",
      "config_hash": "sha256|null"
    },
    "peft_lora": {
      "enabled": true,
      "r": 16,
      "alpha": 32,
      "target_modules": ["q_proj", "v_proj"],
      "dropout": 0.0
    },
    "optimizer": {
      "name": "adamw_torch_fused",
      "learning_rate": 1e-6,
      "weight_decay": 0.0,
      "scheduler": "linear"
    }
  },
  "unknown_fields": [],
  "warnings": [],
  "artifacts": {
    "manifest_path": "min_report_rl.manifest.v1.json",
    "telemetry_path": "min_report_rl.steps.v1.jsonl",
    "summary_path": "min_report_rl.summary.v1.json"
  }
}
```

The `minimum_reportable_stack` object is deliberately seven-keyed. Additional useful details, such as model, LoRA, optimizer, hardware, and package versions, live under `implementation_context` so the canonical seven-item block stays stable.

---

## 6. Per-step ZVF/GU JSONL schema

The plugin writes one JSON object per rollout/scoring event to:

```text
{output_dir}/min_report_rl/min_report_rl.steps.v1.jsonl
```

Each line:

```json
{
  "schema_version": "min-report-rl.trl-step.v1",
  "run_id": "uuid",
  "emitted_at_utc": "2026-06-21T00:00:05Z",
  "phase": "train",
  "global_step": 12,
  "rollout_id": 12,
  "optimizer_step": 12,
  "reused_rollout": false,
  "world_size": 8,
  "grouping": {
    "num_prompt_groups": 64,
    "eligible_groups": 64,
    "ineligible_groups": 0,
    "group_size_min": 8,
    "group_size_mean": 8.0,
    "group_size_max": 8,
    "variable_group_size": false
  },
  "zvf": {
    "value": 0.3125,
    "zero_variance_groups": 20,
    "tolerance": {
      "atol": 1e-12,
      "rtol": 0.0
    },
    "reward_stage": "raw_aggregated_before_scaling"
  },
  "gu": {
    "value": 0.6875,
    "mixed_groups": 44
  },
  "reward": {
    "mean": 0.421875,
    "std": 0.493857,
    "min": 0.0,
    "max": 1.0,
    "none_count": 0,
    "nan_count": 0,
    "component_means": {
      "accuracy_reward": 0.421875
    }
  },
  "trl_metrics": {
    "frac_reward_zero_std": 0.3125,
    "reward": 0.421875,
    "reward_std": 0.493857,
    "kl": null
  },
  "group_size_schedule": {
    "schedule_type": "fixed",
    "actual_num_generations": 8,
    "adaptive_rule": null,
    "changed_this_step": false
  },
  "collapse_triage": {
    "window": 5,
    "zvf_threshold": 0.8,
    "reward_threshold": 0.05,
    "first_five_rule_evaluable": true,
    "collapse_candidate": false
  },
  "privacy": {
    "raw_prompts_logged": false,
    "prompt_hashes_logged": true,
    "prompt_hash_sample": [
      "sha256:..."
    ]
  }
}
```

If `num_iterations > 1` reuses the same rollout for multiple optimizer steps, the plugin should either log the same `rollout_id` with `reused_rollout=true`, or log once and include:

```json
"applies_to_optimizer_steps": [12, 13, 14]
```

That avoids pretending a reused reward batch is fresh signal.

---

## 7. Failure and warning behavior

`min_report_rl_strict="warn"` should be the default for usability. The plugin should still emit the JSON block with explicit `unknown` values and a `warnings` array. `min_report_rl_strict="error"` should fail before training if any of the seven mandatory items are absent, except for values that are genuinely inapplicable and marked as such.

Examples:

```json
{
  "field": "6_heldout_split.heldout.disjoint_from_reward_environment",
  "severity": "warning",
  "message": "Held-out disjointness was not proven. Provide min_report_rl_decontam_report or set strict='off'."
}
```

```json
{
  "field": "3_sampler_backend_precision.precision.sampler_dtype",
  "severity": "warning",
  "message": "Managed backend did not expose sampler logit precision; marked as closed/unknown."
}
```

---

## 8. Minimal upstream patch

The smallest TRL patch is:

```python
# inside GRPOTrainer, after rewards are aggregated and before reward scaling
if self.min_report_rl_observer is not None:
    self.min_report_rl_observer.on_grpo_rewards_computed(
        GRPORewardTelemetryPayload(
            run_id=self.min_report_rl_run_id,
            phase="train",
            global_step=self.state.global_step,
            rollout_id=self._min_report_rl_rollout_id,
            prompt_group_ids=prompt_group_ids,
            prompt_hashes=prompt_hashes,
            completion_indices=completion_indices,
            group_sizes=group_sizes,
            raw_aggregated_rewards=raw_rewards,
            reward_components=reward_components,
            reward_weights=reward_weights,
            reward_stage="raw_aggregated_before_scaling",
            world_size=self.accelerator.num_processes,
            process_rank=self.accelerator.process_index,
        )
    )
```

Everything else can live in the plugin package. The paper’s adoption claim is therefore realistic: the mandatory telemetry is “a few lines over the per-prompt group rewards” plus a manifest writer, not a new training method.

---
