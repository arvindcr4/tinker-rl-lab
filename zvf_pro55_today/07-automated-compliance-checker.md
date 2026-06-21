# Automated Compliance Checker
- ID `6a379b30-3280-83ee-a374-ea8cebc756d7` created 2026-06-21 08:05 UTC | model gpt-5-5-pro

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

## StackCheck-RL: an automated MIN-REPORT-RL compliance checker

I would build **StackCheck-RL** as an evidence-first linter for GRPO-family papers and repositories. Its job is not to decide whether a method is good; it decides whether the paper or repo reports the full experimental treatment named by the MIN-REPORT-RL checklist. The attached draft frames the problem exactly this way: “GRPO” is not a complete treatment label, because loss form, KL/reference handling, sampler/backend, group-size behavior, telemetry, held-out evaluation, and decontamination can each flip a comparison. fileciteturn0file0 The paper’s Section 3 defines the seven reportable fields, and Table 1 summarizes them as mandatory “flip levers,” not optional hyperparameters. fileciteturn1file3 fileciteturn1file6

### What StackCheck-RL scans

**Inputs**

`stackcheck-rl paper.pdf repo/ --mode paper+repo`

It accepts a manuscript, LaTeX source, Markdown appendix, model card, README, experiment config, training scripts, logs, TensorBoard or W&B exports, reward parser tests, and evaluation artifacts.

**Outputs**

It emits:

1. A **headline compliance score** from 0–100.
2. A **seven-item scorecard** with evidence snippets and file paths.
3. A **strict reported score**: what the paper or README explicitly tells the reader.
4. A **reconstructed score**: what an auditor can infer from code/config/logs.
5. A **contradiction report**: places where paper and repo disagree.
6. A **machine-readable `min_report_rl.audit.json`**.
7. Optional CI outputs: Markdown badge, GitHub check, SARIF annotations.

The core design principle is: **implemented but not reported is not full compliance**. Code can rescue recoverability, but MIN-REPORT-RL is a reporting standard.

---

## Scoring model

Each checklist item receives **0–10 points**, for a raw total of **70**. The headline score is:

\[
\text{MIN-REPORT-RL Score} = \mathrm{round}\left(100 \cdot \frac{\sum_{i=1}^7 s_i}{70}\right)
\]

Default bands:

| Score | Label | Meaning |
|---:|---|---|
| 90–100 | **Compliant** | Complete, auditable, low ambiguity. |
| 75–89 | **Mostly compliant** | Minor gaps; a reader can usually reconstruct the treatment. |
| 55–74 | **Partial** | Several stack levers remain ambiguous. |
| 30–54 | **Weak** | Most claims are not reproducibility-auditable. |
| 0–29 | **Non-compliant** | “GRPO” label is doing most of the work. |

Hard flags override the label:

| Flag | Effect |
|---|---|
| Paper/repo contradiction on a checklist field | Item capped at 4/10; overall marked **needs human audit**. |
| “Default settings,” “standard GRPO,” or “as usual” with no values | Relevant subfield capped at 1/10 or 2/10. |
| Closed backend hides loss/KL internals | Items 1–2 cannot receive full credit; result must be scoped as “platform implementation,” not generic GRPO, matching the paper’s own objection/response logic. fileciteturn1file9 |
| No held-out split distinct from reward environment | Item 6 capped at 2/10. |
| No usable-signal trajectory, ZVF/GU, reward variance, or gradient-variance substitute | Item 4 capped at 1/10. |
| Broken artifact links | Affected item capped at 6/10. |
| Only code inference, no paper/README/checklist report | Affected item capped at 7/10. |

Evidence quality is reported separately:

| Evidence grade | Meaning |
|---|---|
| A | Explicitly reported and verified in repo/logs. |
| B | Explicitly reported, not machine-verified. |
| C | Recoverable from repo only. |
| D | Vague mention only. |
| F | Missing or contradicted. |

---

# Per-item rubric and heuristics

## 1. Loss form — 10 points

The paper says authors should report whether a PPO-style importance ratio is used, whether clipping exists and with what bounds, whether the loss mask is completion-only or whole-sequence, and how advantages are normalized. fileciteturn1file3

**Rubric**

| Subfield | Points |
|---|---:|
| PPO / importance ratio status: yes/no, formula, old-policy source | 2 |
| Clip rule: none, symmetric ε, asymmetric lower/upper, DAPO-style clip-higher | 2 |
| Token mask: completion-only, whole-sequence, prompt included/excluded | 2 |
| Advantage normalization: per-group, per-batch, running estimate, formula | 2 |
| Paper, config, and code agree | 2 |

**Paper heuristics**

Positive signals:

- “PPO ratio,” “importance ratio,” `πθ / πold`, “old logprobs”
- “clip ε=0.2,” “clip range,” “clip higher,” “asymmetric clip”
- “completion-only loss,” “response mask,” “prompt tokens masked”
- “per-group advantage normalization,” “subtract group mean,” “divide by group std”
- “length normalization,” “Dr.GRPO normalization”

Negative signals:

- “standard GRPO loss”
- “we use GRPO as implemented in TRL”
- “loss follows prior work”
- “details are in code” with no field values

**Repo heuristics**

StackCheck-RL searches Python AST, configs, and trainer arguments for:

```python
ratio = torch.exp(new_logprobs - old_logprobs)
torch.clamp(ratio, 1 - eps, 1 + eps)
completion_mask
response_mask
attention_mask[:, prompt_len:]
advantages = (rewards - rewards.mean(dim=1)) / rewards.std(dim=1)
num_generations
loss_type
clip_range
epsilon
epsilon_high
```

It also detects likely whole-sequence leakage by checking whether prompt-token positions have nonzero loss weight.

**Caps**

- “Standard GRPO loss” alone: max 2/10.
- Loss recoverable only from code: max 7/10.
- Paper says completion-only but code applies loss to prompt tokens: cap 4/10 and raise contradiction.

---

## 2. Reference policy and KL handling — 10 points

The checklist requires the frozen reference status, KL location, coefficient/schedule, and estimator granularity. The draft emphasizes that KL in the loss, KL in the reward, and no frozen reference are different objectives. fileciteturn1file3

**Rubric**

| Subfield | Points |
|---|---:|
| Frozen reference retained? checkpoint/revision given? | 2 |
| KL location: loss, reward, absent, dual anchor | 2 |
| KL coefficient and schedule | 2 |
| KL estimator: forward/reverse, per-token/per-sequence | 2 |
| Paper, config, and code agree | 2 |

**Paper heuristics**

Positive signals:

- “frozen reference model”
- “reference = SFT checkpoint”
- “KL in reward,” “non-score reward,” “KL penalty in loss”
- `β=0.04`, `kl_coef`, “linear schedule,” “adaptive KL”
- “reverse KL,” “forward KL,” “per-token KL,” “sequence KL”

Negative signals:

- “KL regularization as usual”
- “we follow PPO/GRPO defaults”
- no mention of reference model
- no β value

**Repo heuristics**

Searches for:

```python
ref_model
reference_model
kl_coef
kl_beta
kl_controller
non_score_reward
compute_kl
old_logprobs
ref_logprobs
per_token_kl
sequence_kl
```

It checks whether `ref_model` is actually loaded, frozen, and used in the loss/reward path.

**Caps**

- KL coefficient reported but no frozen-reference status: max 6/10.
- Closed backend with hidden KL/loss: max 5/10, or 6/10 if the paper explicitly scopes the result as a platform-specific implementation.
- Paper says “KL absent,” code computes KL penalty: cap 4/10.

---

## 3. Sampler, backend, and precision — 10 points

The paper treats rollout engine, decoding parameters, sampler precision, trainer precision, tokenizer, and chat template as mandatory because the sampler defines the rollout distribution consumed by group-relative updates. fileciteturn1file3

**Rubric**

| Subfield | Points |
|---|---:|
| Rollout engine and version: vLLM, SGLang, HF `generate`, managed API | 2 |
| Decoding parameters: temperature, top-p/top-k, max tokens, stop rules | 2 |
| Sampler precision and trainer precision | 2 |
| Tokenizer and chat template identity, revision, same/different across sampler/trainer | 2 |
| Backend/config evidence agrees with paper | 2 |

**Paper heuristics**

Positive signals:

- “vLLM 0.x,” “SGLang,” “HF generate,” “managed API”
- “temperature=0.8,” “top_p=1.0,” “max_tokens=512”
- `bf16`, `fp16`, `fp32`, “sampler logits”
- “same tokenizer and chat template in sampler and trainer”
- “tokenizer revision/hash”

Negative signals:

- “generated with default settings”
- “we sample responses”
- “OpenAI-compatible endpoint” with no model/version/template details
- tokenizer omitted

**Repo heuristics**

Searches for imports and configs:

```python
import vllm
from vllm import LLM, SamplingParams
import sglang
model.generate(...)
SamplingParams(temperature=..., top_p=..., max_tokens=...)
torch_dtype=torch.bfloat16
bf16=True
fp16=True
tokenizer.apply_chat_template(...)
chat_template
```

It cross-checks tokenizer IDs and model revisions between rollout and trainer code.

**Caps**

- Decoding defaults only: max 5/10.
- No tokenizer/chat-template evidence: max 6/10.
- Managed backend with unknown sampler internals: max 7/10 even if decoding parameters are reported.

---

## 4. Per-step ZVF and GU trajectory — 10 points

The paper defines ZVF as the fraction of prompts whose group completions all receive identical reward, and GU as `1 − ZVF`; it also allows substitutes such as per-step reward variance or gradient-norm variance when dense rewards make ZVF less meaningful. fileciteturn1file12 fileciteturn1file9

**Rubric**

| Subfield | Points |
|---|---:|
| Metric defined: ZVF/GU or justified substitute | 2 |
| Per-step trajectory logged, not just final summary | 3 |
| Correct computation from per-prompt group rewards | 2 |
| Collapse/saturation summary, e.g. first-five-step rule | 1 |
| Artifact exists and matches paper | 2 |

**Paper heuristics**

Positive signals:

- “Zero-Variance Fraction”
- “ZVF”
- “Gradient Utilization”
- “GU = 1 − ZVF”
- “per-step reward variance”
- “gradient-norm variance”
- “mean ZVF@25”
- “collapse rule”
- “trajectory CSV”

Negative signals:

- only final reward
- only loss curve
- only training accuracy
- “training was stable” with no usable-signal metric

**Repo/log heuristics**

Searches logs for columns:

```text
step,zvf,gu
zero_variance_fraction
gradient_utilization
reward_variance
group_reward_std
mean_zvf_at_25
```

Checks computation shape:

```python
# rewards shape: [num_prompts, group_size]
zvf = ((rewards.max(dim=1).values - rewards.min(dim=1).values) == 0).float().mean()
gu = 1 - zvf
```

or equivalent.

**Caps**

- Final ZVF only: max 4/10.
- No per-step usable-signal telemetry: max 1/10.
- Dense/process reward substitute can receive full credit only if the paper explains why ZVF degenerates and provides a per-step alternative.

---

## 5. Group-size schedule, fixed or adaptive — 10 points

The checklist requires `G` at every step and the rule for changing it if adaptive, because `G` directly changes the mixed-group probability and thus available gradient. fileciteturn1file10

**Rubric**

| Subfield | Points |
|---|---:|
| Fixed or adaptive schedule clearly stated | 3 |
| If adaptive: rule, thresholds, bounds, window | 2 |
| Per-step `G(t)` recoverable from logs/config | 2 |
| Compute implications reported for adaptive schedules | 1 |
| Paper, sampler config, and logs agree | 2 |

**Paper heuristics**

Positive signals:

- “fixed G=8 throughout”
- “num_generations=8”
- “adaptive G ∈ {4, 8, 16}”
- “double when rolling ZVF > 0.8”
- “halve when ZVF < 0.3”
- “window=10”

Negative signals:

- “group size 8” but no statement about fixed/adaptive
- “dynamic sampling” with no schedule
- “we sample multiple completions” with no `G`

**Repo heuristics**

Searches for:

```python
num_generations
group_size
n_samples_per_prompt
rollout_n
samples_per_prompt
adaptive_group_size
zvf_window
max_group_size
min_group_size
```

It also reads logs to confirm whether the group count changes over steps.

**Caps**

- Initial `G` only, no fixed/adaptive statement: max 5/10.
- Adaptive code detected but paper says fixed: cap 4/10.
- Dynamic sampling with no rule: max 6/10.

---

## 6. Held-out split distinct from the reward environment — 10 points

The paper requires a held-out slice disjoint from training prompts, scored by a harness, with sample size and confidence interval, reported separately from online training reward. fileciteturn1file10

**Rubric**

| Subfield | Points |
|---|---:|
| Held-out slice is disjoint from training prompts | 2 |
| Dataset/split, sample size, seed, and selection procedure | 2 |
| Evaluation harness/scoring described | 2 |
| CI/statistical test and seed aggregation | 2 |
| Reported separately from online training reward, with checkpoint-selection rule | 2 |

**Paper heuristics**

Positive signals:

- “held-out”
- “test split”
- “validation split”
- “disjoint from training prompts”
- `N=500`
- “Wilson CI,” “bootstrap CI,” “95% CI”
- “paired test,” “p-value”
- “lm-eval-harness,” “LightEval,” “EvalPlus”
- “checkpoint selected before held-out evaluation”

Negative signals:

- “training reward”
- “online reward”
- “last-10 reward” used as capability
- “accuracy” with no split
- selecting best checkpoint by eval without declaring selection rule

**Repo heuristics**

Searches for dataset loading and split names:

```python
load_dataset("gsm8k", split="train")
load_dataset("gsm8k", split="test")
eval_dataset
heldout_dataset
validation_prompts
lm_eval
lighteval
bootstrap
wilson
confidence_interval
```

It computes hashes of training prompts and eval prompts when files are present.

**Caps**

- Only training reward reported: max 2/10.
- Held-out score reported but no N or CI: max 5/10.
- Held-out appears to overlap training prompts: max 4/10 and high-severity flag.
- No checkpoint-selection rule: max 8/10.

---

## 7. Decontamination probe results — 10 points

The paper requires both train/test contamination checking and parser behavior on adversarial format-only inputs, because reward hacking and train-prompt overlap can inflate apparent gains. fileciteturn1file10

**Rubric**

| Subfield | Points |
|---|---:|
| Train/test overlap metric and threshold: n-gram, exact hash, embedding similarity | 3 |
| Reported overlap result and action taken on overlaps | 2 |
| Parser adversarial tests: format-only, no-answer, length/shortcut cases | 3 |
| Reproducible artifact or command | 1 |
| Parser source/tests agree with paper | 1 |

**Paper heuristics**

Positive signals:

- “decontamination”
- “8-gram overlap”
- “MinHash”
- “embedding overlap”
- “cosine similarity threshold”
- “train/test leakage”
- “format-only adversarial input”
- “parser rejects”
- “no-answer adversarial”
- “reward parser unit tests”

Negative signals:

- “dataset is standard, so no contamination”
- “we decontaminated” with no metric
- no parser tests
- no overlap threshold

**Repo heuristics**

Searches for:

```python
ngram_overlap
deduplicate
minhash
embedding_similarity
cosine_similarity
contamination
decontam
parser_test
format_only
no_answer
reject_rate
reward_parser
pytest
```

It can optionally run parser tests against generated adversarial strings like:

```text
"####"
"The answer is boxed."
"<answer></answer>"
"Final answer: "
"Therefore, the answer is"
```

**Caps**

- Generic “we decontaminated” with no metric/result: max 2/10.
- Overlap probe but no parser adversarial probe: max 5/10.
- Parser tests but no train/test overlap probe: max 5/10.
- Parser rewards format-only input: high-severity flag even if overlap probe passes.

---

# Checker architecture

```text
                 ┌────────────────────┐
                 │ paper.pdf / .tex   │
                 │ README / model card│
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Document extractor │
                 │ PDF/LaTeX/Markdown │
                 └─────────┬──────────┘
                           │
repo/ ───────────┐         │
configs/logs ────┼─────────▼──────────┐
wandb export ────┘   Evidence indexer │
                     regex + AST +    │
                     semantic chunks  │
                     └──────┬─────────┘
                            │
                    ┌───────▼────────┐
                    │ Rule engine     │
                    │ 7 item scorers  │
                    │ caps + flags    │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │ Outputs         │
                    │ MD / JSON / CI  │
                    └────────────────┘
```

The checker has three internal passes:

**Pass 1: GRPO-family detector**

It first decides whether the repo or paper is in scope. It looks for terms such as `GRPO`, `DAPO`, `GSPO`, `Dr.GRPO`, `MAD-GRPO`, `AERO`, `CPPO`, `NGRPO`, `Scaf-GRPO`, `group relative`, `num_generations`, `group_size`, `GRPOTrainer`, `verl`, `OpenRLHF`, and reward-group advantage normalization.

**Pass 2: Evidence extraction**

It builds an evidence table:

```json
{
  "claim": "completion-only token mask",
  "source": "paper",
  "location": "appendix_a.tex:118",
  "text": "Token mask: completion-only",
  "confidence": 0.96
}
```

For repos, it prefers exact config values over inferred defaults.

**Pass 3: Consistency and scoring**

It scores each subfield, applies caps, and flags contradictions. For example:

```json
{
  "field": "loss_form.token_mask",
  "paper_value": "completion-only",
  "repo_value": "whole-sequence",
  "status": "contradiction",
  "severity": "high",
  "cap_applied": 4
}
```

This is important because MIN-REPORT-RL is meant to make the stack auditable, not merely present somewhere in a codebase. The paper itself proposes a JSON-like block and argues that trainer defaults should emit the seven fields automatically. fileciteturn1file2

---

# Suggested manifest schema

A repo can get near-automatic full credit by shipping `min_report_rl.json`:

```json
{
  "method_label": "GRPO",
  "base_checkpoint": {
    "id": "Qwen/Qwen3-8B",
    "revision": "abc123"
  },
  "tokenizer_chat_template": {
    "tokenizer_id": "Qwen/Qwen3-8B",
    "tokenizer_revision": "abc123",
    "chat_template_hash": "sha256:...",
    "same_sampler_and_trainer": true
  },
  "loss_form": {
    "ppo_ratio_used": true,
    "clip": {
      "type": "symmetric",
      "epsilon": 0.2
    },
    "token_mask": "completion_only",
    "advantage_normalization": {
      "scope": "per_group",
      "formula": "(r - mean_group(r)) / (std_group(r) + 1e-8)"
    }
  },
  "reference_kl": {
    "frozen_reference": true,
    "reference_checkpoint": "Qwen/Qwen3-8B-SFT@def456",
    "kl_location": "loss",
    "kl_coefficient": 0.04,
    "kl_schedule": "constant",
    "kl_estimator": "reverse_per_token"
  },
  "sampler_backend_precision": {
    "rollout_engine": "vLLM",
    "engine_version": "0.8.5",
    "decoding": {
      "temperature": 0.8,
      "top_p": 1.0,
      "max_tokens": 512
    },
    "precision": {
      "sampler_logits": "bf16",
      "trainer_logits": "bf16"
    }
  },
  "usable_signal_trajectory": {
    "metric": "zvf_gu",
    "artifact": "artifacts/zvf_gu.csv",
    "mean_zvf_at_25": 0.43,
    "mean_gu_at_25": 0.57,
    "collapse_rule_trips": "0/5"
  },
  "group_size_schedule": {
    "type": "fixed",
    "g": 8,
    "artifact": "artifacts/group_size_by_step.csv"
  },
  "heldout_split": {
    "dataset": "GSM8K",
    "split": "test",
    "n": 500,
    "disjoint_from_training": true,
    "harness": "lm-eval-harness",
    "base_accuracy": 0.82,
    "post_accuracy": 0.833,
    "ci": "Wilson 95%",
    "reported_separately_from_training_reward": true
  },
  "decontamination": {
    "overlap_probe": {
      "metric": "8gram_overlap",
      "value": 0.0,
      "threshold": 0.0
    },
    "parser_adversarial": {
      "format_only_reject_rate": 1.0,
      "artifact": "tests/test_reward_parser_adversarial.py"
    }
  }
}
```

---

# Example output A: highly compliant repo

Synthetic example:

```text
$ stackcheck-rl paper.pdf repo/ --mode paper+repo

MIN-REPORT-RL Score: 94/100
Raw: 66/70
Verdict: COMPLIANT
Evidence confidence: 0.91
Critical flags: none
```

| Item | Score | Evidence grade | Finding |
|---|---:|---|---|
| 1. Loss form | 10/10 | A | PPO ratio present; symmetric clip ε=0.2; completion-only mask; per-group normalization; paper and `train_grpo.py` agree. |
| 2. Reference + KL | 10/10 | A | Frozen SFT reference; KL in loss; β=0.04 constant; reverse per-token KL. |
| 3. Sampler/backend/precision | 9/10 | A | vLLM rollouts; temp/top-p/max tokens reported; bf16 sampler/trainer; tokenizer/template shared. Missing exact vLLM commit, version present. |
| 4. ZVF/GU trajectory | 10/10 | A | Per-step CSV released; mean ZVF@25=0.43, GU@25=0.57; no collapse-rule trips. |
| 5. Group-size schedule | 9/10 | A | Fixed G=8 in paper/config/logs. Minor: no compute-normalization note. |
| 6. Held-out split | 9/10 | A | GSM8K held-out N=500, disjoint, harnessed, Wilson CI. Minor: seed list in artifact, not paper. |
| 7. Decontamination | 9/10 | A | 8-gram overlap 0.0%; parser rejects 100% format-only adversarial inputs. Minor: no embedding-overlap probe. |

Recommended patch:

```text
Add vLLM commit hash and seed list to appendix.
Optional: add embedding-overlap probe to complement 8-gram overlap.
```

---

# Example output B: typical under-specified GRPO paper

Synthetic example:

```text
$ stackcheck-rl paper.pdf --mode paper-only

MIN-REPORT-RL Score: 21/100
Raw: 15/70
Verdict: NON-COMPLIANT
Critical flags:
  - No usable-signal trajectory.
  - Training reward is reported as capability.
  - No decontamination or parser adversarial probe.
```

| Item | Score | Finding |
|---|---:|---|
| 1. Loss form | 2/10 | Says “standard GRPO loss.” No ratio, clip, token mask, or advantage-normalization details. |
| 2. Reference + KL | 1/10 | Mentions KL regularization but no reference-policy status, β, location, or estimator. |
| 3. Sampler/backend/precision | 3/10 | Reports temperature and max tokens, but not backend, precision, tokenizer, or chat template. |
| 4. ZVF/GU trajectory | 0/10 | No ZVF, GU, reward variance, or gradient-variance trajectory. |
| 5. Group-size schedule | 4/10 | Reports G=8, but not whether fixed/adaptive or logged per step. |
| 6. Held-out split | 3/10 | Reports “accuracy” but does not state disjointness, N, CI, or whether separate from training reward. |
| 7. Decontamination | 2/10 | Says “we remove duplicates” but gives no overlap metric, threshold, or parser adversarial tests. |

Generated reviewer note:

> The method comparison is not auditable under MIN-REPORT-RL. The paper reports the algorithm label and some visible hyperparameters, but omits the stack fields that the standard identifies as comparison-flipping levers. At minimum, add a MIN-REPORT-RL block with loss form, KL/reference handling, sampler/backend/precision, per-step usable-signal telemetry, group-size schedule, disjoint held-out evaluation, and decontamination/parser probes.

---

# Example output C: closed managed-backend result

Synthetic example:

```json
{
  "min_report_rl_score": 57,
  "raw_score": "40/70",
  "verdict": "PARTIAL_WITH_SCOPE_WARNING",
  "flags": [
    {
      "severity": "high",
      "item": "loss_form",
      "message": "Closed backend does not expose loss implementation. Result should be scoped as ManagedRunner-GRPO, not generic GRPO."
    },
    {
      "severity": "high",
      "item": "reference_kl",
      "message": "KL/reference handling hidden by platform."
    }
  ],
  "items": {
    "loss_form": {
      "score": 4,
      "evidence_grade": "D",
      "summary": "Paper says platform uses GRPO but cannot expose ratio/clip/mask/advantage normalization."
    },
    "reference_kl": {
      "score": 4,
      "evidence_grade": "D",
      "summary": "KL/reference status unavailable; paper honestly scopes the result as platform-specific."
    },
    "sampler_backend_precision": {
      "score": 7,
      "evidence_grade": "B",
      "summary": "Managed API named; decoding parameters reported; tokenizer/template and logit precision unavailable."
    },
    "zvf_gu_trajectory": {
      "score": 8,
      "evidence_grade": "B",
      "summary": "Per-step ZVF/GU logged from returned group rewards; computation script released."
    },
    "group_size_schedule": {
      "score": 8,
      "evidence_grade": "B",
      "summary": "Fixed G=8 reported and confirmed in request logs."
    },
    "heldout_split": {
      "score": 9,
      "evidence_grade": "A",
      "summary": "Disjoint held-out set, N, CI, and harness reported."
    },
    "decontamination": {
      "score": 0,
      "evidence_grade": "F",
      "summary": "No train/test overlap probe or parser adversarial probe."
    }
  }
}
```

Generated reviewer note:

> This result is useful as a platform measurement but not as an algorithm-level GRPO claim. The paper is honest about hidden backend internals, which improves scoping, but MIN-REPORT-RL items 1 and 2 remain unauditable.

---

# Practical heuristics that make the checker robust

StackCheck-RL should not merely keyword-match. It should use a layered evidence policy:

**1. Prefer explicit values over method names.**  
“GRPO” gets no credit for loss details unless ratio, clipping, mask, and advantage normalization are actually stated.

**2. Prefer reported evidence over inferred defaults.**  
If TRL’s default config implies a value, but the paper never reports it, the item gets recoverability credit but not full reporting credit.

**3. Treat “default settings” as an anti-pattern.**  
Default settings are version-dependent and often invisible to readers.

**4. Separate “absent” from “unknown.”**  
“KL absent, no frozen reference” is a reportable choice. Silence is not.

**5. Cross-check paper, config, and logs.**  
A paper that says `G=8` but logs `G=4,8,16` under adaptive sampling should be flagged.

**6. Require trajectories where the checklist says trajectories.**  
For ZVF/GU and group size, a final scalar is not equivalent to per-step evidence.

**7. Make closed-stack limitations visible, not fatal by default.**  
A managed backend can still score on sampler disclosure, held-out evaluation, and decontamination, but hidden loss/KL internals prevent full algorithm-level compliance.

**8. Emit reviewer-ready language.**  
The tool should produce copy-pasteable warnings: “Item 4 missing: no usable-signal trajectory,” not just “score low.”

---

## One-line badge format

For papers and repos, I would make the output badge intentionally blunt:

```text
MIN-REPORT-RL: 94/100 ✅ compliant
MIN-REPORT-RL: 63/100 ⚠ partial; missing ZVF/GU + decontamination
MIN-REPORT-RL: 21/100 ❌ non-compliant; stack under-specified
MIN-REPORT-RL: 57/100 ⚠ platform-scoped; closed loss/KL
```

That turns the paper’s proposed “cost is a JSON block” adoption path into a concrete automated gate: a repo either exposes the seven stack fields, or the checker tells reviewers exactly which unreported lever could be confounding the claimed GRPO gain.

---
