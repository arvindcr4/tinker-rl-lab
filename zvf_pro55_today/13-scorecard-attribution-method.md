# Scorecard Attribution Method
- ID `6a37aa2b-92ec-83ee-af6d-28135a16b99d` created 2026-06-21 09:09 UTC | model gpt-5-5-pro

---

## QUERY

Read the attached paper. Invent a scorecard that attributes an observed head-to-head delta between two GRPO variants to stack differences vs. algorithmic differences. Define the attribution method, required measurements, and confidence language.

### File: zvf-program/position/min_report_rl.tex
Lines: 1-692
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
 37 | \newcommand{\todo}[1]{\textcolor{red}{\textbf{[TODO:\ #1]}}}
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
278 | own \texttt{generate}; decoding parameters (temperature, top-$p$,
279 | \texttt{max\_tokens}; logit precision in the sampler vs.\ the trainer (bf16 /
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
511 | The companion \texttt{grpo-stackdiff} tool described in \S\ref{sec:tooling}
512 | then consumes these manifests to flag confounded comparisons automatically.
513 | 
514 | \paragraph{Make the venues ask for it.}
515 | Add a \minreport{} block to the reproducibility checklist for the RL-for-LLM
516 | track, analogous to the existing NeurIPS/ICML reproducibility checklists. The
517 | companion \texttt{CHECKLIST.md} is written to be dropped verbatim into a paper's
518 | appendix.
519 | 
520 | \paragraph{Make it auditable post hoc.}
521 | For already-published work, the same JSON block can be reconstructed from
522 | release artifacts where they exist. The audit of \S\ref{sec:audit} doubles as a
523 | demonstration: it re-derives the \minreport{} block for four variants and shows
524 | the format is sufficient to detect confounded comparisons.
525 | 
526 | % =============================================================================
527 | \section{Objections and Responses}
528 | \label{sec:objections}
529 | % =============================================================================
530 | 
531 | \paragraph{``This is just `report your hyperparameters.'\,''}
532 | No. Hyperparameter tables already exist and the $17\times$~\todo{trace to v1 audit citation} gap occurred
533 | \emph{despite} matched visible hyperparameters. \minreport{} names the fields
534 | that hyperparameter tables systematically omit---loss form, KL placement,
535 | sampler precision, per-step \zvf{}/\gu{}, group-size schedule, held-out
536 | disjointness, contamination probe---precisely because each is a demonstrated
537 | flip lever, not a tuning knob.
538 | 
539 | \paragraph{``Seven items is too many / too few.''}
540 | Each item earns its place by a flip mechanism (\S\ref{sec:standard}); removing
541 | any one reopens a known confound. The list is a \emph{minimum}, not a maximum:
542 | authors may report more. We deliberately stop at seven because that is the set
543 | that, in our audit, was sufficient to explain the comparisons that flipped.
544 | \todo{If the audit surfaces an eighth recurring confound, extend the list and
545 | justify it the same way.}
546 | 
547 | \paragraph{``Re-implementations are unfaithful, so the audit is unfair.''}
548 | The audit reports \emph{both} readings (published and controlled) and frames the
549 | result as survival, not refutation. An unfaithful re-implementation is itself a
550 | finding: if a variant's gain depends on an unstated stack detail that a careful
551 | re-implementer cannot reproduce from the paper, that is a reporting failure the
552 | standard is designed to surface. We will publish every arm's full \minreport{}
553 | block so disagreements are about specific fields, not vibes.
554 | 
555 | \paragraph{``Closed/managed backends can't comply.''}
556 | Partly true, and that is the point: a closed runner cannot expose its loss form
557 | or KL handling, so results from it should be scoped as ``this platform's
558 | implementation,'' not ``GRPO'' \cite{zvfaudit2026}. The standard makes that
559 | scoping explicit rather than letting a closed-stack number stand in for an
560 | algorithm.
561 | 
562 | \paragraph{``\zvf{}/\gu{} is your own metric; why mandate it?''}
563 | \zvf{}/\gu{} is one cheap instantiation of a necessary quantity---\emph{did the
564 | run have usable group-relative signal?}---and we accept substitutes. Under
565 | dense process rewards or scaffolded exploration, \zvf{} degenerates and should
566 | be replaced by per-step reward variance or a gradient-norm-variance surrogate
567 | \cite{zvfaudit2026}. Item~4 mandates \emph{a usable-signal trajectory}, of which
568 | \zvf{}/\gu{} is the default; the point is that \emph{some} such telemetry must be
569 | reported.
570 | 
571 | % =============================================================================
572 | \section{Companion Tooling: \texttt{grpo-stackdiff}}
573 | \label{sec:tooling}
574 | % =============================================================================
575 | 
576 | A standard is easier to adopt if reviewers can check it mechanically. We sketch
577 | a companion CLI, \texttt{grpo-stackdiff}, that consumes two \minreport{}
578 | run manifests and reports whether their differences are large enough to flip a
579 | comparison. The tool is not a replacement for the audit of
580 | \S\ref{sec:audit}; it is an enforcement layer that tells a reviewer, in
581 | seconds, whether a head-to-head is stack-comparable.
582 | 
583 | \paragraph{Manifest contract.} Each run is a YAML/JSON block containing the
584 | seven \minreport{} fields, plus run identity (trainer, version, seed, hardware),
585 | model and tokenizer hashes, LoRA configuration, optimizer settings, and results.
586 | The schema is the JSON block proposed in \S\ref{sec:adoption}.
587 | 
588 | \paragraph{Diff taxonomy.} Every field difference receives three labels:
589 | \begin{itemize}[leftmargin=1.4em, itemsep=2pt]
590 |   \item \textbf{Lever.} One of eight lever groups: L0~run identity/provenance,
591 |     L1~loss form, L2~reference policy and KL handling, L3~sampler/backend/
592 |     precision, L4~usable-signal telemetry (\zvf/\gu{}), L5~group-size
593 |     schedule, L6~held-out evaluation and checkpoint selection, L7~decontamination
594 |     and parser robustness.
595 |   \item \textbf{Diff kind.} EQUAL, COSMETIC, PROVENANCE, PARAMETRIC, SCHEDULE,
596 |     SEMANTIC\_OBJECTIVE, DISTRIBUTIONAL, TELEMETRY, MISSING, OPAQUE, DERIVED, or
597 |     INVALID\_TARGET.
598 |   \item \textbf{Role.} TREATMENT\_DELTA if the user declares it as the
599 |     algorithmic delta being tested; NUISANCE\_DELTA if it is an uncontrolled
600 |     stack difference; COVERAGE\_GAP if the field is missing; INVALIDATOR if the
601 |     comparison target is not common.
602 | \end{itemize}
603 | 
604 | \paragraph{Flip-risk classes.} The tool compares each nuisance difference to
605 | the reported comparison margin and classifies the overall risk as one of:
606 | \begin{center}
607 | \small
608 | \begin{tabular}{@{}ll@{}}
609 | \toprule
610 | \textbf{Class} & \textbf{Meaning}\\
611 | \midrule
612 | R0 \texttt{same} & No meaningful difference.\\
613 | R1 \texttt{cosmetic} & Version string or comment only.\\
614 | R2 \texttt{small} & Estimated effect $<25\%$ of the comparison margin.\\
615 | R3 \texttt{material} & Effect $25$--$100\%$ of the margin; can shrink a claim.\\
616 | R4 \texttt{flip\_capable} & Effect bound $\	ext{\textgreater=}$ margin; can flip the ranking.\\
617 | R5 \texttt{invalidating} & The comparison target is not common.\\
618 | RU \texttt{unknown} & Missing or opaque evidence; comparison unverifiable.\\
619 | \bottomrule
620 | \end{tabular}
621 | \end{center}
622 | 
623 | \paragraph{Verdict and CI integration.} The tool emits a deterministic verdict
624 | (\texttt{STACK\_MATCHED}, \texttt{STACK\_MATERIAL}, \texttt{STACK\_CONFOUNDED},
625 | \texttt{UNVERIFIABLE}, or \texttt{INVALID\_COMPARISON}) and an exit code, so a
626 | paper's reproducibility workflow can fail a comparison that is
627 | flip-capable. Example:
628 | \begin{verbatim}
629 | grpo-stackdiff compare dapo.yaml grpo.yaml \
630 |   --metric heldout.exact_match --claim-delta +0.024 \
631 |   --fail-on flip-capable
632 | \end{verbatim}
633 | 
634 | \paragraph{Role in the \minreport{} ecosystem.} Authors emit the JSON block;
635 | venues ask for it; reviewers run \texttt{grpo-stackdiff}. The cost to an author
636 | is still a few lines of telemetry; the cost to a reviewer is one command. The
637 | tool turns the abstract stack-conditioning thesis of \S\ref{sec:stack} into a
638 | concrete, reproducible check.
639 | 
640 | % =============================================================================
641 | \section{Conclusion}
642 | \label{sec:conclusion}
643 | % =============================================================================
644 | 
645 | The RL-for-LLM literature is in a position the deep-RL community has seen
646 | before: a three-letter label is doing the work of a full experimental
647 | specification, and the unreported remainder of the stack can move a result by
648 | more than the algorithmic effect anyone is trying to claim. We have argued that
649 | this stack-conditioning is the default, demonstrated it with a matched-config
650 | comparison that flips by $17\times$, and proposed two coupled remedies: a
651 | seven-item minimum-reportable-stack (\minreport{}), where every item is a
652 | documented flip lever, and a controlled single-stack audit of DAPO, GSPO,
653 | Dr.GRPO, and MAD-GRPO that reports which claimed gains survive. Neither remedy
654 | requires new science---only a few fields of telemetry and the discipline to log
655 | them. The cost is a JSON block; the payoff is a literature whose comparisons one
656 | can actually trust.
657 | 
658 | % =============================================================================
659 | % Bibliography. MANUAL STUB ONLY. We do NOT invent authors/years. Every entry
660 | % is a TODO key to be replaced by a real citation before submission. When
661 | % promoting to a venue template, delete this block, use \usepackage{natbib},
662 | % and point \bibliography at a real .bib.
663 | %
664 | % SUBMISSION GATE: a reviewer of this manuscript will see a red
665 | % [cite:KEY] marker for every in-text citation and a red [TODO] marker
666 | % for every entry below. The paper will be treated as a draft, not as
667 | % a submission, until the bibliography is real. The 12 cited works are
668 | % all findable; if the position paper cannot stand on the citations it
669 | % has, the claims need to be cut, not the citations stubbed.
670 | % =============================================================================
671 | \section*{References}
672 | \small
673 | \noindent\textcolor{red}{\textbf{[TODO: replace this stub with a real .bib.
674 | Every \texttt{[cite:KEY]} marker above corresponds to one entry below. Do not
675 | ship with placeholder keys. Each entry below carries a [\textsc{required}]
676 | tag indicating its status; all are required for submission.]}}
677 | \begin{itemize}[leftmargin=1.4em, itemsep=1pt, label={}]
678 |   \item \texttt{zvfaudit2026} --- \todo{the ZVF Program v1 audit paper (this group's own work: ``Reward Contrast, Not Algorithm Labels''); cite the arXiv/venue version once assigned.}
679 |   \item \texttt{henderson2018deeprl} --- \todo{Henderson et al., ``Deep Reinforcement Learning that Matters'' (reproducibility crisis in deep RL).}
680 |   \item \texttt{islam2017reproducibility} --- \todo{Islam et al., reproducibility of benchmarked deep RL (verify exact reference).}
681 |   \item \texttt{dapo2025} --- \todo{DAPO paper (clip-higher + dynamic sampling GRPO variant); fill authors/venue/year.}
682 |   \item \texttt{gspo2025} --- \todo{GSPO paper (sequence-level importance sampling); fill authors/venue/year.}
683 |   \item \texttt{drgrpo2025} --- \todo{Dr.GRPO paper (length/normalization fix); fill authors/venue/year.}
684 |   \item \texttt{madgrpo2025} --- \todo{MAD-GRPO paper (multi-agent/diversity GRPO); fill authors/venue/year.}
685 |   \item \texttt{aero2024} --- \todo{AERO (adaptive rollout sizing); from variance-mitigation bib fragment.}
686 |   \item \texttt{cppo2024} --- \todo{CPPO (clip-pruned PPO); from variance-mitigation bib fragment.}
687 |   \item \texttt{ngrpo2025} --- \todo{NGRPO (normalized GRPO); from variance-mitigation bib fragment.}
688 |   \item \texttt{scafgrpo2025} --- \todo{Scaf-GRPO (scaffolded exploration); from variance-mitigation bib fragment.}
689 |   \item \texttt{dar2024} --- \todo{DAR (dual-alignment / dual-KL regularization); from extended-RW bib fragment.}
690 | \end{itemize}
691 | 
692 | \end{document}
```

---

## Pro 5.5 REPLY

Here’s a drop-in scorecard I’d add as a new appendix or as the formal backend for `grpo-stackdiff`.

The core idea is: do **not** ask “did variant A beat variant B?” Ask “how much of the observed A-on-stack-\(S_A\) vs B-on-stack-\(S_B\) delta survives when the stack is neutralized?” That matches the paper’s thesis that GRPO-family rankings are full-stack comparisons, not algorithm-label comparisons; the paper’s motivating example is exactly a matched visible GRPO config whose backend swap yielded 84.4% vs. 5.0% last-10 reward, with the omitted stack fields doing the causal work. fileciteturn0file0 fileciteturn1file14

## GRPO Delta Attribution Scorecard, GDAS

**Purpose.** Given an observed head-to-head delta between two GRPO variants, GDAS outputs:

\[
\Delta_{\text{obs}}
=
\Delta_{\text{algorithm}}
+
\Delta_{\text{stack}}
\quad
\text{plus an interaction/unknown warning when the evidence cannot separate them.}
\]

The scorecard has three products:

| Output | Meaning |
|---|---|
| **Algorithmic attribution** | The part of the delta that survives when both variants are run in the same stack. |
| **Stack attribution** | The part explained by backend, sampler, KL/reference handling, loss-mask differences, LoRA/optimizer, reward parser, evaluation harness, or other nuisance stack differences. |
| **Confidence language** | A constrained verbal claim: algorithm-dominant, stack-dominant, mixed, reversed, unverifiable, or invalid. |

Use **percentage-point deltas**, not ratios, for reward/accuracy metrics. A 5 pp held-out gain and a 17× training-reward ratio are different evidence types.

---

## 1. Estimand

Let:

\[
Y(v,s)
\]

be the mean performance of variant \(v\) in stack \(s\), measured on the pre-registered primary metric, preferably held-out exact match or another held-out task metric.

Let the published or observed comparison be:

\[
\Delta_{\text{obs}}
=
Y(A,S_A)-Y(B,S_B).
\]

This is the diagonal comparison: variant A in its own stack against variant B in its own stack.

GDAS defines the **algorithmic effect** as the effect of changing the declared GRPO variant while holding the stack fixed:

\[
\Delta_{\text{alg}}(s)
=
Y(A,s)-Y(B,s).
\]

It defines the **stack effect** as the effect of changing the stack while holding the algorithm fixed:

\[
\Delta_{\text{stack}}(v)
=
Y(v,S_A)-Y(v,S_B).
\]

The paper already gestures at this structure through its controlled audit proposal: report the published delta on each variant’s own stack, then report the controlled delta in one shared stack, with a survival verdict. fileciteturn1file3

---

## 2. Evidence tiers

The scorecard’s first field is the **identification tier**. This determines the maximum strength of the attribution language.

| Tier | Evidence available | What GDAS may say |
|---|---|---|
| **T0: Manifest-only** | Only run manifests and published numbers. | “Stack-comparable,” “stack-material,” “stack-confounded,” or “unverifiable.” No numeric algorithmic attribution. |
| **T1: Single-stack survival** | Both variants re-run in one pre-registered shared stack \(S^\*\). | “The claim survives / shrinks / disappears / reverses in a common stack.” Residual is **stack-associated**, not pure stack. |
| **T2: Crossed bridge** | Four cells: \(A,S_A\), \(A,S_B\), \(B,S_A\), \(B,S_B\). | Quantitative decomposition into algorithm main effect and stack main effect. |
| **T3: Crossed bridge + lever ablations** | T2 plus one-at-a-time or Shapley-style swaps of sampler, KL, loss mask, parser, etc. | Attribution to specific stack levers, not just “stack.” |
| **T4: Multi-stack replication** | T3 repeated across models, datasets, and backends. | Strongest general claim: “algorithmic effect is robust across tested stacks.” |

A single-seed comparison is always **descriptive only**. The paper’s statistical protocol says the independent unit is the seed, not the per-step reward trace, and recommends seed-level uncertainty, bootstrap 95% CIs, and multiple-test control over pre-registered survival tests. fileciteturn1file10

---

## 3. Attribution method

### 3.1 Gold-standard crossed bridge

Run the four-cell bridge:

|  | Stack \(S_A\) | Stack \(S_B\) |
|---|---:|---:|
| Variant A | \(Y(A,S_A)\) | \(Y(A,S_B)\) |
| Variant B | \(Y(B,S_A)\) | \(Y(B,S_B)\) |

Then compute:

\[
\widehat{\Delta}_{\text{alg}}
=
\frac{1}{2}
\left[
Y(A,S_A)-Y(B,S_A)
+
Y(A,S_B)-Y(B,S_B)
\right].
\]

This is the average same-stack algorithmic advantage.

Compute:

\[
\widehat{\Delta}_{\text{stack}}
=
\frac{1}{2}
\left[
Y(A,S_A)-Y(A,S_B)
+
Y(B,S_A)-Y(B,S_B)
\right].
\]

This is the average advantage of stack \(S_A\) over stack \(S_B\), averaged across the two variants.

These add exactly to the observed diagonal comparison:

\[
\Delta_{\text{obs}}
=
\widehat{\Delta}_{\text{alg}}
+
\widehat{\Delta}_{\text{stack}}.
\]

Also compute the algorithm-by-stack instability:

\[
\widehat{I}_{A\times S}
=
\frac{1}{2}
\left[
Y(A,S_A)-Y(B,S_A)
-
\bigl(Y(A,S_B)-Y(B,S_B)\bigr)
\right].
\]

This term is not added to the diagonal decomposition; it is a **fragility warning**. If it is large, the algorithmic effect is stack-dependent, even if the average algorithmic main effect is nonzero.

Report all three with bootstrap 95% CIs across seeds.

---

### 3.2 Single-stack survival estimate

When the full crossed bridge is too expensive, run both variants in one shared pre-registered stack \(S^\*\):

\[
\widehat{\Delta}_{\text{survive}}
=
Y(A,S^\*)-Y(B,S^\*).
\]

Then:

\[
\widehat{\Delta}_{\text{residual}}
=
\Delta_{\text{obs}}
-
\widehat{\Delta}_{\text{survive}}.
\]

Interpretation:

| Quantity | Interpretation |
|---|---|
| \(\widehat{\Delta}_{\text{survive}}\) | Same-stack algorithmic survival estimate. |
| \(\widehat{\Delta}_{\text{residual}}\) | Stack-associated residual: stack + algorithm-by-stack interaction + publication-stack effects. |

Do **not** call the residual “pure stack” unless the crossed bridge or lever ablations were run.

Survival fraction:

\[
f_{\text{survive}}
=
\frac{\widehat{\Delta}_{\text{survive}}}{\Delta_{\text{obs}}}.
\]

Use signed fractions, but report absolute magnitudes when signs conflict.

---

### 3.3 Manifest-only stack-risk attribution

When no bridge run exists, GDAS gives no causal attribution. It only reports whether the observed delta is **stack-comparable**.

Use the paper’s lever taxonomy: run identity/provenance, loss form, reference/KL, sampler/backend/precision, usable-signal telemetry, group-size schedule, held-out evaluation/checkpoint selection, and decontamination/parser robustness. The paper’s `grpo-stackdiff` section already classifies differences as treatment deltas, nuisance deltas, coverage gaps, or invalidators, and grades risk from same/cosmetic through material, flip-capable, invalidating, or unknown. fileciteturn1file1

For each nuisance difference \(j\), assign an effect bound \(b_j\):

| Risk class | Effect-bound rule |
|---|---|
| R0 same | \(b_j=0\). |
| R1 cosmetic | \(b_j=0\). |
| R2 small | \(b_j < 0.25|\Delta_{\text{obs}}|\). |
| R3 material | \(0.25|\Delta_{\text{obs}}| \le b_j < |\Delta_{\text{obs}}|\). |
| R4 flip-capable | \(b_j \ge |\Delta_{\text{obs}}|\). |
| R5 invalidating | Target is not common; stop. |
| RU unknown | Field missing/opaque; attribution unavailable. |

Conservative stack explainability bound:

\[
B_{\text{stack}}
=
\sum_j b_j.
\]

If \(B_{\text{stack}}\ge |\Delta_{\text{obs}}|\), the stack differences are sufficient to explain or flip the observed claim, so the comparison cannot support an algorithmic conclusion without bridge runs.

---

## 4. Required measurements

GDAS requires the paper’s Min-Report-RL manifest, plus raw seed-level results. The seven-item checklist is the natural base: loss form; reference/KL; sampler/backend/precision; per-step ZVF/GU; group-size schedule; held-out split; and decontamination/parser probes. The paper explicitly justifies each as a flip lever. fileciteturn1file7

| Lever | Required measurement | Attribution use |
|---|---|---|
| **L0 provenance** | Trainer/library and version, hardware, seed set, base checkpoint hash, tokenizer hash, chat template, LoRA targets/rank, optimizer settings, checkpoint-selection rule. | Invalidates or constrains the comparison target. Different base model/tokenizer/evaluator usually means no clean algorithmic claim. |
| **L1 loss form** | Ratio definition, old-policy handling, clipping bounds, completion-only vs whole-sequence mask, advantage normalization. | Semantic objective changes are high-risk stack or treatment differences. Count as algorithmic only if explicitly declared as the variant delta. |
| **L2 reference/KL** | Frozen reference present/absent, KL in loss vs reward, coefficient/schedule, forward/reverse, token/sequence KL. | Separates actual objective differences from “same GRPO” labels. |
| **L3 sampler/backend/precision** | vLLM/SGLang/API/trainer generate, decoding params, logit precision, max tokens, tokenizer/template parity. | Measures rollout-distribution differences; high-risk because sampler changes usable groups. |
| **L4 usable-signal telemetry** | Per-step ZVF, GU \(=1-\text{ZVF}\), reward variance substitute if ZVF is not appropriate, collapse rate, time-to-collapse. | Mechanistic attribution: a “better algorithm” may merely have more usable gradient signal. The paper’s telemetry table is designed for exactly this distinction. fileciteturn1file10 |
| **L5 group-size schedule** | \(G\) per step, adaptive rule, compute budget, total sampled completions/tokens. | Separates algorithmic adaptive-\(G\) claims from extra compute or easier mixed-group formation. |
| **L6 held-out evaluation** | Disjoint held-out set, evaluator/harness, sample size, confidence interval, primary metric, checkpoint-selection rule. | Prevents training reward or reward-harness selection from masquerading as capability. The paper states training reward is not capability evidence. fileciteturn1file13 |
| **L7 decontamination/parser probes** | Train/test overlap probe, parser adversarial tests, format-only and length-only controls. | Detects reward hacking, overlap, or parser artifacts. |

Additional raw data required:

| Data | Why |
|---|---|
| Per-seed held-out scores | Unit for inference. |
| Per-prompt paired held-out scores | Enables paired bootstrap or randomization tests. |
| Per-step online reward | Dynamics diagnostic only. |
| Per-step ZVF/GU or reward variance | Mechanistic stack-vs-algorithm evidence. |
| Per-step KL, entropy, completion length, parse success | Helps explain whether gains come from exploration, formatting, or length. |
| Rollout counts and token counts | Needed when adaptive sampling or dynamic group sizing changes compute. |
| Full manifests for both original and controlled runs | Required for attribution and `grpo-stackdiff` comparison. |

---

## 5. Scorecard fields

Each head-to-head comparison gets one GDAS card.

| Field | Fill-in rule |
|---|---|
| **Claim** | “Variant A beats variant B by \(\Delta_{\text{obs}}\) on metric \(M\).” |
| **Primary metric** | Held-out metric, not online reward unless explicitly scoped as training dynamics. |
| **Observed delta** | \(Y(A,S_A)-Y(B,S_B)\), with 95% CI. |
| **Evidence tier** | T0–T4. |
| **Algorithm estimate** | \(\widehat{\Delta}_{\text{alg}}\) from crossed bridge, or \(\widehat{\Delta}_{\text{survive}}\) from shared stack. |
| **Stack estimate** | \(\widehat{\Delta}_{\text{stack}}\) from crossed bridge; otherwise residual is “stack-associated.” |
| **Interaction warning** | \(\widehat{I}_{A\times S}\), with CI. |
| **Stack-risk bound** | \(B_{\text{stack}}/|\Delta_{\text{obs}}|\). |
| **Telemetry alignment** | “Matched,” “winner has more usable signal,” “collapse/saturation explains delta,” or “unmeasured.” |
| **Validity gates** | Held-out disjoint? Decontamination passed? Parser robust? |
| **Statistical gates** | Seed count, bootstrap CI, MDE/SESOI, FDR correction if multiple variants. |
| **Verdict** | Algorithm-dominant, stack-dominant, mixed, disappears, reverses, unverifiable, or invalid. |
| **Allowed wording** | The strongest sentence the evidence permits. |

---

## 6. Verdict rules

Let \(\delta_{\min}\) be the pre-registered smallest effect size of interest, for example 1 pp held-out exact match or the audit’s MDE.

| Verdict | Rule | Allowed wording |
|---|---|---|
| **Algorithm-dominant** | Same-stack or crossed \(\Delta_{\text{alg}}\) has same sign as \(\Delta_{\text{obs}}\), is at least 75% of \(\Delta_{\text{obs}}\), CI excludes 0 and exceeds \(\delta_{\min}\); stack component is below 25% or CI includes 0; interaction low. | “Most of the observed delta is attributable to the declared algorithmic difference under the tested stacks.” |
| **Algorithm-led mixed** | Algorithmic component is material and same sign, but stack component is also material. | “The gain survives, but the published magnitude is partly stack-conditioned.” |
| **Stack-led mixed** | Stack component is material and larger than algorithmic component; algorithm still nonzero. | “The comparison contains a surviving algorithmic effect, but most of the observed published delta is stack-associated.” |
| **Stack-dominated** | Algorithmic component is <25% of observed delta or not distinguishable from 0; stack component or stack-risk bound can explain the margin. | “The observed head-to-head is better explained as a stack difference than as an algorithmic improvement.” |
| **Disappears** | Shared-stack delta is within \(\pm\delta_{\min}\) or CI includes 0 and point estimate is small. | “The published gain does not survive stack control.” |
| **Reverses** | Shared-stack or crossed algorithmic estimate has the opposite sign and exceeds \(\delta_{\min}\). | “The ranking reverses under stack control.” |
| **Unverifiable** | Missing/opaque L1–L4 or L6 fields, closed backend with unknown loss/KL, no usable-signal telemetry, no seed-level data, or manifest-only with R4/RU differences. | “The comparison is between A-on-\(S_A\) and B-on-\(S_B\); no algorithmic attribution is warranted.” |
| **Invalid** | Different target model/task/evaluator, contaminated held-out set, non-comparable metric, or invalid checkpoint selection. | “This is not a valid head-to-head for the stated claim.” |

---

## 7. Confidence language

Use confidence as a cap, not decoration.

| Confidence | Requirements | Phrase template |
|---|---|---|
| **High confidence** | T2 or T3; complete manifests; held-out disjoint; decontamination/parser probes pass; seed-level bootstrap CIs; interaction low or explicitly reported. | “With high confidence, we attribute \(x\) pp \([CI]\) of the \(d\) pp observed delta to the algorithm and \(s\) pp \([CI]\) to stack differences under the tested stacks.” |
| **Moderate confidence** | T1 shared-stack survival; complete manifests; held-out valid; enough seeds; no crossed bridge. | “The claim partially/fully survives in a shared stack. The remaining delta is stack-associated, but stack and interaction are not separately identified.” |
| **Low confidence** | T0 manifest-only, incomplete bridge, wide CIs, weak seed count, or telemetry gaps. | “The observed delta is compatible with an algorithmic effect, but stack differences are material; the result should be scoped as an implementation comparison.” |
| **Descriptive only** | Single seed, training reward only, or no seed-level uncertainty. | “This run is descriptive and should not be used for inferential attribution.” |
| **Unverifiable** | Missing/opaque high-leverage fields or closed stack with unknown objective. | “No attribution is possible from the reported evidence.” |

Avoid wording like “A beats B” unless the verdict is algorithm-dominant or algorithm-led mixed. Prefer “A-on-\(S_A\) beats B-on-\(S_B\)” for uncontrolled comparisons.

---

## 8. Example report template

```text
GDAS Attribution Card

Claim:
  Variant A beats Variant B on held-out exact match.

Observed comparison:
  A on S_A: ___
  B on S_B: ___
  Δ_obs: ___ pp [95% CI ___, ___]

Evidence tier:
  T0 / T1 / T2 / T3 / T4

Controlled estimates:
  Δ_alg or Δ_survive: ___ pp [95% CI ___, ___]
  Δ_stack: ___ pp [95% CI ___, ___]
  Stack-associated residual: ___ pp
  A×S interaction: ___ pp [95% CI ___, ___]

Manifest risk:
  L0 provenance: R__
  L1 loss form: R__
  L2 reference/KL: R__
  L3 sampler/backend/precision: R__
  L4 usable-signal telemetry: R__
  L5 group-size schedule: R__
  L6 held-out/checkpoint/evaluator: R__
  L7 decontamination/parser: R__
  Stack-risk bound B_stack / |Δ_obs|: ___

Telemetry:
  Mean ZVF/GU trajectories matched? yes/no
  Collapse rate difference: ___
  Winner has more usable signal because of stack? yes/no/unknown

Verdict:
  Algorithm-dominant / algorithm-led mixed / stack-led mixed /
  stack-dominated / disappears / reverses / unverifiable / invalid

Allowed claim:
  “______________________________________________”
```

The important discipline is that GDAS separates **survival** from **attribution**. A shared-stack run can show that a gain survives, shrinks, disappears, or reverses. A crossed bridge can decompose the observed diagonal delta into algorithm and stack main effects. Lever ablations are needed to say *which* stack difference caused the stack component.

---
