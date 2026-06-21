# GRPO Registry Format Proposal
- ID `6a379a57-a814-83e8-8768-476bfcc2407c` created 2026-06-21 08:01 UTC | model gpt-5-5-pro

---

## QUERY

Read the attached paper. Invent a public registry format for GRPO-family variants where each entry records the defining delta plus the full MIN-REPORT-RL block. Provide the JSON schema, an example entry for DAPO, and a query API.

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

## GRPO Variant Registry, v0.1

I’d make the registry’s atomic object **not** “DAPO” or “GSPO” alone, but **a variant identity plus one or more stack reports**. That matches the paper’s central claim: an algorithm label is not a full experimental treatment, because loss form, KL handling, sampler/backend, precision, reward parsing, group schedule, held-out evaluation, and decontamination can all change the result. fileciteturn0file0 The attached paper also says the controlled audit should hold the full Min-Report-RL block identical across arms except the variant’s defining delta; Table 2 lists DAPO’s defining delta as “clip-higher + dyn. sampling.” fileciteturn1file4

The key registry rule is:

> **Every registry entry must contain a `defining_delta` and at least one `report` whose `min_report_rl` object has all seven required fields. Unknown values are allowed, but only as explicit `unknown` / `not_reported` fields, never by omission.**

---

## 1. JSON Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://grpo-registry.org/schemas/gvr-entry-v0.1.0.schema.json",
  "title": "GRPO-family Variant Registry Entry",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "registry_id",
    "variant",
    "defining_delta",
    "reports",
    "governance"
  ],
  "properties": {
    "schema_version": {
      "const": "gvr.entry.v0.1.0"
    },
    "registry_id": {
      "type": "string",
      "pattern": "^gvr:[a-z0-9][a-z0-9_.-]*$"
    },
    "variant": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "canonical_name",
        "family",
        "base_algorithm",
        "aliases"
      ],
      "properties": {
        "canonical_name": {
          "type": "string"
        },
        "family": {
          "type": "string",
          "examples": ["GRPO-family"]
        },
        "base_algorithm": {
          "type": "string",
          "examples": ["GRPO"]
        },
        "aliases": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "uniqueItems": true
        },
        "parents": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "tags": {
          "type": "array",
          "items": {
            "type": "string"
          },
          "uniqueItems": true
        }
      }
    },
    "defining_delta": {
      "$ref": "#/$defs/DefiningDelta"
    },
    "reports": {
      "type": "array",
      "minItems": 1,
      "items": {
        "$ref": "#/$defs/Report"
      }
    },
    "audit_summary": {
      "$ref": "#/$defs/AuditSummary"
    },
    "governance": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "entry_status",
        "created_at",
        "updated_at",
        "maintainers"
      ],
      "properties": {
        "entry_status": {
          "enum": [
            "draft_stub",
            "submitted",
            "verified",
            "disputed",
            "superseded",
            "retracted"
          ]
        },
        "created_at": {
          "type": "string",
          "format": "date-time"
        },
        "updated_at": {
          "type": "string",
          "format": "date-time"
        },
        "maintainers": {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "change_log_uri": {
          "type": ["string", "null"],
          "format": "uri"
        }
      }
    },
    "extensions": {
      "type": "object",
      "additionalProperties": true
    }
  },
  "$defs": {
    "Status": {
      "enum": [
        "reported",
        "partial",
        "not_reported",
        "unknown",
        "not_applicable",
        "planned",
        "redacted",
        "inferred",
        "verified"
      ]
    },
    "Evidence": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "kind",
        "status"
      ],
      "properties": {
        "kind": {
          "enum": [
            "paper",
            "code",
            "config",
            "run_log",
            "telemetry",
            "artifact",
            "maintainer_attestation",
            "audit_note"
          ]
        },
        "status": {
          "$ref": "#/$defs/Status"
        },
        "citation_key": {
          "type": ["string", "null"]
        },
        "uri": {
          "type": ["string", "null"],
          "format": "uri"
        },
        "locator": {
          "type": ["string", "null"],
          "description": "Page, section, table, commit, config path, or artifact path."
        },
        "sha256": {
          "type": ["string", "null"],
          "pattern": "^[a-fA-F0-9]{64}$"
        },
        "note": {
          "type": ["string", "null"]
        }
      }
    },
    "SourceRef": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "source_type",
        "status"
      ],
      "properties": {
        "source_type": {
          "enum": [
            "paper",
            "preprint",
            "official_repo",
            "third_party_repo",
            "controlled_audit",
            "library_default",
            "registry_stub"
          ]
        },
        "status": {
          "$ref": "#/$defs/Status"
        },
        "title": {
          "type": ["string", "null"]
        },
        "citation_key": {
          "type": ["string", "null"]
        },
        "uri": {
          "type": ["string", "null"],
          "format": "uri"
        },
        "version": {
          "type": ["string", "null"]
        },
        "accessed_at": {
          "type": ["string", "null"],
          "format": "date-time"
        }
      }
    },
    "DefiningDelta": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "base_algorithm",
        "summary",
        "delta_types",
        "hooks",
        "target_invariance"
      ],
      "properties": {
        "base_algorithm": {
          "type": "string"
        },
        "summary": {
          "type": "string"
        },
        "delta_types": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "string",
            "examples": [
              "loss.clip.upper_asymmetry",
              "loss.advantage_normalization",
              "importance_sampling.sequence_level",
              "group_size.adaptive",
              "rollout.dynamic_sampling",
              "reward.diversity_bonus"
            ]
          },
          "uniqueItems": true
        },
        "hooks": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "hook_id",
              "hook_type",
              "semantics",
              "target_min_report_rl_paths",
              "parameters"
            ],
            "properties": {
              "hook_id": {
                "type": "string"
              },
              "hook_type": {
                "type": "string"
              },
              "semantics": {
                "type": "string"
              },
              "target_min_report_rl_paths": {
                "type": "array",
                "items": {
                  "type": "string",
                  "description": "JSON Pointer-like path into min_report_rl."
                }
              },
              "parameters": {
                "type": "array",
                "items": {
                  "type": "object",
                  "additionalProperties": false,
                  "required": [
                    "name",
                    "status",
                    "value"
                  ],
                  "properties": {
                    "name": {
                      "type": "string"
                    },
                    "status": {
                      "$ref": "#/$defs/Status"
                    },
                    "value": {},
                    "unit": {
                      "type": ["string", "null"]
                    }
                  }
                }
              }
            }
          }
        },
        "target_invariance": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "rule",
            "non_delta_paths_expected_constant"
          ],
          "properties": {
            "rule": {
              "enum": [
                "all_min_report_rl_except_delta_paths",
                "declared_subset",
                "none"
              ]
            },
            "non_delta_paths_expected_constant": {
              "type": "array",
              "items": {
                "type": "string"
              }
            }
          }
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "Report": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "report_id",
        "report_type",
        "lifecycle_status",
        "source",
        "implementation",
        "min_report_rl",
        "completeness"
      ],
      "properties": {
        "report_id": {
          "type": "string"
        },
        "report_type": {
          "enum": [
            "published",
            "official_code_reconstruction",
            "controlled_audit",
            "third_party_replication",
            "library_default"
          ]
        },
        "lifecycle_status": {
          "enum": [
            "draft_stub",
            "submitted",
            "verified",
            "disputed",
            "superseded",
            "retracted"
          ]
        },
        "source": {
          "$ref": "#/$defs/SourceRef"
        },
        "implementation": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "trainer",
            "repo_uri",
            "commit",
            "config_uri"
          ],
          "properties": {
            "trainer": {
              "type": ["string", "null"]
            },
            "repo_uri": {
              "type": ["string", "null"],
              "format": "uri"
            },
            "commit": {
              "type": ["string", "null"]
            },
            "config_uri": {
              "type": ["string", "null"],
              "format": "uri"
            },
            "artifact_bundle_uri": {
              "type": ["string", "null"],
              "format": "uri"
            }
          }
        },
        "min_report_rl": {
          "$ref": "#/$defs/MinReportRL"
        },
        "results": {
          "type": "object",
          "additionalProperties": true,
          "properties": {
            "published_delta": {
              "type": ["number", "null"]
            },
            "controlled_last10": {
              "type": ["number", "null"]
            },
            "controlled_heldout": {
              "type": ["number", "null"]
            },
            "controlled_delta": {
              "type": ["number", "null"]
            },
            "confidence_interval_95": {
              "type": ["array", "null"],
              "items": {
                "type": "number"
              },
              "minItems": 2,
              "maxItems": 2
            },
            "survival_verdict": {
              "enum": [
                "survives",
                "shrinks",
                "disappears",
                "reverses",
                "not_audited",
                "descriptive_only",
                null
              ]
            }
          }
        },
        "completeness": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "required_items_present",
            "required_items_reported",
            "missing_paths",
            "completeness_score"
          ],
          "properties": {
            "required_items_present": {
              "type": "integer",
              "minimum": 0,
              "maximum": 7
            },
            "required_items_reported": {
              "type": "integer",
              "minimum": 0,
              "maximum": 7
            },
            "missing_paths": {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            "completeness_score": {
              "type": "number",
              "minimum": 0,
              "maximum": 1
            }
          }
        }
      }
    },
    "MinReportRL": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "loss_form",
        "reference_policy_kl",
        "sampler_backend_precision",
        "usable_signal_trajectory",
        "group_size_schedule",
        "heldout_evaluation",
        "decontamination_probe"
      ],
      "properties": {
        "loss_form": {
          "$ref": "#/$defs/LossForm"
        },
        "reference_policy_kl": {
          "$ref": "#/$defs/ReferencePolicyKL"
        },
        "sampler_backend_precision": {
          "$ref": "#/$defs/SamplerBackendPrecision"
        },
        "usable_signal_trajectory": {
          "$ref": "#/$defs/UsableSignalTrajectory"
        },
        "group_size_schedule": {
          "$ref": "#/$defs/GroupSizeSchedule"
        },
        "heldout_evaluation": {
          "$ref": "#/$defs/HeldoutEvaluation"
        },
        "decontamination_probe": {
          "$ref": "#/$defs/DecontaminationProbe"
        }
      }
    },
    "LossForm": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "ppo_ratio",
        "ratio_granularity",
        "clip",
        "token_mask",
        "advantage_normalization",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "ppo_ratio": {
          "type": ["boolean", "null"]
        },
        "ratio_granularity": {
          "enum": ["token", "sequence", "batch", "none", "other", "unknown", null]
        },
        "clip": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "enabled",
            "lower",
            "upper",
            "asymmetric",
            "clip_higher"
          ],
          "properties": {
            "enabled": {
              "type": ["boolean", "null"]
            },
            "lower": {
              "type": ["number", "null"]
            },
            "upper": {
              "type": ["number", "null"]
            },
            "asymmetric": {
              "type": ["boolean", "null"]
            },
            "clip_higher": {
              "type": ["boolean", "null"]
            },
            "schedule": {
              "type": ["object", "null"],
              "additionalProperties": true
            }
          }
        },
        "token_mask": {
          "enum": [
            "completion_only",
            "whole_sequence",
            "prompt_and_completion",
            "custom",
            "unknown",
            null
          ]
        },
        "advantage_normalization": {
          "enum": [
            "per_group",
            "per_batch",
            "running",
            "none",
            "other",
            "unknown",
            null
          ]
        },
        "length_normalization": {
          "enum": [
            "per_token_mean",
            "sum",
            "fixed_denominator",
            "sequence_length",
            "other",
            "unknown",
            null
          ]
        },
        "loss_equation_uri": {
          "type": ["string", "null"],
          "format": "uri"
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "ReferencePolicyKL": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "reference_policy_retained",
        "kl_placement",
        "kl_estimator",
        "kl_coefficient",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "reference_policy_retained": {
          "type": ["boolean", "null"]
        },
        "reference_checkpoint": {
          "type": ["string", "null"]
        },
        "kl_placement": {
          "enum": ["loss", "reward", "both", "absent", "unknown", null]
        },
        "kl_estimator": {
          "enum": [
            "forward_per_token",
            "reverse_per_token",
            "forward_per_sequence",
            "reverse_per_sequence",
            "other",
            "unknown",
            null
          ]
        },
        "kl_coefficient": {
          "type": ["number", "null"]
        },
        "kl_schedule": {
          "type": ["object", "null"],
          "additionalProperties": true
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "SamplerBackendPrecision": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "rollout_engine",
        "backend_kind",
        "decoding",
        "sampler_precision",
        "trainer_precision",
        "tokenizer",
        "chat_template",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "rollout_engine": {
          "type": ["string", "null"]
        },
        "backend_kind": {
          "enum": ["open", "managed", "closed", "mixed", "unknown", null]
        },
        "decoding": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "seeded"
          ],
          "properties": {
            "temperature": {
              "type": ["number", "null"]
            },
            "top_p": {
              "type": ["number", "null"]
            },
            "top_k": {
              "type": ["integer", "null"]
            },
            "max_tokens": {
              "type": ["integer", "null"]
            },
            "seeded": {
              "type": ["boolean", "null"]
            }
          }
        },
        "sampler_precision": {
          "enum": ["fp32", "tf32", "bf16", "fp16", "int8", "mixed", "unknown", null]
        },
        "trainer_precision": {
          "enum": ["fp32", "tf32", "bf16", "fp16", "int8", "mixed", "unknown", null]
        },
        "tokenizer": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "name_or_path",
            "revision",
            "shared_with_trainer"
          ],
          "properties": {
            "name_or_path": {
              "type": ["string", "null"]
            },
            "revision": {
              "type": ["string", "null"]
            },
            "shared_with_trainer": {
              "type": ["boolean", "null"]
            }
          }
        },
        "chat_template": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "name",
            "sha256"
          ],
          "properties": {
            "name": {
              "type": ["string", "null"]
            },
            "sha256": {
              "type": ["string", "null"],
              "pattern": "^[a-fA-F0-9]{64}$"
            }
          }
        },
        "dynamic_sampling": {
          "type": ["object", "null"],
          "additionalProperties": true
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "UsableSignalTrajectory": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "metric_kind",
        "logged_per_step",
        "trajectory_uri",
        "summary",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "metric_kind": {
          "enum": [
            "zvf_gu",
            "reward_variance",
            "gradient_norm_variance",
            "other",
            "not_reported"
          ]
        },
        "logged_per_step": {
          "type": ["boolean", "null"]
        },
        "trajectory_uri": {
          "type": ["string", "null"],
          "format": "uri"
        },
        "trajectory_inline": {
          "type": ["array", "null"],
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": [
              "step"
            ],
            "properties": {
              "step": {
                "type": "integer",
                "minimum": 0
              },
              "zvf": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 1
              },
              "gu": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 1
              },
              "reward_variance": {
                "type": ["number", "null"]
              },
              "mean_reward": {
                "type": ["number", "null"]
              },
              "n_prompts": {
                "type": ["integer", "null"]
              }
            }
          }
        },
        "summary": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "mean_zvf_step_25": {
              "type": ["number", "null"]
            },
            "mean_gu_step_25": {
              "type": ["number", "null"]
            },
            "collapse_rate_seeds": {
              "type": ["number", "null"]
            },
            "time_to_collapse_median_steps": {
              "type": ["number", "null"]
            },
            "collapse_rule": {
              "type": ["string", "null"]
            }
          }
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "GroupSizeSchedule": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "kind",
        "initial_g",
        "per_step_uri",
        "adaptive_rule",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "kind": {
          "enum": ["fixed", "adaptive", "curriculum", "unknown", null]
        },
        "initial_g": {
          "type": ["integer", "null"],
          "minimum": 1
        },
        "final_g": {
          "type": ["integer", "null"],
          "minimum": 1
        },
        "per_step_uri": {
          "type": ["string", "null"],
          "format": "uri"
        },
        "per_step_inline": {
          "type": ["array", "null"],
          "items": {
            "type": "object",
            "required": ["step", "g"],
            "properties": {
              "step": {
                "type": "integer",
                "minimum": 0
              },
              "g": {
                "type": "integer",
                "minimum": 1
              }
            }
          }
        },
        "adaptive_rule": {
          "type": ["object", "null"],
          "additionalProperties": true
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "HeldoutEvaluation": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "reward_environment_disjoint",
        "dataset",
        "split",
        "n",
        "evaluator",
        "metrics",
        "checkpoint_selection",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "reward_environment_disjoint": {
          "type": ["boolean", "null"]
        },
        "dataset": {
          "type": ["string", "null"]
        },
        "split": {
          "type": ["string", "null"]
        },
        "n": {
          "type": ["integer", "null"],
          "minimum": 0
        },
        "prompt_hash_manifest_uri": {
          "type": ["string", "null"],
          "format": "uri"
        },
        "evaluator": {
          "type": ["string", "null"]
        },
        "metrics": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["name", "value"],
            "properties": {
              "name": {
                "type": "string"
              },
              "value": {
                "type": ["number", "null"]
              },
              "standard_error": {
                "type": ["number", "null"]
              },
              "confidence_interval_95": {
                "type": ["array", "null"],
                "items": {
                  "type": "number"
                },
                "minItems": 2,
                "maxItems": 2
              }
            }
          }
        },
        "checkpoint_selection": {
          "enum": [
            "pre_registered",
            "final_step",
            "best_training_reward",
            "best_validation",
            "unknown",
            null
          ]
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "DecontaminationProbe": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "status",
        "train_test_overlap_probe",
        "parser_adversarial_probe",
        "evidence"
      ],
      "properties": {
        "status": {
          "$ref": "#/$defs/Status"
        },
        "train_test_overlap_probe": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "method",
            "passed",
            "max_overlap"
          ],
          "properties": {
            "method": {
              "type": ["string", "null"]
            },
            "passed": {
              "type": ["boolean", "null"]
            },
            "max_overlap": {
              "type": ["number", "null"]
            },
            "artifact_uri": {
              "type": ["string", "null"],
              "format": "uri"
            }
          }
        },
        "parser_adversarial_probe": {
          "type": "object",
          "additionalProperties": false,
          "required": [
            "method",
            "passed",
            "failure_modes"
          ],
          "properties": {
            "method": {
              "type": ["string", "null"]
            },
            "passed": {
              "type": ["boolean", "null"]
            },
            "failure_modes": {
              "type": "array",
              "items": {
                "type": "string"
              }
            },
            "artifact_uri": {
              "type": ["string", "null"],
              "format": "uri"
            }
          }
        },
        "evidence": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/Evidence"
          }
        }
      }
    },
    "AuditSummary": {
      "type": "object",
      "additionalProperties": false,
      "properties": {
        "audit_status": {
          "enum": [
            "not_audited",
            "planned",
            "running",
            "complete",
            "disputed"
          ]
        },
        "baseline_report_id": {
          "type": ["string", "null"]
        },
        "controlled_report_id": {
          "type": ["string", "null"]
        },
        "survival_verdict": {
          "enum": [
            "survives",
            "shrinks",
            "disappears",
            "reverses",
            "not_audited",
            "descriptive_only",
            null
          ]
        },
        "controlled_delta": {
          "type": ["number", "null"]
        },
        "confidence_interval_95": {
          "type": ["array", "null"],
          "items": {
            "type": "number"
          },
          "minItems": 2,
          "maxItems": 2
        },
        "notes": {
          "type": ["string", "null"]
        }
      }
    }
  }
}
```

---

## 2. Example entry: DAPO

This DAPO entry deliberately does **not** invent missing paper-specific values. The attached draft identifies DAPO’s registry-level defining delta as “clip-higher + dynamic sampling,” but its audit tables still contain TODO cells, so unknown values are represented as explicit `unknown` / `not_reported` fields. fileciteturn1file4

```json
{
  "schema_version": "gvr.entry.v0.1.0",
  "registry_id": "gvr:dapo",
  "variant": {
    "canonical_name": "DAPO",
    "family": "GRPO-family",
    "base_algorithm": "GRPO",
    "aliases": ["DAPO"],
    "parents": ["GRPO"],
    "tags": [
      "loss-clip",
      "clip-higher",
      "dynamic-sampling",
      "group-relative-rl"
    ]
  },
  "defining_delta": {
    "base_algorithm": "GRPO",
    "summary": "DAPO is registered as a GRPO-family variant whose defining delta is clip-higher plus dynamic sampling.",
    "delta_types": [
      "loss.clip.upper_asymmetry",
      "rollout.dynamic_sampling"
    ],
    "hooks": [
      {
        "hook_id": "clip_higher",
        "hook_type": "loss_clip_rule",
        "semantics": "Use an asymmetric upper clipping rule relative to the GRPO baseline. Exact bound values must be supplied by the source report or audit implementation.",
        "target_min_report_rl_paths": [
          "/loss_form/clip/asymmetric",
          "/loss_form/clip/upper",
          "/loss_form/clip/clip_higher"
        ],
        "parameters": [
          {
            "name": "clip_upper",
            "status": "unknown",
            "value": null,
            "unit": null
          },
          {
            "name": "clip_lower",
            "status": "unknown",
            "value": null,
            "unit": null
          }
        ]
      },
      {
        "hook_id": "dynamic_sampling",
        "hook_type": "rollout_selection_rule",
        "semantics": "Apply a dynamic rollout or sample-selection rule. Exact prompt/completion filtering, regeneration, or eligibility rule must be supplied by the source report or audit implementation.",
        "target_min_report_rl_paths": [
          "/sampler_backend_precision/dynamic_sampling",
          "/usable_signal_trajectory",
          "/group_size_schedule"
        ],
        "parameters": [
          {
            "name": "selection_rule",
            "status": "unknown",
            "value": null,
            "unit": null
          },
          {
            "name": "resampling_budget",
            "status": "unknown",
            "value": null,
            "unit": "rollouts"
          }
        ]
      }
    ],
    "target_invariance": {
      "rule": "all_min_report_rl_except_delta_paths",
      "non_delta_paths_expected_constant": [
        "/reference_policy_kl",
        "/sampler_backend_precision/rollout_engine",
        "/sampler_backend_precision/backend_kind",
        "/sampler_backend_precision/sampler_precision",
        "/sampler_backend_precision/trainer_precision",
        "/sampler_backend_precision/tokenizer",
        "/sampler_backend_precision/chat_template",
        "/heldout_evaluation",
        "/decontamination_probe"
      ]
    },
    "evidence": [
      {
        "kind": "paper",
        "status": "reported",
        "citation_key": "min_report_rl_draft",
        "uri": null,
        "locator": "Table 2: DAPO row",
        "sha256": null,
        "note": "The attached Min-Report-RL draft lists DAPO as 'clip-higher + dyn. sampling'."
      }
    ]
  },
  "reports": [
    {
      "report_id": "dapo-2025-as-published-stub",
      "report_type": "published",
      "lifecycle_status": "draft_stub",
      "source": {
        "source_type": "paper",
        "status": "partial",
        "title": "DAPO paper",
        "citation_key": "dapo2025",
        "uri": null,
        "version": null,
        "accessed_at": null
      },
      "implementation": {
        "trainer": null,
        "repo_uri": null,
        "commit": null,
        "config_uri": null,
        "artifact_bundle_uri": null
      },
      "min_report_rl": {
        "loss_form": {
          "status": "partial",
          "ppo_ratio": null,
          "ratio_granularity": "unknown",
          "clip": {
            "enabled": true,
            "lower": null,
            "upper": null,
            "asymmetric": true,
            "clip_higher": true,
            "schedule": null
          },
          "token_mask": "unknown",
          "advantage_normalization": "unknown",
          "length_normalization": "unknown",
          "loss_equation_uri": null,
          "evidence": [
            {
              "kind": "paper",
              "status": "partial",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Table 2 and Min-Report-RL item 1",
              "sha256": null,
              "note": "Only the clip-higher delta is available from the attached draft; other loss details remain unknown in this registry stub."
            }
          ]
        },
        "reference_policy_kl": {
          "status": "not_reported",
          "reference_policy_retained": null,
          "reference_checkpoint": null,
          "kl_placement": "unknown",
          "kl_estimator": "unknown",
          "kl_coefficient": null,
          "kl_schedule": null,
          "evidence": [
            {
              "kind": "audit_note",
              "status": "not_reported",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Registry stub",
              "sha256": null,
              "note": "Not specified in the attached draft's DAPO row."
            }
          ]
        },
        "sampler_backend_precision": {
          "status": "partial",
          "rollout_engine": null,
          "backend_kind": "unknown",
          "decoding": {
            "temperature": null,
            "top_p": null,
            "top_k": null,
            "max_tokens": null,
            "seeded": null
          },
          "sampler_precision": "unknown",
          "trainer_precision": "unknown",
          "tokenizer": {
            "name_or_path": null,
            "revision": null,
            "shared_with_trainer": null
          },
          "chat_template": {
            "name": null,
            "sha256": null
          },
          "dynamic_sampling": {
            "enabled": true,
            "rule": null,
            "rule_status": "unknown"
          },
          "evidence": [
            {
              "kind": "paper",
              "status": "partial",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Table 2: DAPO row",
              "sha256": null,
              "note": "Dynamic sampling is listed as part of the DAPO defining delta; rollout engine and precision remain unknown here."
            }
          ]
        },
        "usable_signal_trajectory": {
          "status": "not_reported",
          "metric_kind": "not_reported",
          "logged_per_step": null,
          "trajectory_uri": null,
          "trajectory_inline": null,
          "summary": {
            "mean_zvf_step_25": null,
            "mean_gu_step_25": null,
            "collapse_rate_seeds": null,
            "time_to_collapse_median_steps": null,
            "collapse_rule": null
          },
          "evidence": [
            {
              "kind": "audit_note",
              "status": "not_reported",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Table 3 TODO cells",
              "sha256": null,
              "note": "The attached draft's telemetry table leaves DAPO ZVF/GU cells to be filled from the audit corpus."
            }
          ]
        },
        "group_size_schedule": {
          "status": "unknown",
          "kind": "unknown",
          "initial_g": null,
          "final_g": null,
          "per_step_uri": null,
          "per_step_inline": null,
          "adaptive_rule": null,
          "evidence": [
            {
              "kind": "audit_note",
              "status": "unknown",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Table 3 TODO cells",
              "sha256": null,
              "note": "DAPO's dynamic sampling is known, but this stub does not infer a group-size schedule."
            }
          ]
        },
        "heldout_evaluation": {
          "status": "not_reported",
          "reward_environment_disjoint": null,
          "dataset": null,
          "split": null,
          "n": null,
          "prompt_hash_manifest_uri": null,
          "evaluator": null,
          "metrics": [],
          "checkpoint_selection": "unknown",
          "evidence": [
            {
              "kind": "audit_note",
              "status": "not_reported",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Table 2 TODO cells",
              "sha256": null,
              "note": "Controlled held-out results are placeholders in the attached draft."
            }
          ]
        },
        "decontamination_probe": {
          "status": "not_reported",
          "train_test_overlap_probe": {
            "method": null,
            "passed": null,
            "max_overlap": null,
            "artifact_uri": null
          },
          "parser_adversarial_probe": {
            "method": null,
            "passed": null,
            "failure_modes": [],
            "artifact_uri": null
          },
          "evidence": [
            {
              "kind": "audit_note",
              "status": "not_reported",
              "citation_key": "min_report_rl_draft",
              "uri": null,
              "locator": "Registry stub",
              "sha256": null,
              "note": "No DAPO-specific decontamination or parser-probe results are provided in the attached draft."
            }
          ]
        }
      },
      "results": {
        "published_delta": null,
        "controlled_last10": null,
        "controlled_heldout": null,
        "controlled_delta": null,
        "confidence_interval_95": null,
        "survival_verdict": "not_audited"
      },
      "completeness": {
        "required_items_present": 7,
        "required_items_reported": 1,
        "missing_paths": [
          "/loss_form/ppo_ratio",
          "/loss_form/ratio_granularity",
          "/loss_form/token_mask",
          "/loss_form/advantage_normalization",
          "/reference_policy_kl",
          "/sampler_backend_precision/rollout_engine",
          "/sampler_backend_precision/sampler_precision",
          "/sampler_backend_precision/trainer_precision",
          "/sampler_backend_precision/tokenizer",
          "/sampler_backend_precision/chat_template",
          "/usable_signal_trajectory",
          "/group_size_schedule",
          "/heldout_evaluation",
          "/decontamination_probe"
        ],
        "completeness_score": 0.14
      }
    }
  ],
  "audit_summary": {
    "audit_status": "planned",
    "baseline_report_id": null,
    "controlled_report_id": null,
    "survival_verdict": "not_audited",
    "controlled_delta": null,
    "confidence_interval_95": null,
    "notes": "The attached Min-Report-RL paper proposes a controlled single-stack audit; numeric DAPO results are not yet filled."
  },
  "governance": {
    "entry_status": "draft_stub",
    "created_at": "2026-06-21T00:00:00Z",
    "updated_at": "2026-06-21T00:00:00Z",
    "maintainers": ["grpo-registry-maintainers"],
    "change_log_uri": null
  },
  "extensions": {}
}
```

---

## 3. Query API

### Core objects

The registry exposes three main resources:

```text
VariantEntry
  └── reports[]
        └── min_report_rl
              ├── loss_form
              ├── reference_policy_kl
              ├── sampler_backend_precision
              ├── usable_signal_trajectory
              ├── group_size_schedule
              ├── heldout_evaluation
              └── decontamination_probe
```

This follows the paper’s audit framing: for each variant, store both “as published” and “controlled stack” readings when available, and judge survival by the controlled delta against a shared GRPO baseline. fileciteturn1file4

---

### REST endpoints

```http
GET /v1/variants
```

List variants.

Supported query parameters:

```text
q                         free-text search
family                    e.g. GRPO-family
base_algorithm            e.g. GRPO
delta_type                e.g. loss.clip.upper_asymmetry
tag                       e.g. dynamic-sampling
entry_status              draft_stub | verified | disputed | superseded | retracted
audit_status              not_audited | planned | running | complete | disputed
survival_verdict          survives | shrinks | disappears | reverses | not_audited
min_completeness          float from 0 to 1
include                   reports,audit_summary,defining_delta
limit                     page size
cursor                    pagination cursor
```

Example:

```http
GET /v1/variants?base_algorithm=GRPO&delta_type=loss.clip.upper_asymmetry&include=defining_delta,audit_summary
```

---

```http
GET /v1/variants/{registry_id}
```

Fetch one full registry entry.

Example:

```http
GET /v1/variants/gvr:dapo?include=reports
```

---

```http
GET /v1/reports
```

Search report-level stack records.

Supported query parameters:

```text
variant                   registry id, e.g. gvr:dapo
report_type               published | controlled_audit | third_party_replication
trainer                   TRL | verl | OpenRLHF | custom
rollout_engine            vLLM | SGLang | managed_api | trainer_generate
backend_kind              open | managed | closed | mixed
token_mask                completion_only | whole_sequence | custom | unknown
kl_placement              loss | reward | both | absent | unknown
sampler_precision         fp32 | bf16 | fp16 | mixed | unknown
trainer_precision         fp32 | bf16 | fp16 | mixed | unknown
group_schedule_kind       fixed | adaptive | curriculum | unknown
zvf_gu_available          true | false
heldout_disjoint          true | false
decontamination_passed    true | false
missing                   JSON Pointer path, e.g. /reference_policy_kl
```

Example: find DAPO-like reports that actually logged ZVF/GU.

```http
GET /v1/reports?delta_type=rollout.dynamic_sampling&zvf_gu_available=true
```

---

```http
GET /v1/reports/{report_id}/min-report-rl
```

Return only the seven-field Min-Report-RL block.

Example:

```http
GET /v1/reports/dapo-2025-as-published-stub/min-report-rl
```

---

```http
GET /v1/reports/{report_id}/telemetry/usable-signal
```

Stream per-step telemetry when available.

Query parameters:

```text
metric                    zvf_gu | reward_variance | gradient_norm_variance
from_step                 integer
to_step                   integer
format                    jsonl | csv | parquet
cursor                    pagination cursor
```

Example JSONL row:

```json
{
  "report_id": "dapo-controlled-audit-001",
  "step": 25,
  "zvf": 0.62,
  "gu": 0.38,
  "mean_reward": 0.41,
  "n_prompts": 512
}
```

---

### Comparability API

The most important endpoint is not simple search; it is **stack comparability**.

```http
POST /v1/compare/stack-equivalence
Content-Type: application/json
```

Request:

```json
{
  "baseline_report_id": "grpo-baseline-controlled-001",
  "candidate_report_id": "dapo-controlled-audit-001",
  "rule": "all_min_report_rl_except_delta_paths",
  "delta_paths": [
    "/loss_form/clip/asymmetric",
    "/loss_form/clip/upper",
    "/loss_form/clip/clip_higher",
    "/sampler_backend_precision/dynamic_sampling"
  ],
  "require_evidence": true
}
```

Response:

```json
{
  "stack_equivalent": false,
  "verdict": "not_comparable",
  "matched_paths": [
    "/reference_policy_kl/kl_placement",
    "/sampler_backend_precision/rollout_engine"
  ],
  "differing_paths": [
    {
      "path": "/sampler_backend_precision/trainer_precision",
      "baseline_value": "bf16",
      "candidate_value": "unknown",
      "severity": "blocking"
    }
  ],
  "missing_paths": [
    "/decontamination_probe",
    "/heldout_evaluation/prompt_hash_manifest_uri"
  ],
  "notes": "Candidate report is missing fields needed to assert that only the DAPO delta changed."
}
```

---

### Structured search API

For richer public queries:

```http
POST /v1/search
Content-Type: application/json
```

Example: “Find GRPO-family variants whose defining delta touches clipping and whose controlled audit survived.”

```json
{
  "where": {
    "variant.base_algorithm": "GRPO",
    "defining_delta.delta_types": {
      "contains": "loss.clip.upper_asymmetry"
    },
    "reports": {
      "some": {
        "report_type": "controlled_audit",
        "results.survival_verdict": {
          "in": ["survives", "shrinks"]
        },
        "completeness.completeness_score": {
          "gte": 0.95
        }
      }
    }
  },
  "select": [
    "registry_id",
    "variant.canonical_name",
    "defining_delta.summary",
    "audit_summary.survival_verdict",
    "audit_summary.controlled_delta"
  ],
  "order_by": [
    {
      "path": "audit_summary.controlled_delta",
      "direction": "desc"
    }
  ],
  "limit": 50
}
```

---

### Missing-report query

This supports the paper’s adoption goal: make omissions visible and auditable.

```http
GET /v1/reports?missing=/usable_signal_trajectory&base_algorithm=GRPO
```

Response sketch:

```json
{
  "data": [
    {
      "report_id": "dapo-2025-as-published-stub",
      "variant": "DAPO",
      "missing_paths": [
        "/usable_signal_trajectory",
        "/heldout_evaluation",
        "/decontamination_probe"
      ],
      "completeness_score": 0.14
    }
  ],
  "next_cursor": null
}
```

---

### Canonical registry queries

These are the queries I’d expect reviewers and audit authors to use most:

```http
GET /v1/variants?base_algorithm=GRPO&include=audit_summary
```

Show all GRPO-family variants and their survival status.

```http
GET /v1/reports?variant=gvr:dapo&report_type=published
```

Show DAPO as published.

```http
GET /v1/reports?variant=gvr:dapo&report_type=controlled_audit
```

Show DAPO in controlled single-stack audits.

```http
GET /v1/reports?missing=/reference_policy_kl
```

Find papers that do not say where KL lives.

```http
GET /v1/reports?token_mask=whole_sequence&heldout_disjoint=false
```

Find potentially confounded reports using whole-sequence loss and no disjoint held-out split.

```http
POST /v1/compare/stack-equivalence
```

Ask whether a DAPO-vs-GRPO comparison really differs only by DAPO’s declared delta.

---

## 4. Registry invariants

The public registry should enforce these invariants:

1. **No silent omissions.** All seven Min-Report-RL fields must be present in every `report`.
2. **Unknown is a value.** `unknown` and `not_reported` are valid but queryable.
3. **No result without stack.** A `controlled_delta` may be present only if the corresponding `controlled_audit` report contains a full `min_report_rl` block.
4. **No comparability by label.** Two reports are comparable only if `/v1/compare/stack-equivalence` passes after ignoring the declared delta paths.
5. **Telemetry is externalizable.** Full per-step ZVF/GU can live in a linked artifact, but the report must state whether it exists and provide summaries.
6. **Published and controlled readings are separate.** A variant can have one as-published stack and multiple controlled-audit stacks.
7. **Delta paths are machine-readable.** Every defining delta must point to the Min-Report-RL paths it changes, so audits can mechanically check “all else equal.”

---
