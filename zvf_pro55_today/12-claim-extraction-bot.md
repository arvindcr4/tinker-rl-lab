# Claim Extraction Bot
- ID `6a37a9b3-a2b0-83e8-8e59-f5addc5ddbcd` created 2026-06-21 09:07 UTC | model gpt-5-5-pro

---

## QUERY

Read the attached paper. Invent a bot that scans GRPO-family papers, extracts every claim tied to a specific stack lever, and checks whether that lever is actually reported in the cited paper. Explain the claim-extraction model and the report format.

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

## Bot: **LeverTrace**

**LeverTrace** is a citation-audit bot for GRPO-family papers. It reads a paper, finds every atomic claim that depends on a stack lever, follows the citation attached to that claim, and checks whether the cited paper actually reports the lever detail needed to support the claim.

The motivation comes directly from the attached Min-Report-RL paper: it argues that “GRPO” is not a complete experimental description, because runs can differ in loss form, KL/reference handling, sampler/backend/precision, group-size schedule, reward parser, held-out evaluation, and telemetry while still wearing the same algorithm label. The paper’s core point is that algorithm rankings are really rankings over full stacks, not labels alone. fileciteturn1file1 fileciteturn1file11

LeverTrace is therefore an upstream companion to the paper’s proposed `grpo-stackdiff`: `grpo-stackdiff` compares manifests; LeverTrace builds the claim/evidence ledger that tells reviewers whether the cited papers contain enough information to build those manifests. The paper already defines a useful downstream taxonomy: lever, diff kind, role, flip-risk class, and verdict. LeverTrace reuses that vocabulary but applies it to claims and citations rather than only to two completed run manifests. fileciteturn1file0

---

## 1. What LeverTrace scans for

LeverTrace extracts only **lever-bound claims**: claims whose truth, attribution, or interpretability depends on one of the Min-Report-RL stack fields.

The lever ontology is:

| Lever | What LeverTrace looks for | Required reporting evidence |
|---|---|---|
| **L0: Provenance** | trainer, version, seed, hardware, model/tokenizer hashes, optimizer, LoRA config | explicit run identity or linked artifact |
| **L1: Loss form** | PPO ratio, clipping, clip bounds, asymmetric clip-higher, token mask, advantage normalization, length normalization | formula, objective description, mask definition, normalization rule |
| **L2: Reference policy / KL** | frozen reference, absent reference, KL in loss vs reward, KL coefficient, schedule, forward/reverse KL, token/sequence KL | explicit KL/reference description |
| **L3: Sampler / backend / precision** | vLLM/SGLang/API/generate, temperature, top-p, max tokens, bf16/fp16/fp32, tokenizer/chat template sharing | sampler config and precision/tokenizer details |
| **L4: Usable-signal telemetry** | ZVF, GU, reward variance, gradient-signal proxy, collapse rule | per-step trajectory or substitute metric |
| **L5: Group-size schedule** | fixed G, adaptive G, dynamic sampling, rollout allocation, AERO-style rules | per-step G or adaptive rule |
| **L6: Held-out evaluation** | disjoint held-out split, evaluator, sample size, confidence interval, checkpoint-selection rule | held-out protocol separate from training reward |
| **L7: Decontamination / parser robustness** | train/test overlap, n-gram or embedding checks, adversarial format-only parser tests, reward parser behavior | probe results and parser tests |

This matches the attached paper’s seven mandatory stack fields, where each field is included because it can flip a comparison: loss form, KL/reference handling, sampler/backend/precision, ZVF/GU telemetry, group-size schedule, held-out evaluation, and decontamination/parser probes. fileciteturn1file8 fileciteturn1file9

---

## 2. Claim-extraction model

LeverTrace uses a **recall-first, schema-constrained claim model**. The model does not simply summarize papers. It converts text into atomic audit records.

### Claim tuple

Every extracted claim is represented as:

```json
{
  "claim_id": "C-0042",
  "claim_text": "DAPO improves GRPO by using clip-higher and dynamic sampling.",
  "source_location": {
    "paper": "paper_under_review.pdf",
    "page": 6,
    "section": "Method"
  },
  "claim_type": [
    "method_attribution",
    "comparison_mechanism"
  ],
  "subject": "DAPO",
  "baseline_or_target": "GRPO",
  "metric_or_outcome": null,
  "lever_tags": [
    {
      "lever": "L1.loss_form",
      "sublever": "clip_bounds",
      "required_slots": ["ratio_formula", "clip_low", "clip_high", "mask", "advantage_norm"]
    },
    {
      "lever": "L5.group_size_schedule",
      "sublever": "dynamic_sampling",
      "required_slots": ["group_size_rule", "sampling_rule", "compute_accounting"]
    }
  ],
  "citations": [
    {
      "key": "dapo2025",
      "anchor_text": "[cite: dapo2025]",
      "citation_scope": "same_sentence"
    }
  ],
  "polarity": "positive",
  "confidence": 0.91
}
```

### Claim types

LeverTrace separates claims into several types because they require different verification standards:

| Claim type | Example | What must be checked |
|---|---|---|
| **Method-attribution claim** | “GSPO uses a sequence-level IS ratio.” | Does the cited GSPO paper report that ratio and its formula? |
| **Implementation claim** | “We use vLLM with bf16 rollouts.” | Does the paper report engine, precision, decoding parameters, tokenizer/template? |
| **Comparison claim** | “Method A beats GRPO by 2.4 points.” | Are both stacks reported well enough to compare? |
| **Mechanism claim** | “The gain comes from dynamic sampling.” | Is the lever reported and isolated from nuisance stack deltas? |
| **Telemetry claim** | “The method avoids collapse.” | Are ZVF/GU or substitute signal metrics reported per step? |
| **Capability claim** | “The method improves GSM8K accuracy.” | Is there a held-out split, harness, sample size, CI, and checkpoint-selection rule? |
| **Integrity claim** | “The evaluation is uncontaminated.” | Is there a train/test overlap probe and parser robustness test? |

The paper itself gives the reason for treating capability claims specially: online training reward is not capability evidence, and a variant can show a training-reward gain that vanishes on held-out evaluation. fileciteturn1file9

---

## 3. Extraction pipeline

LeverTrace runs in five stages.

### Stage A — Document normalization

The bot ingests PDFs, LaTeX, appendices, supplementary files, and linked code/configs. It preserves:

```text
section boundaries
paragraphs
equations
tables
figure captions
bibliography entries
inline citation anchors
appendix/hyperparameter tables
```

Tables matter because stack levers are often reported in small experimental-setting tables rather than prose.

### Stage B — Candidate claim detection

A high-recall detector marks candidate spans using three signals:

1. **Citation neighborhoods**: sentences or paragraphs near citations such as “DAPO,” “GSPO,” “Dr.GRPO,” “AERO,” or “GRPO baseline.”
2. **Lever lexicon triggers**: words like `clip`, `ratio`, `KL`, `reference`, `vLLM`, `top-p`, `temperature`, `G=`, `group size`, `held-out`, `contamination`, `ZVF`, `reward parser`.
3. **Result/attribution syntax**: phrases like “improves over,” “outperforms,” “we attribute,” “uses,” “changes,” “removes,” “normalizes,” “reports,” “does not report.”

The detector is intentionally over-inclusive. False positives are cheaper than missed claims.

### Stage C — Atomic claim splitting

A sentence like:

> “DAPO improves GRPO through clip-higher, dynamic sampling, and better held-out accuracy.”

becomes three separate records:

```text
C1: DAPO uses clip-higher.                 → L1
C2: DAPO uses dynamic sampling.            → L5
C3: DAPO improves held-out accuracy.       → L6
```

This matters because one citation may support the loss-form claim but not the held-out-evaluation claim.

### Stage D — Lever tagging and slot generation

A multi-label classifier maps each atomic claim to one or more levers. Then a deterministic slot generator expands the lever into required evidence.

For example:

```json
{
  "lever": "L2.reference_policy_kl",
  "claim": "The baseline is KL-regularized GRPO.",
  "required_slots": [
    "frozen_reference_policy",
    "kl_location_loss_or_reward",
    "kl_coefficient",
    "kl_schedule",
    "kl_estimator",
    "per_token_or_per_sequence"
  ]
}
```

This strictness follows the Min-Report-RL standard: “KL-regularized” is not enough unless the paper reports where KL enters the objective, how it is estimated, and how it is scheduled. fileciteturn1file8

### Stage E — Citation resolution

The citation resolver assigns each claim to a target paper or artifact:

```text
same-sentence citation     → high confidence
same-paragraph citation    → medium confidence
section-level citation     → low confidence
no citation                → MISSING_CITATION
TODO placeholder citation  → TARGET_UNRESOLVED
```

For the attached paper itself, many references are intentionally TODO stubs, so LeverTrace would not pretend those citations are verified. It would mark claims tied to `dapo2025`, `gspo2025`, `drgrpo2025`, and similar placeholder keys as `TARGET_UNRESOLVED` until real bibliography entries are supplied. The paper explicitly says the bibliography is a manual TODO stub that should not ship as-is. fileciteturn1file12

---

## 4. Evidence-checking model

Once a claim is extracted, LeverTrace asks a narrower question:

> “Does the cited paper actually report the lever detail needed to support this claim?”

It does **not** ask:

> “Is the algorithm good?”  
> “Is the result real?”  
> “Would the gain survive reimplementation?”

That survival question belongs to the controlled audit proposed in the paper. LeverTrace only audits claim-to-report coverage.

### Evidence tuple

For each claim, the evidence checker produces:

```json
{
  "evidence_id": "E-0042-a",
  "target_paper": "dapo2025.pdf",
  "target_location": {
    "page": 4,
    "section": "Algorithm"
  },
  "evidence_text": "The objective clips the upper ratio bound at ...",
  "evidence_type": "formula",
  "matched_slots": ["ratio_formula", "clip_high"],
  "missing_slots": ["token_mask", "advantage_norm"],
  "coverage": "partial",
  "confidence": 0.84
}
```

### Verification verdicts

| Verdict | Meaning |
|---|---|
| **SUPPORTED_REPORTED** | The cited paper explicitly reports all required lever slots. |
| **SUPPORTED_PARTIAL** | The cited paper reports the lever but omits mandatory details. |
| **UNREPORTED** | The cited paper does not report the lever. |
| **CONTRADICTED** | The cited paper reports a different value or mechanism than the claim states. |
| **TARGET_UNRESOLVED** | The citation cannot be resolved, e.g. TODO key, missing bibliography, dead link. |
| **OPAQUE** | The cited paper points to a closed backend or artifact whose lever cannot be inspected. |
| **AMBIGUOUS_SCOPE** | The claim has multiple citations or a paragraph-level citation and the support cannot be uniquely assigned. |
| **NOT_A_LEVER_CLAIM** | The candidate was extracted but does not depend on a stack lever. |

For closed or managed backends, the bot does not infer hidden details. It marks the field `OPAQUE`, consistent with the paper’s point that closed runners should be scoped as a platform implementation rather than treated as fully specified “GRPO.” fileciteturn1file2

---

## 5. Risk scoring

LeverTrace reuses the attached paper’s `grpo-stackdiff` risk vocabulary, but applies it at the claim level.

| Risk | Claim-level meaning |
|---|---|
| **R0 same** | The cited paper reports the lever exactly as claimed. |
| **R1 cosmetic** | Only naming/version/cosmetic detail differs. |
| **R2 small** | Missing detail is unlikely to affect the claim’s interpretation. |
| **R3 material** | Missing or partial lever report could shrink the claim. |
| **R4 flip_capable** | Missing/contradicted lever is capable of flipping the comparison. |
| **R5 invalidating** | The comparison target is not common or the citation target is wrong. |
| **RU unknown** | Evidence is missing, unresolved, or opaque. |

The paper’s own downstream taxonomy includes these risk classes and verdicts for stack comparisons; LeverTrace uses them to prioritize reviewer attention. fileciteturn1file0

A missing **L1 loss-form** detail in a claimed 2–5 point GRPO-family improvement is usually high risk, because the paper argues that many GRPO variants are small deltas on the base loop and that unreported stack effects can exceed the claimed method effect. fileciteturn1file6

---

## 6. Report format

LeverTrace emits two reports:

1. a **human-readable audit report** for reviewers;
2. a **machine-readable JSON report** for CI, reproducibility checklists, and `grpo-stackdiff`.

### Human-readable report

```markdown
# LeverTrace Citation Audit Report

## 1. Executive summary

Source paper: <paper title>
Corpus scanned: <N source papers, M cited targets>
Claims extracted: <count>
Lever-bound claims: <count>
Fully supported: <count>
Partially supported: <count>
Unreported: <count>
Contradicted: <count>
Unresolved / opaque: <count>

Overall verdict: UNVERIFIABLE | STACK_CLAIMS_PARTIAL | STACK_CLAIMS_REPORTED

Highest-risk findings:
- C-0042: DAPO clip-higher claim cites dapo2025, but citation target is unresolved. Risk: RU.
- C-0051: Baseline KL claim lacks KL coefficient and placement in cited paper. Risk: R4.
- C-0068: Held-out improvement claim reports training reward only. Risk: R3/R4.

## 2. Coverage by lever

| Lever | Claims | Supported | Partial | Unreported | Contradicted | Opaque/Unresolved |
|---|---:|---:|---:|---:|---:|---:|
| L1 Loss form | 12 | 5 | 4 | 1 | 0 | 2 |
| L2 Reference/KL | 8 | 2 | 3 | 2 | 0 | 1 |
| L3 Sampler/backend/precision | 9 | 3 | 2 | 2 | 0 | 2 |
| L4 ZVF/GU telemetry | 3 | 0 | 1 | 2 | 0 | 0 |
| L5 Group-size schedule | 6 | 2 | 2 | 1 | 0 | 1 |
| L6 Held-out evaluation | 10 | 4 | 3 | 2 | 0 | 1 |
| L7 Decontamination/parser | 4 | 1 | 1 | 2 | 0 | 0 |

## 3. Claim ledger

| ID | Source claim | Lever | Citation target | Required evidence | Evidence found | Verdict | Risk |
|---|---|---|---|---|---|---|---|
| C-0042 | “DAPO uses clip-higher + dynamic sampling.” | L1, L5 | dapo2025 | clip bounds; dynamic sampling rule | citation unresolved | TARGET_UNRESOLVED | RU |
| C-0051 | “The baseline is KL-regularized GRPO.” | L2 | baseline paper | frozen ref; KL placement; beta schedule | KL mentioned, beta omitted | SUPPORTED_PARTIAL | R4 |
| C-0068 | “The method improves held-out GSM8K.” | L6 | same paper | disjoint split; harness; n; CI | accuracy table only | SUPPORTED_PARTIAL | R3 |
| C-0074 | “No contamination affects the benchmark.” | L7 | appendix | overlap probe; parser adversarial test | no probe found | UNREPORTED | R4 |

## 4. Per-cited-paper report cards

### dapo2025
Status: TARGET_UNRESOLVED
Claims depending on this target: C-0042, C-0043, C-0044
Action: replace TODO citation with real paper and rerun.

### gspo2025
Status: PARTIAL
Reported levers: L1 sequence-level ratio
Missing levers: sampler precision, held-out CI

## 5. Reviewer action list

1. Require real bibliography entries for unresolved targets.
2. Ask authors to report missing L1/L2/L3 fields before accepting any algorithmic-gain claim.
3. Treat L6 capability claims as training-dynamics claims unless held-out disjointness, harness, n, CI, and checkpoint rule are reported.
4. Queue R4/R5 findings for human review.
```

### Machine-readable JSON

```json
{
  "levertrace_version": "0.1.0",
  "source_paper": {
    "title": "Example GRPO-family paper",
    "sha256": "abc123",
    "parsed_from": ["pdf", "latex", "appendix"]
  },
  "summary": {
    "claims_total": 87,
    "lever_bound_claims": 42,
    "supported_reported": 18,
    "supported_partial": 11,
    "unreported": 6,
    "contradicted": 1,
    "target_unresolved": 4,
    "opaque": 2,
    "overall_verdict": "STACK_CLAIMS_PARTIAL"
  },
  "claims": [
    {
      "claim_id": "C-0042",
      "claim_text": "DAPO uses clip-higher and dynamic sampling.",
      "source_location": {
        "page": 6,
        "section": "Related Work",
        "char_start": 18420,
        "char_end": 18473
      },
      "claim_type": ["method_attribution"],
      "subject": "DAPO",
      "lever_tags": [
        {
          "lever": "L1.loss_form",
          "sublever": "clip_higher",
          "required_slots": ["ratio_formula", "clip_low", "clip_high", "token_mask", "advantage_norm"]
        },
        {
          "lever": "L5.group_size_schedule",
          "sublever": "dynamic_sampling",
          "required_slots": ["sampling_rule", "per_step_group_size", "compute_accounting"]
        }
      ],
      "citations": [
        {
          "key": "dapo2025",
          "resolved": false,
          "reason": "bibliography_entry_is_todo_stub"
        }
      ],
      "evidence_checks": [],
      "verdict": "TARGET_UNRESOLVED",
      "risk": "RU",
      "confidence": {
        "claim_extraction": 0.94,
        "lever_mapping": 0.89,
        "citation_resolution": 0.22,
        "evidence_verification": null
      },
      "recommended_action": "Resolve citation target and rerun evidence check."
    }
  ]
}
```

---

## 7. How LeverTrace would treat the attached paper

For the attached paper, LeverTrace would extract claims such as:

| Claim in attached paper | Lever | Citation key | Likely LeverTrace result |
|---|---|---|---|
| “DAPO uses clip-higher + dynamic sampling.” | L1, L5 | `dapo2025` | `TARGET_UNRESOLVED` until real DAPO citation is supplied |
| “GSPO changes the importance-sampling granularity.” | L1 | `gspo2025` | `TARGET_UNRESOLVED` until real GSPO citation is supplied |
| “Dr.GRPO changes length/normalization in the loss.” | L1 | `drgrpo2025` | `TARGET_UNRESOLVED` until real Dr.GRPO citation is supplied |
| “AERO is adaptive-G.” | L5 | `aero2024` | `TARGET_UNRESOLVED` until real AERO citation is supplied |
| “ZVF/GU reveals whether the run had usable learning signal.” | L4 | `zvfaudit2026` | `TARGET_UNRESOLVED` because the v1 audit citation is still a TODO |
| “The proposed audit holds sampler, KL, LoRA, optimizer, parser, group size, held-out slice, harness, and seeds fixed.” | L1–L7 | same paper | `SUPPORTED_REPORTED` as a proposed protocol, not as completed evidence |

That is not a criticism of the draft; the draft explicitly labels these references as placeholders. It is exactly the kind of issue LeverTrace is designed to surface before submission.

---

## 8. One-sentence product definition

**LeverTrace is a GRPO-family citation auditor that turns prose claims into lever-tagged, citation-linked evidence checks, then reports whether each cited paper actually reports the stack detail needed to make the claim auditable.**

---
