# E9 — `mle_bench_eval` (OpenAI MLE-bench, 75 Kaggle competitions)

## 2026-08-22/23 Modal streaming execution update

A production-shaped single-competition path now runs on Modal without retaining
the 3.3 TB suite locally. It builds the pinned MLE-bench environment once, then
for each competition it downloads into ephemeral storage, verifies upstream
checksums, prepares the public/private split, runs generated code as `nonroot`,
invokes the native grader as root, returns artifacts, and deletes the dataset.
Concurrency is fixed at one.

The first eleven successfully graded competitions have completed. The
smallest, `spooky-author-identification` (1.9 MB), produced:

- pinned Tinker model: `pavlov-qwen36-tinker`
- immutable HF commit: `64444133c55d88c3f1bf0df8a2f5d7ac646125c8`
- W&B online before sampling: `hnuoymmo`
- native multi-class log-loss: **0.43391** (lower is better)
- valid submission: `true`; above median: `false`
- Tinker charge for the successful sampling run: **$0.00756003**
- replay used the exact saved solution and charged **$0.00** Tinker
- full 75-competition suite score: **`null`**

Complete replay artifacts:
`outputs/e9_mle_bench/modal_streaming/spooky-author-identification-df30738f5cc3/`.
The receipt, solution, and full 131 KB submission all pass their recorded
SHA-256 checks. The original sampled solution has the same solution hash as the
replay. The first sampled execution also produced a valid native grade of
0.43405 but downloaded only a preview; the complete replay receipt is the
preferred artifact.

Runner:
`zvf-program/flagship/modal_e9_mle_bench_streaming.py`. Planning, budget,
checkpoint/W&B gates, score-boundary, code extraction, and audited repair tests:
`zvf-program/flagship/test_e9_mle_bench_streaming.py` (8 passing).

The second-smallest competition, `detecting-insults-in-social-commentary`
(2.0 MB), also completed through the same native path:

- native AUROC: **0.87426** (higher is better)
- valid submission: `true`; above median: `true`
- gold threshold: 0.83321; native medal classification: **gold**
- successful sampling charge: **$0.008949** Tinker
- artifact hashes verified locally: receipt, native-grade payload, solution,
  and full 608 KB submission
- run artifacts:
  `outputs/e9_mle_bench/modal_streaming/detecting-insults-in-social-commentary-3c81ecb58f42/`

One preceding sample for this competition was rejected by the native grader and
correctly produced no score; it charged $0.00893658. The runner now preserves
invalid native reports instead of losing their diagnostics during ephemeral
cleanup, and the prompt handles suffixed sample-submission filenames. Total
Tinker charge for this competition's two attempts was **$0.01788558**. The
persistent campaign budget is $10.00, with $5.602712565 charged and
$4.397287435 remaining after the successful run.

The third-smallest competition, `us-patent-phrase-to-phrase-matching`
(2.14 MB), completed through the same native path:

- native score: **0.43676** (higher is better)
- valid submission: `true`; above median: `false`
- successful W&B run: `nppxr26z`
- successful sampling charge: **$0.01014186** Tinker
- receipt, native-grade payload, solution, and full 101 KB submission hashes
  all verified locally
- run artifacts:
  `outputs/e9_mle_bench/modal_streaming/us-patent-phrase-to-phrase-matching-1379ca868af6/`

One preceding sample did not create `submission.csv`, so it was rejected and no
score was recorded; it charged $0.01014186. The runner now also preserves the
solution, native report, and logs for missing-submission failures. Total Tinker
charge for the two attempts was **$0.02028372**. The persistent campaign budget
now has $5.622996285 charged and $4.377003715 remaining.

The fourth-smallest competition, `random-acts-of-pizza` (3.0 MB), completed
through the same native path after deterministic replay repairs:

- native AUROC: **0.60535** (higher is better)
- valid submission: `true`; above median: `true`
- retained sampled failure receipt:
  `random-acts-of-pizza-7ebdb299a489` (`AGENT_EXECUTION_FAILED`, score `null`)
- repaired exact-sample replay charged **$0.00** Tinker
- successful receipt, native-grade payload, repaired solution, and full 27 KB
  submission hashes all verified locally
- successful run artifacts:
  `outputs/e9_mle_bench/modal_streaming/random-acts-of-pizza-cc109c6bfc00/`

The sampled program initially mishandled train-only fields and mixed boolean
values in TF-IDF. The repairs only aligned train/test schemas, converted text
inputs to strings, preserved one TF-IDF row per request, and supplied neutral
defaults for absent optional fields. The original failed solution was restored
byte-for-byte after replay and still matches its receipt. Across the initial
failed sample, a transient bridge HTTP 500, and the retained sampled failure,
the persistent budget decreased by **$0.02517678**; both exact-solution replays
charged $0.00. The campaign now has $5.648173065 charged and $4.351826935
remaining.

The fifth-smallest competition, `tweet-sentiment-extraction` (3.0 MB), completed
through the same native path:

- native word-level Jaccard: **0.59324** (higher is better)
- valid submission: `true`; above median: `false`
- first sampled attempt: `AGENT_EXECUTION_FAILED`, score `null`; the response
  exhausted its token limit on planning and truncated its Python mid-statement
- compact-response resample: valid native grade, **$0.00757509** Tinker
- total Tinker charge across both samples: **$0.01635993**
- both receipts plus the successful native-grade payload, solution, and full
  231 KB submission hashes verify locally
- successful run artifacts:
  `outputs/e9_mle_bench/modal_streaming/tweet-sentiment-extraction-9d3132e24230/`

The agent prompt now requires the response to begin with an import, forbids
analysis and Markdown, and caps the requested implementation at 200 lines. The
campaign now has $5.664532995 charged and $4.335467005 remaining.

The sixth-smallest competition,
`nomad2018-predict-transparent-conductors` (6.24 MB), completed on its first
attempt:

- native score: **0.0678** (lower is better)
- valid submission: `true`; above median: `true`
- successful W&B run: `t3j131jx`
- Tinker charge: **$0.00874704**
- receipt, native-grade payload, solution, and full submission hashes all
  verified locally
- run artifacts:
  `outputs/e9_mle_bench/modal_streaming/nomad2018-predict-transparent-conductors-52b5301c552a/`

The persistent campaign now has $5.673280035 charged and $4.326719965
remaining.

The seventh successfully graded competition, `google-quest-challenge`,
completed on its first attempt on 2026-08-24:

- native score: **0.25726** (higher is better)
- valid submission: `true`; above median: `false`
- successful W&B run: `66q4fssl`
- Tinker charge: **$0.00441516**
- receipt, native-grade payload, solution, and submission hashes are recorded
  in the receipt
- run artifacts:
  `outputs/e9_mle_bench/modal_streaming/google-quest-challenge-71335d95ce21/`

The persistent campaign now has $5.691509655 charged and $4.308490345 gross
remaining; the bridge additionally reports a stale $0.0084009 reservation, so
the runner's fail-closed available balance is $4.300089445.

The eighth successfully graded competition, `aerial-cactus-identification`,
completed on its first attempt on 2026-08-24:

- native score: **0.5** (higher is better)
- Tinker charge: **$0.00511755**
- receipt, native-grade payload, solution, and submission hashes are recorded
  in the receipt
- run artifacts:
  `outputs/e9_mle_bench/modal_streaming/aerial-cactus-identification-93422ea5a6ef/`

The persistent campaign now has $5.696627205 charged and $4.303372795 gross
remaining; after the stale reservation, fail-closed availability is
$4.294971895.

The ninth successfully graded competition,
`chaii-hindi-and-tamil-question-answering`, completed on its first attempt on
2026-08-24 with native score **0.00618**, a valid submission, and a Tinker
charge of **$0.00480519**. Its locally verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/chaii-hindi-and-tamil-question-answering-c7928ad6b928/`.
The persistent campaign now has $5.701432395 charged and $4.290166705
fail-closed available after the stale reservation.

The tenth successfully graded competition, `leaf-classification`, required one
audited repair that dropped an already-present `id` column before restoring the
index as `id`. The exact saved-sample replay charged **$0.00**, produced a valid
native score of **0.88567** (lower is better), and is retained under
`outputs/e9_mle_bench/modal_streaming/leaf-classification-bdf29643b3d4/`.
Its receipt's repair label incorrectly names the older `hstack` repair; the
solution artifact and hash contain the actual duplicate-`id` repair. The label
selection bug is fixed and covered by 12 passing helper tests. A subsequent
$0 replay of the already repaired solution timed out at the enforced 900-second
limit and is retained separately as
`outputs/e9_mle_bench/modal_streaming/leaf-classification-0ff61340316b/`; it
does not replace or invalidate the earlier native grade.

The initial paid sample cost **$0.00255426**. The persistent campaign has
$5.703986655 charged and $4.287612445 fail-closed available after the stale
reservation.

The eleventh successfully graded competition,
`learning-agency-lab-automated-essay-scoring-2`, initially failed on an unused
scikit-learn metric import. An audited exact-sample replay removed only that
unused unavailable import, charged **$0.00**, and produced a valid native score
of **0.62867**. The successful artifacts are under
`outputs/e9_mle_bench/modal_streaming/learning-agency-lab-automated-essay-scoring-2-19945cd4c6b0/`.
The initial paid sample cost **$0.004730625**. The persistent campaign now has
$5.70871728 charged and $4.28288182 fail-closed available after the stale
reservation.

The twelfth successfully graded competition, `denoising-dirty-documents`,
initially failed because OpenCV was unavailable. A first exact replay replaced
OpenCV grayscale reads with Pillow, exposing one-based pixel coordinates in the
competition data. A second audited exact-sample replay converted only those
coordinates to zero-based indices, charged **$0.00**, and produced a valid
native score of **203.592** (lower is better). The successful artifacts are
under
`outputs/e9_mle_bench/modal_streaming/denoising-dirty-documents-ec618151b1d7/`.
The initial paid sample cost **$0.00361749**. The persistent campaign now has
$5.71233477 charged and $4.27926433 fail-closed available after the stale
reservation.

The thirteenth successfully graded competition,
`jigsaw-toxic-comment-classification-challenge`, completed on its first attempt
with native score **0.97424**, a valid submission, and a Tinker charge of
**$0.002130255**. Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/jigsaw-toxic-comment-classification-challenge-1338bfc1da6c/`.
The persistent campaign now has $5.714465025 charged and $4.277134075
fail-closed available after the stale reservation.

The fourteenth successfully graded competition, `lmsys-chatbot-arena`,
completed on its first attempt with native score **1.09861** (lower is better),
a valid submission, and a Tinker charge of **$0.004747905**. Its locally
hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/lmsys-chatbot-arena-2cfdd30613c2/`.
The persistent campaign now has $5.71921293 charged and $4.27238617
fail-closed available after the stale reservation.

The fifteenth successfully graded competition,
`the-icml-2013-whale-challenge-right-whale-redux`, initially produced an
invalid submission because ZIP member names retained a `test2/` prefix while
the native sample submission used basenames. An audited exact-sample replay
normalized only those identifiers, charged **$0.00**, and produced a valid
native score of **0.5**. The successful artifacts are under
`outputs/e9_mle_bench/modal_streaming/the-icml-2013-whale-challenge-right-whale-redux-8b39a9f70fae/`.
The initial paid sample cost **$0.004459755**. The persistent campaign now has
$5.723672685 charged and $4.267926415 fail-closed available after the stale
reservation.

The sixteenth successfully graded competition,
`statoil-iceberg-classifier-challenge`, initially attempted to read the native
single-file `.7z` inputs as plain UTF-8 and CSV. The image now includes pinned
`py7zr`, and an audited exact-sample replay extracted only those three archives,
charged **$0.00**, and produced a valid native score of **0.33831** (lower is
better). The successful artifacts are under
`outputs/e9_mle_bench/modal_streaming/statoil-iceberg-classifier-challenge-65cdb3830e2a/`.
The initial paid sample cost **$0.002630625**. The persistent campaign now has
$5.72630331 charged and $4.26529579 fail-closed available after the stale
reservation.

The seventeenth successfully graded competition,
`tabular-playground-series-may-2022`, completed on its first attempt with native
score **0.71992**, a valid submission, and a Tinker charge of **$0.004389945**.
Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/tabular-playground-series-may-2022-4d87d361aacd/`.
The persistent campaign now has $5.736275595 charged and $4.255323505
fail-closed available after the stale reservation. This cumulative amount also
includes the preserved **$0.00558234** structurally failed TGS Salt attempt.

The eighteenth successfully graded competition, `mlsp-2013-birds`, initially
failed on native header rows, a redundant unguarded optional `librosa` import,
and use of `LabelBinarizer` for an explicitly multi-label target. Audited
exact-sample replays skipped only the two identified headers, activated the
sample's existing dependency fallback, skipped the supplemental feature header,
and used `MultiLabelBinarizer` for the intended target representation. The
successful replay charged **$0.00** and produced a valid native score of
**0.6225**. Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/mlsp-2013-birds-b865b4ad5299/`. The
initial paid sample cost **$0.009016155**. The persistent campaign now has
$5.74529175 charged and $4.24630735 fail-closed available after the stale
reservation.

The nineteenth successfully graded competition,
`tabular-playground-series-dec-2021`, initially failed because XGBoost requires
zero-based classes while the competition labels are 1–7. An audited exact-sample
replay shifted only the training labels to 0–6 and restored predictions to 1–7
before submission. It charged **$0.00** and produced a valid native score of
**0.96075**, above the recorded gold threshold. Its locally hash-verified
artifacts are under
`outputs/e9_mle_bench/modal_streaming/tabular-playground-series-dec-2021-cb3e13ef7452/`.
The initial paid sample cost **$0.002105535**. The persistent campaign now has
$5.747397285 charged and $4.244201815 fail-closed available after the stale
reservation.

The twentieth successfully graded competition,
`ventilator-pressure-prediction`, completed on its first attempt with native
score **2.18084** (lower is better), a valid submission, and a Tinker charge of
**$0.00259623**. Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/ventilator-pressure-prediction-8eb00d0ff8f6/`.
The persistent campaign now has $5.749993515 charged and $4.241605585
fail-closed available after the stale reservation.

The twenty-first successfully graded competition,
`whale-categorization-playground`, completed on its first attempt with native
score **0.17932**, a valid submission, and a Tinker charge of **$0.00464163**.
Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/whale-categorization-playground-7556a5cbe2dd/`.
The persistent campaign now has $5.754635145 charged and $4.236963955
fail-closed available after the stale reservation.

The twenty-second successfully graded competition,
`dog-breed-identification`, initially failed because image filenames retained
their `.jpg` suffix while the label identifiers did not. An audited exact-sample
replay normalized both sides to filename stems, then a second $0 replay excluded
the two-column label file from sample-submission discovery and required the
competition's wide probability schema. The final replay charged **$0.00** and
produced a valid native score of **4.63268** (lower is better). Its locally
hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/dog-breed-identification-023030ffba88/`.
The initial paid sample cost **$0.00519786**. The persistent campaign now has
$5.759833005 charged and $4.231766095 fail-closed available after the stale
reservation.

The twenty-third successfully graded competition,
`plant-pathology-2020-fgvc7`, initially failed because the sampled program used
the obsolete `combinations` label instead of the prepared split's native
`multiple_diseases` label. An audited exact-sample replay changed only that
schema name, charged **$0.00**, and produced a valid native score of **0.63166**.
Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/plant-pathology-2020-fgvc7-8341fcafc069/`.
The initial paid sample cost **$0.00247023**. The persistent campaign now has
$5.762303235 charged and $4.229295865 fail-closed available after the stale
reservation.

The twenty-fourth successfully graded competition,
`dogs-vs-cats-redux-kernels-edition`, completed on its first attempt with native
score **0.62233** (lower is better), a valid submission, and a Tinker charge of
**$0.003681405**. Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/dogs-vs-cats-redux-kernels-edition-64afe753f533/`.
The persistent campaign now has $5.76598464 charged and $4.22561446 fail-closed
available after the stale reservation.

The twenty-fifth successfully graded competition,
`petfinder-pawpularity-score`, completed on its first attempt with native score
**20.17751** (lower is better), a valid submission, and a Tinker charge of
**$0.004681095**. Its locally hash-verified artifacts are under
`outputs/e9_mle_bench/modal_streaming/petfinder-pawpularity-score-e21af4986cea/`.
The persistent campaign now has $5.770665735 charged and $4.220933365
fail-closed available after the stale reservation.

The twenty-sixth successfully graded competition,
`champs-scalar-coupling`, initially failed because its intended one-hot loop
rebound a temporary dataframe variable, leaving `type` categorical, and because
an unused evaluation set paired test features with training labels. An audited
exact-sample replay fixed the dataframe assignment and removed only that invalid
unused evaluation set. The replay charged **$0.00** and produced a valid native
score of **3.98269** (lower is better). Its locally hash-verified artifacts are
under
`outputs/e9_mle_bench/modal_streaming/champs-scalar-coupling-e5b128918511/`.
The initial paid sample cost **$0.004531905**. The persistent campaign now has
$5.77519764 charged and $4.21640146 fail-closed available after the stale
reservation.

`billion-word-imputation` then reached a terminal ungraded attempt. Its paid
sample cost **$0.00650928**, but the bridge response exceeded the output limit:
the saved artifact begins with an unclosed Python fence and ends mid-statement,
so no complete exact program exists for a safe replay. The failed solution and
receipt are preserved under
`outputs/e9_mle_bench/modal_streaming/billion-word-imputation-29d0cdf64322/`.
No resample was made. The persistent campaign now has $5.78170692 charged and
$4.20989218 fail-closed available after the stale reservation.

The twenty-seventh successfully graded competition, `AI4Code`, completed on its
first attempt with native score **0.60314** (higher is better), a valid
submission, and a Tinker charge of **$0.011663745**. Its locally hash-verified
artifacts are under
`outputs/e9_mle_bench/modal_streaming/AI4Code-cf0179ecc807/`. The persistent
campaign now has $5.793370665 charged and $4.198228435 fail-closed available
after the stale reservation.

`jigsaw-unintended-bias-in-toxicity-classification` then reached a terminal
ungraded attempt. Its paid sample cost **$0.00562875** and produced a complete
classification program, but the sampled program passed the competition's
continuous toxicity target directly to `LogisticRegression`. An audited exact
replay thresholded only that target at 0.5, charged **$0.00**, and preserved the
sampled features, model, solver, and hyperparameters. The unchanged repaired
program exceeded both the original 900-second process limit and one final
bounded 1,800-second process limit, so it produced no submission or native
grade. The paid failure and both exact-replay receipts are preserved under
`outputs/e9_mle_bench/modal_streaming/jigsaw-unintended-bias-in-toxicity-classification-13ef8b898e03/`,
`outputs/e9_mle_bench/modal_streaming/jigsaw-unintended-bias-in-toxicity-classification-4d7abbb3ba89/`,
and
`outputs/e9_mle_bench/modal_streaming/jigsaw-unintended-bias-in-toxicity-classification-2b5aec66fae2/`.
No resample or further replay was made. The persistent campaign now has
$5.798999415 charged and $4.192599685 fail-closed available after the stale
reservation.

The twenty-eighth successfully graded competition,
`uw-madison-gi-tract-image-segmentation`, completed with native score
**0.24608** (higher is better) and a valid submission. Its initial paid sample
cost **$0.0055551** and exposed two deterministic path/parsing defects: the
training scan glob omitted the official nested case/day directory while a
NumPy feature vector was unpacked as a mapping, and the later test loop assumed
each underscore-delimited ID had exactly three parts. Two audited exact replays
fixed only those defects, each charged **$0.00**, and preserved the sampled
features, model, solver, and hyperparameters. The paid failure, intermediate
replay, and final hash-verified native-grade receipt are preserved under
`outputs/e9_mle_bench/modal_streaming/uw-madison-gi-tract-image-segmentation-30a339f2e177/`,
`outputs/e9_mle_bench/modal_streaming/uw-madison-gi-tract-image-segmentation-3b792bfe9102/`,
and
`outputs/e9_mle_bench/modal_streaming/uw-madison-gi-tract-image-segmentation-b114091c0a03/`.
No resample was made. The persistent campaign now has $5.804554515 charged and
$4.187044585 fail-closed available after the stale reservation.

The twenty-ninth successfully graded competition, `stanford-covid-vaccine`,
completed with native score **0.42804** (lower is better), a valid submission,
and no above-median or medal claim. Its initial paid sample cost **$0.00489615**
and exposed a deterministic per-position feature/target expansion defect. The
first exact replay fixed that defect but produced only the first 68 test
positions; the second synchronized full-sequence test features but not the
`id_seqpos` loop. A final audited exact replay synchronized both loops to each
record's `seq_length`, charged **$0.00**, and preserved the sampled features,
models, solvers, and hyperparameters. The paid failure, intermediate invalid and
execution-failure receipts, and final locally hash-verified native-grade receipt
are preserved under
`outputs/e9_mle_bench/modal_streaming/stanford-covid-vaccine-b9529b489199/`,
`outputs/e9_mle_bench/modal_streaming/stanford-covid-vaccine-557b219238eb/`,
`outputs/e9_mle_bench/modal_streaming/stanford-covid-vaccine-f8ad6f6ba610/`,
and
`outputs/e9_mle_bench/modal_streaming/stanford-covid-vaccine-7b4c2e4562e4/`.
No resample was made. The persistent campaign now has $5.809450665 charged and
$4.182148435 fail-closed available after the stale reservation.

`facebook-recruiting-iii-keyword-extraction` then reached a terminal ungraded
attempt. Its paid sample cost **$0.002324535** and produced a complete program,
but that program loaded the full multi-gigabyte train and test tables, built a
100,000-feature TF-IDF matrix, binarized the entire tag vocabulary, and trained
one SGD classifier per tag. It exceeded the 900-second process limit without a
submission. Replacing that approach would be an algorithm change rather than a
deterministic defect repair, so no exact replay or resample was made. The failed
solution and locally hash-verified receipt are preserved under
`outputs/e9_mle_bench/modal_streaming/facebook-recruiting-iii-keyword-extraction-72acd3275ce6/`.
The persistent campaign now has $5.8117752 charged and $4.1798239 fail-closed
available after the stale reservation.

The thirtieth successfully graded competition,
`tensorflow-speech-recognition-challenge`, completed on its first attempt with
native score **0.08018** (higher is better), a valid submission, and no
above-median or medal claim. Its Tinker charge was **$0.004966965**. The locally
hash-verified artifacts are preserved under
`outputs/e9_mle_bench/modal_streaming/tensorflow-speech-recognition-challenge-a4c45d403e3f/`.
The persistent campaign now has $5.816742165 charged and $4.174856935
fail-closed available after the stale reservation.

`kuzushiji-recognition` then reached a terminal ungraded attempt. Its paid
sample cost **$0.00820842** and initially failed because the execution image did
not provide OpenCV. Three audited exact replays reused that saved program and
charged **$0.00** each: they supplied pinned OpenCV, removed one unlabeled
full-image feature append, sourced test IDs from `sample_submission.csv`, and
matched the native preparer's `.jpg` ZIP members. The final replay reached real
image loading and model execution but exceeded the full 1,800-second process
limit without producing a submission. Further progress would require changing
the sampled algorithm, so no resample was made. The paid failure and all replay
receipts are preserved under
`outputs/e9_mle_bench/modal_streaming/kuzushiji-recognition-ca53dcc29f54/`,
`outputs/e9_mle_bench/modal_streaming/kuzushiji-recognition-72bbaeb8c306/`,
`outputs/e9_mle_bench/modal_streaming/kuzushiji-recognition-f06baafdbb53/`, and
`outputs/e9_mle_bench/modal_streaming/kuzushiji-recognition-d3435b28fdb8/`.
Their artifact and canonical receipt hashes were verified locally. The
persistent campaign now has $5.824950585 charged and $4.166648515 fail-closed
available after the stale reservation.

`nfl-player-contact-detection` then reached a terminal ungraded attempt. Its
paid sample cost **$0.005235645** and produced a complete program, but feature
extraction performed full tracking-table boolean scans separately for every
contact row. It exceeded the 900-second process limit without a submission.
Vectorizing or replacing that feature pipeline would change the sampled
algorithm rather than repair a deterministic environment mismatch, so no exact
replay or resample was made. The solution and locally hash-verified receipt are
preserved under
`outputs/e9_mle_bench/modal_streaming/nfl-player-contact-detection-caef191fd2da/`.
The persistent campaign now has $5.83018623 charged and $4.16141287
fail-closed available after the stale reservation.

The thirty-first successfully graded competition,
`new-york-city-taxi-fare-prediction`, completed with native RMSE **4.58718**
(lower is better), a valid submission, and no above-median or medal claim. Its
paid sample cost **$0.00588285** and exposed an exact prepared-data filename
mismatch: the native MLE-bench preparer writes the training split as
`labels.csv`, while the saved program only discovered names containing
`train`. A competition-fingerprinted repair added `labels.csv` to that existing
discovery expression without changing features, model, hyperparameters, or
row selection. The exact saved-program replay charged **$0.00** and produced
the native grade. The failed and successful receipts are preserved under
`outputs/e9_mle_bench/modal_streaming/new-york-city-taxi-fare-prediction-6a1dcfa9b83b/`
and
`outputs/e9_mle_bench/modal_streaming/new-york-city-taxi-fare-prediction-860134c097ea/`.
All solution, submission, native-grade, and canonical receipt hashes were
verified locally. No resample was made. The persistent campaign now has
$5.83606908 charged and $4.15553002 fail-closed available after the stale
reservation.

The thirty-second successfully graded competition,
`cassava-leaf-disease-classification`, completed on its first attempt with
native accuracy **0.62145** (higher is better), a valid submission, and no
above-median or medal claim. Its Tinker charge was **$0.003768285**. The
solution, submission, native-grade, and canonical receipt hashes were verified
locally and are preserved under
`outputs/e9_mle_bench/modal_streaming/cassava-leaf-disease-classification-6b09d324d903/`.
The persistent campaign now has $5.839837365 charged and $4.151761735
fail-closed available after the stale reservation.

The thirty-third successfully graded competition,
`histopathologic-cancer-detection`, completed on its first attempt with native
ROC AUC **0.5012** (higher is better), a valid submission, and no above-median
or medal claim. Its Tinker charge was **$0.0040827**. The solution, submission,
native-grade, and canonical receipt hashes were verified locally and are
preserved under
`outputs/e9_mle_bench/modal_streaming/histopathologic-cancer-detection-bcc92aa8e4cf/`.
The persistent campaign now has $5.843920065 charged and $4.147679035
fail-closed available after the stale reservation.

The thirty-fourth successfully graded competition,
`bms-molecular-translation`, completed after native preparation was unblocked.
Two preparation-only attempts hit 1,200- and 1,800-second subprocess limits
before any bridge call, so they produced no solution or receipt and charged
**$0.00** Tinker. Extending the infrastructure-only preparation allowance to
3,600 seconds, together with a 7,200-second Modal function cap, allowed the
unchanged official preparer to complete. The first and only paid sample then
yielded native Levenshtein distance **83.24016** (lower is better), a valid
submission, and no above-median or medal claim. Its Tinker charge was
**$0.00307953**. The solution, submission, native-grade, and canonical receipt
hashes were verified locally and are preserved under
`outputs/e9_mle_bench/modal_streaming/bms-molecular-translation-7ffcd9d6180c/`.
No resample was made. The persistent campaign now has $5.846999595 charged and
$4.144599505 fail-closed available after the stale reservation.

The thirty-fifth successfully graded competition,
`aptos2019-blindness-detection`, completed after an exact saved-program replay.
The paid sample cost **$0.005580135** and selected the same first CSV for both
training and test rows, producing 3,295 predictions for the prepared 367-row
test split. A competition-fingerprinted repair selected the official prepared
`train.csv` and `test.csv` separately, without changing features, model,
hyperparameters, or prediction logic. The exact replay charged **$0.00** and
produced native quadratic weighted kappa **0.0** (higher is better), a valid
submission, and no above-median or medal claim. The failed and successful
receipts are preserved under
`outputs/e9_mle_bench/modal_streaming/aptos2019-blindness-detection-98c605855240/`
and
`outputs/e9_mle_bench/modal_streaming/aptos2019-blindness-detection-dd169d6e3ae1/`.
All solution, submission, native-grade, and canonical receipt hashes were
verified locally. No resample was made. The persistent campaign now has
$5.85257973 charged and $4.13901937 fail-closed available after the stale
reservation.

This resolves the prior “no production model submission” blocker for thirty-five
competitions. E9 remains incomplete because **40 competitions still lack native
grades** and the training-corpus contamination/disjointness receipt is still
absent. The full-suite score remains **`null`**.

Date: 2026-08-09. Status: **PARTIAL** — the runner, the split/verifier binding,
the real competition split and the grading harness (host and container) all
execute. **The harness reproduces upstream's recorded reference score exactly.**
Remaining blockers are the canonical agent image, a model submission artifact,
and a contamination receipt. Suite score is `null` and no model was run.

Receipt: `outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json`

## What now runs

**Venv rebuilt 2026-08-09** after it was deleted in a disk emergency:
`uv venv --python 3.11 outputs/_setup/venvs/e9` then
`uv pip install -e outputs/e9_mle_bench/mle-bench-source`. Cold cache, ~2 min,
1597 MB. `mlebench --help` works; Python 3.11.15, sklearn 1.9.0, pandas 3.0.5,
diskcache 5.6.3 — same stack as the first pass, and the 41 pre-existing tests
plus the new ones still pass against it.

```bash
V=outputs/_setup/venvs/e9/bin

# 1. survey all 75 competitions by recorded download size
$V/python zvf-program/flagship/mle_bench_eval.py survey --top 10

# 2. prove Kaggle rules are accepted (download endpoint; exit 0 == accepted)
$V/python zvf-program/flagship/mle_bench_eval.py check-rules

# 3. prepare the real split (1.8 MB, ~4 s, verifies two checksum manifests)
cd outputs/e9_mle_bench/mle-bench-source && $V/mlebench prepare \
    -c spooky-author-identification --data-dir ../data && cd -

# 3b. drive the official grader against the REAL split
$V/python zvf-program/flagship/mle_bench_eval.py harness-validate \
    --out outputs/e9_mle_bench/evidence/harness_validation.json

# 4. fail-closed receipt
$V/python zvf-program/flagship/mle_bench_eval.py receipt \
    --harness-json outputs/e9_mle_bench/evidence/harness_validation.json \
    --rule-probe-json outputs/e9_mle_bench/evidence/kaggle_rules_check.json \
    --observed-at 2026-08-09 --out outputs/e9_mle_bench/e9_mle_bench_receipt_2026-08-09.json

# 5. unit tests — 53 pass
PYTHONPATH=zvf-program $V/python -m unittest -q flagship.test_mle_bench_eval

# 6. the same grading, inside the container that was built this session
docker run --rm --platform linux/amd64 \
  -v "$PWD/outputs/e9_mle_bench/data:/private/data:ro" \
  --entrypoint /bin/bash mlebench-env:verifier-noheavy -lc \
  '/opt/conda/bin/conda run -n mleb mlebench grade-sample \
     /private/data/spooky-author-identification/prepared/public/sample_submission.csv \
     spooky-author-identification --data-dir /private/data'
```

New files (mine): `zvf-program/flagship/mle_bench_eval.py`,
`zvf-program/flagship/test_mle_bench_eval.py`.

## 1. Smallest competition — measured, not guessed

The upstream repository ships its own size table at
`experiments/competition_categories.csv`
(sha256 `5b6967e944e6b105f54f943cd13042158f49fec5206c0ff536d16d42cf39b634`,
75 rows, one per competition in `experiments/splits/split75.txt`). Ranking every
row by `dataset_size_GB`:

| Rank | Competition | GB | MB | Complexity |
|---|---|---|---|---|
| 1 | **spooky-author-identification** | **0.00190** | **1.95** | Low |
| 2 | detecting-insults-in-social-commentary | 0.00200 | 2.05 | Low |
| 3 | us-patent-phrase-to-phrase-matching | 0.00214 | 2.19 | Medium |
| 4 | random-acts-of-pizza | 0.00300 | 3.07 | Low |
| 5 | tweet-sentiment-extraction | 0.00300 | 3.07 | Medium |

Total across all 75: **3283.69 GB**, matching the README's "3.3TB for the full
set". Independent cross-check via the Kaggle file-listing API for the pick
(`evidence/kaggle_files_spooky_author_identification.csv`):
`sample_submission.zip` 29 KB + `test.zip` 538 KB + `train.zip` 1 MB — the same
order of magnitude at the API's rounded precision.

Full ranking: `outputs/e9_mle_bench/evidence/competition_size_survey.json`.

## 2. `mlebench prepare` — SUCCEEDED against the real split

Rules for `spooky-author-identification` were accepted by a human on 2026-08-09
and verified against the **download** endpoint (`accepted: true`,
`download_endpoint_ok: true`, exit 0 — `evidence/kaggle_rules_check.json`).

```
$ mlebench prepare -c spooky-author-identification --data-dir outputs/e9_mle_bench/data
Downloading spooky-author-identification.zip  1.81M [00:01, 1.18MB/s]
Checksum for `spooky-author-identification.zip` matches the expected checksum.
Preparing the dataset using `prepare` from `.../spooky-author-identification/prepare.py`
Data for competition `spooky-author-identification` prepared successfully.
Checksums for files in `.../spooky-author-identification` match the expected checksums.
```

EXIT=0 in 4 s. **Both checksum gates matched**: the downloaded zip against the
pinned `checksums.yaml`, and then every prepared public/private file against the
same manifest. The local split is therefore bit-identical to the official one —
this is provenance, not just "it ran".

- private answers `prepared/private/test.csv` sha256 `2cf7dc57…`
- sample submission `prepared/public/sample_submission.csv` sha256 `5a9d7015…`

### The false-positive check — keep this, it cost a round trip

An earlier "verified, rules accepted" report was wrong, and the failure mode is
easy to repeat. **`kaggle competitions files` is not an acceptance check.** That
metadata endpoint returns file names and byte sizes for competitions whose rules
have never been accepted; only the *download* endpoint is gated. During the
blocked period the two endpoints disagreed, same account, seconds apart
(`evidence/kaggle_endpoint_contrast.log`):

```
## kaggle competitions files    -> sample_submission.zip,29KB / test.zip,538KB / train.zip,1MB
## kaggle competitions download -> 403 Forbidden - You must accept this competition's rules
```

That listing was byte-identical to one captured before any acceptance attempt,
so it could not have evidenced a state change.

**Use this instead** — it probes the download endpoint and exits 0 only when
bytes are actually served:

```bash
outputs/_setup/venvs/e9/bin/python zvf-program/flagship/mle_bench_eval.py check-rules
```

### The other 74 competitions — a structural finding

Kaggle rule acceptance is **per competition** and non-automatable. Exactly one of
75 is accepted. A full MLE-bench run needs **74 further manual acceptances** by a
signed-in human, each on its own competition page. That is a reproducibility
property of the benchmark itself, not a defect of this lane, and it is recorded
in the receipt under `competition_binding.other_competitions_rule_state`.

The ten smallest competitions were probed during the blocked period; the other
nine remain un-accepted (`evidence/kaggle_rule_acceptance_probe.json`).

## 3. Container — verifier-only image BUILT; canonical `mlebench-env` not built

**Built and working:**

```bash
mkdir outputs/_setup/docker.lock          # mutex taken before the build
cd outputs/e9_mle_bench/mle-bench-source
docker build --platform linux/amd64 --build-arg INSTALL_HEAVY_DEPENDENCIES=false \
  -t mlebench-env:verifier-noheavy -f environment/Dockerfile .
```

- `sha256:59b8e1c643c5b0959f4bdc6a06bb083cf01e3be9dd1d4bf6a744871251a2cc70`
- **2.68 GB**, 18 layers, amd64/linux (qemu emulation on this arm64 host)
- Build time **~811 s of foreground build across two resumed passes**; the first
  pass reached stage 10/18 and the cached resume finished the remaining 8 in
  271 s. (Three earlier detached attempts were killed by the environment's
  background reaper at 72 s and 153 s of build time and produced nothing.)
- Container smoke test: the official CLI runs inside it and the grading-server
  deps import (`evidence/container_smoke.log`).
- **The containerised verifier reproduces the host grading result byte for
  byte** — same score 0.0, same thresholds 0.16506 / 0.26996 / 0.29381 /
  0.418785 (`evidence/container_grade_sample.log`).

**This is not `mlebench-env`.** `INSTALL_HEAVY_DEPENDENCIES=false` skips the
92-line requirements file, `tensorflow[and-cuda]==2.17` and `torch==2.2.0`, so it
contains the grading server and the `mlebench` package but no agent ML stack: it
can grade a submission, it cannot host an agent. The receipt records it under
`environment.verifier_only_variant` with `satisfies_container_gate: false`, and
`container_image_digest_present` stays **failed**.

The canonical build was not attempted, on measured grounds. Wheel sizes from
PyPI, before unpacking and before the other 87 requirements: frameworks
**1.28 GB** (torch 720 MB, tensorflow 573 MB) plus CUDA runtime deps **2.74 GB**
(cudnn 745 MB, cublas 554 MB, …) = **4.0 GB of wheels**, which unpack to roughly
2–2.5×. A 15–25 GB image, built under qemu emulation, on a shared sparse disk
that never shrinks.

### Disk cost — I overshot my estimate, please read

Host free space went **35 GiB → 18 GiB** during this build (the concurrent E1
pull accounts for part of it). My own footprint inside the VM is the 2.68 GB
image plus **9.9 GB of buildx cache** — emulated amd64 layers are expensive. I
estimated 2–3 GB; the true cost was ~12.6 GB. Per the coordinator's note this is
not recoverable on the host.

I did **not** prune: `docker builder prune` is shared and would delete other
lanes' cache. `docker system df` currently reports 9.919 GB of reclaimable build
cache — freeing it would not shrink the host file but would let other lanes
reuse that VM space without growing it further. That is the coordinator's call,
not mine.

## 4. Harness validation — PASS on the REAL split, and it is not a model score

`label: harness_validation`, `is_model_score: false`, `suite_score: null`,
`data_provenance: official_prepared_competition_data`. No model ran. Grading a
gold answer proves the grader; it is not an MLE-bench score.

Everything in the path is upstream: the `multi-class-log-loss` grader, the
medal thresholds, `CompetitionReport`, and the official `mlebench grade-sample`
CLI. The leaderboard is the real Kaggle leaderboard shipped in the repo (1242
teams, sha256 `1087afc6…`). The answers are now the **real prepared split**, not
a fixture.

| Case | Score | Medal | Valid |
|---|---|---|---|
| gold answers (== `private/test.csv`) | **0.0** | gold | yes |
| official `sample_submission.csv` | **1.08468** | none | yes |
| negative control, rows not summing to 1 | `null` | none | **rejected** |

Thresholds: gold 0.16506 / silver 0.26996 / bronze 0.29381 / median 0.418785,
lower-is-better. All seven checks pass.

### Upstream reproduction — exact

Upstream's `tests/constants.py` records `spooky-author-identification: 1.08468`
as the score its own sample submission achieves on the real split. This lane
observed:

```json
{"upstream_expected_score": 1.08468, "observed_score": 1.08468,
 "absolute_delta": 0.0, "relative_delta": 0.0, "matches_upstream": true}
```

**Delta 0.0.** Scores are rounded to 5 decimals by `grade_helpers`, so exact
equality at that precision is the correct bar and it is met. This is a
substantially stronger result than the earlier fixture run, which produced
1.0931 against synthetic labels — I called that a ~1% sanity signal rather than a
claim, and that caution was warranted: the fixture number was never the reference
value, and the real number lands on it exactly.

Reproducing 1.08468 means the download, the upstream preparer's 90/10 split with
`random_state=0`, the log-loss implementation and the leaderboard ranking all
agree with the reference implementation end to end.

The same 1.08468 was then reproduced **inside the built container** with the real
data mounted at `/private/data`, so the verifier is not host-dependent.

Evidence: `evidence/harness_validation.json`,
`evidence/mlebench_grade_sample_real.log`,
`evidence/container_grade_sample.log`.

The negative control is derived from the real sample submission and written to
`outputs/e9_mle_bench/harness_controls/`, deliberately **outside** the prepared
directory so the official data keeps matching its recorded checksums. The old
synthetic fixture in `harness_fixture_data/` is retained only as the
blocked-path artefact; it is no longer what the harness grades.

**Prerequisite discovered earlier:** the repo's 322 Git-LFS files, including
every `leaderboard.csv`, were unresolved pointers. Grading is impossible without
them (`get_leaderboard` would parse a 3-line pointer file). Fixed with
`git lfs install --local && git lfs pull` (39 MB). The runner's
`verifier_identity` now detects an unresolved pointer and fails the verifier gate
rather than grading against garbage.

## 5. Runner and receipt

`zvf-program/flagship/mle_bench_eval.py` — four subcommands (`survey`,
`harness-validate`, `check-rules`, `receipt`), 53 unit tests. The receipt binds:

- **Immutable revision** — `openai/mle-bench@507f92e1138bb6e40dac5c6ee7a6758e6424bf97`
- **Task-ID hashes** — sorted, newline-joined, terminal newline; eval split
  `440ecb5c6a13bc54e0d671e19eb532ee33c8305cca862e4d5b2af49f3d44f85b`
- **Split manifest** — per-file sha256 + count + task-ID hash for split75 and
  low/medium/high (22/38/15). Per-*sample* hashes are explicitly `null`: an
  MLE-bench task ID is a competition ID, and per-row IDs exist only after
  `prepare`.
- **Container digest** — `docker image inspect mlebench-env`, currently absent
- **Verifier identity** — grader name, `grade_fn` dotted path, sha256 of
  `grade.py`, `prepare.py`, `checksums.yaml`, the leaderboard, and
  `grading_server.py`

Fail-closed: `status` is `READY` only if all eight gates pass; a missing gate key
counts as a failure. Current state:

| Gate | |
|---|---|
| upstream_revision_pinned | pass |
| split_manifest_resolved | pass |
| verifier_identity_resolved | pass |
| dataset_license_accepted | **pass** — rules accepted 2026-08-09, verified on the download endpoint |
| competition_data_prepared | **pass** — real split, both checksum gates matched |
| container_image_digest_present | **fail** — verifier-only variant built, canonical `mlebench-env` not |
| model_submission_artifact_present | **fail** |
| contamination_disjointness_receipt | **fail** |

→ `status: BLOCKED`, `score: null`, `is_model_score: false`. Two gates flipped
this pass; three remain.

## 6. Licence position (honest)

- **Repository code: MIT.** `LICENSE` at the pinned revision, sha256
  `8a44e3d5…`.
- **The 75 competition datasets are NOT covered by it.** The licence explicitly
  excludes external datasets downloaded while using the package. Each
  competition carries its own Kaggle competition rules, accepted per competition
  by a signed-in human. An agent cannot accept them, and no blanket acceptance
  exists.
- The `leaderboard.csv` files *are* redistributed in the repository under MIT and
  are what the medal thresholds come from — so the verifier side is licence-clean
  even while the data side is not.
- Recorded in the receipt as `license_position` with
  `acceptance_is_automatable: false`.

## Single next action

Three gates remain, and none is a download:

1. **`container_image_digest_present`** — needs the canonical `mlebench-env`
   (15–25 GB, hours under qemu emulation). Deliberately not built; the
   verifier-only variant does not and should not satisfy this gate.
2. **`model_submission_artifact_present`** — needs an agent run inside that
   container with a model API key. Outside this lane's cost boundary.
3. **`contamination_disjointness_receipt`** — needs a training-corpus task-ID
   manifest to diff against the eval-split hash
   `440ecb5c6a13bc54e0d671e19eb532ee33c8305cca862e4d5b2af49f3d44f85b`. This is a
   paperwork gate, not a compute one, and it is the cheapest of the three.

Scaling beyond this one competition is gated on **74 further human rule
acceptances**, one per competition page — see section 2.
