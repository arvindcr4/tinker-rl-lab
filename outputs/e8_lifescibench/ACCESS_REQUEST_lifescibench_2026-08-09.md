# Access request — LifeSciBench evaluation materials

Prepared 2026-08-09. Send as-is; fill the bracketed fields before sending.

---

## Routing note (read before sending)

**OpenAI has published no request route for LifeSciBench evaluation materials.**
The announcement page's "Get involved" section offers exactly two forms, and
neither one requests benchmark artifacts:

| Link on the page | Actual destination | What it is for |
|---|---|---|
| "Join as a contributor" | `https://openai.com/form/life-science-contributors/` | Recruiting Ph.D. life scientists to author tasks for **future** benchmarks. Fields: name, email, phone, education, background, LinkedIn. |
| "Request access" | `https://openai.com/form/life-sciences-access/` | **GPT-Rosalind research-preview model access** — not benchmark data. Requires legal entity name, OpenAI Organization ID, country list, and a government-ID verification step. |

The preprint lists no corresponding-author address and no data-availability
contact. Appendix A.5 is the only statement on release, and it is a
restriction, not a route.

Ranked routes, best first:

1. **`https://openai.com/contact-sales/`** — the only OpenAI channel that
   accepts an institutional inbound with free-text scope and routes to a human.
   Paste the request below into the message field.
2. **`https://openai.com/form/life-sciences-access/`** — only if the
   institution independently wants GPT-Rosalind access. The free-text field
   "How do you intend to use GPT-Rosalind?" can carry a one-line pointer to
   this request. Do **not** submit it solely to reach the benchmark team; it
   triggers government-ID verification and is scoped to model access.
3. **Named authors via institutional contact** — the paper's author list is
   OpenAI (17 authors) and Tacit Labs (Anne Marie Droste, Katie-Rose Skelly,
   Max Marion, Nicole Fitzgerald). Tacit Labs is the smaller, more reachable
   party and co-built the benchmark.

Expect this to be declined or deferred. Appendix A.5 states release "may be
limited by licensing, privacy, proprietary information, or biological safety
considerations," and that content was excluded or restricted where
dissemination "could create biological safety risks." A refusal is a valid,
citable outcome for the campaign — it converts an open blocker into a closed
one.

---

## Request text

**Subject:** LifeSciBench — request for evaluation materials for an independent
replication run

To the LifeSciBench team,

I am writing about **LifeSciBench**, introduced 2026-06-17 at
`https://openai.com/index/introducing-life-sci-bench/`, with the preprint at
`https://cdn.openai.com/pdf/b4299379-0a97-4ffa-8b9b-c3fbb299caa9/lifescibench_preprint.pdf`.

[ONE OR TWO SENTENCES: who you are, your institution, and what you are
evaluating. Example: "I am a researcher at [institution] running a
reproducibility study across [N] agentic-evaluation suites; LifeSciBench is the
primary life-sciences suite in that portfolio."]

We do not intend to redistribute any material. Our harness is already built and
validated against the published protocol; it fails closed and refuses to emit a
score until each artifact below is pinned to an immutable revision. We are
requesting the six artifacts below. If any cannot be released in full, a
restricted or held-server form is useful to us — see the fallbacks.

### 1. The evaluation task package

The **750 expert-authored LifeSciBench tasks**, each in the form described in
§3.3 of the preprint: the expert-written prompt, the supporting artifacts or
contextual evidence, and the task-specific grading rubric. This includes the
**1,062 task artifacts** (figures, PDFs, tables, sequence files, structure or
chemical files) and the **37 tasks carrying prompt-provided URLs**.

*Needed for:* running the benchmark at all. Nothing substitutes for it — we
have explicitly rejected LAB-Bench, BixBench, SciBench, ScienceWorld and every
other life-sciences benchmark as non-substitutes.

*Fallback if full release is restricted:* a public subset of any size, with the
per-task workflow and biological-domain labels retained, so results can be
reported against a named, citable slice rather than the full 750.

### 2. Licence terms or an access receipt

The **licence or terms of use** governing the evaluation materials, or a dated
written access grant naming the recipient and the permitted use.

*Needed for:* our receipt schema requires an explicitly approved licence
identifier. Preprint appendix A.5 states public release "may be limited by
licensing, privacy, proprietary information, or biological safety
considerations" but does not name the licence that applies to material actually
released. We cannot record `license_status: approved` against an unnamed
licence.

*Fallback:* a one-line statement of the terms under which a named recipient may
run the benchmark internally without redistribution.

### 3. The evaluation interface, at an immutable revision

The **evaluation interface** referred to in §5.1 — the harness that presents
each task's question and context to the model and, for artifact-bearing tasks,
"gives the model access to the relevant files." Please include the container
digest or commit SHA, and confirm the **single-turn** protocol and the
**unrestricted-Internet-browsing** setting used for the published results.

*Needed for:* artifact handling is the single largest reported performance
factor (pass rate drops from 45.1% text-only to 28.1% with artifacts or URLs
for GPT-Rosalind). A locally reimplemented file-presentation layer would not be
comparable to your published numbers, so the interface has to be the one you
ran.

*Fallback:* a written specification of how artifacts are presented (file paths,
MIME handling, size limits, browsing configuration), sufficient to reimplement
faithfully, plus a statement that a reimplementation is acceptable for
comparison.

### 4. The rubric grader, at an immutable revision

The **grader** described in §5.1 and appendix A.3 — the component that
"evaluates each rubric criterion independently and assigns points according to
the task-specific scoring scheme." Please include its revision, and where
grading is automated or model-assisted (per A.3), the grader model identity and
version. Please also confirm the two metric definitions from §5.2:
**normalized rubric score** (awarded points ÷ total possible points, averaged
problem-weighted) and **task pass rate** (fraction of tasks at or above the
**70%** task-specific threshold).

*Needed for:* rubric grading is the benchmark. A locally written grader would
produce a different number under the same name, which is the specific failure
mode we are trying to avoid.

*Fallback:* a held grading service we submit responses to, returning only the
two aggregate metrics. This keeps rubric text unreleased and is our preferred
compromise if rubric confidentiality is the blocker.

### 5. An immutable task manifest

The **ordered list of the 750 task identifiers** at a stated package revision,
with each task's **workflow category** (one of: evidence handling; analysis;
design and optimization; scientific reasoning; validation and operations;
translation; scientific communication) and **biological domain** (one of:
Genomics; Chemistry / MedChem; Protein + Structural Biology; Molecular + Cell
Biology; Assays + Screening; Bioinformatics / Comp Bio; Clinical /
Translational Science).

*Needed for:* we hash every task identifier against the package revision and
seal the ordered manifest, so that a published result names the exact task set
it was measured on. Without it we cannot prove we ran your 750 rather than some
subset.

*Fallback:* the identifier list and stratification labels alone, with no task
content. This is useful to us even if items 1–4 are all declined, because it
lets us state precisely what we did not run.

### 6. A train/evaluation disjointness statement

A statement of whether the LifeSciBench tasks are **disjoint from the training
data** of the evaluated models, and any **contamination or decontamination
controls** applied during construction or release review. If a train or
development split exists, its manifest hash.

*Needed for:* we can only report a result as held-out if disjointness is
attested by the provider. We cannot infer it.

*Note on why we are asking:* the preprint contains no mention of
contamination, decontamination, leakage, or held-out controls anywhere in its
text — we checked. Appendix A.4 discloses that LifeSciBench was developed by
OpenAI and that the evaluated systems include OpenAI models, which makes the
disjointness question load-bearing for anyone reading the published
comparison. A short written statement would close it.

*Fallback:* a statement that no contamination analysis was performed. That is
equally usable — we would report the result as *not* proven held-out, which is
accurate and citable.

---

### What we will do with it

Report the two published metrics on the exact task set, with the package
revision, interface revision, grader revision, and sealed manifest hash
recorded in the result receipt. We will not redistribute tasks, artifacts, or
rubrics. We are glad to sign a licence, an NDA, or a use agreement, and to
share our results with you before publication.

[NAME]
[TITLE, INSTITUTION]
[EMAIL]

---

## Appendix: mapping to our internal blocker codes

| Item | Blocker code | Preflight error it clears |
|---|---|---|
| 1. Task package | `E8_DATASET_REVISION` | `dataset.revision must be an immutable 40-hex or sha256: revision` |
| 2. Licence / access receipt | `E8_LICENSE_RECEIPT` | `dataset license must be explicitly approved`; `dataset.license_id must be pinned, not UNPINNED_REQUIRED` |
| 3. Evaluation interface | `E8_NATIVE_ENVIRONMENT_REVISION` | `native_environment.revision must be an immutable 40-hex or sha256: revision` |
| 4. Rubric grader | `E8_NATIVE_VERIFIER_REVISION` | `native_verifier.revision must be an immutable 40-hex or sha256: revision` |
| 5. Task manifest | `E8_TASK_MANIFEST` | `task_manifest must contain immutable evaluation task rows` |
| 6. Disjointness statement | `E8_HELDOUT_DISJOINTNESS` | `train_split_manifest_hash must be a lowercase SHA-256 digest` |
