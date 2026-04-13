# ACM Submission Checklist

Pre-submission checklist for ACM venues (KDD, SIGIR, CIKM, WWW, CHI, etc.).  
Based on [ACM Author Submission Guidelines](https://www.acm.org/publications/authors/submissions) and [ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current).

---

## 1. Formatting & Template

- [x] **Document class**: `\documentclass[manuscript,review,anonymous]{acmart}` for submission
- [x] **Camera-ready class**: `\documentclass[sigconf]{acmart}` (switch after acceptance)
- [x] **Template version**: acmart v2.16 (August 2025)
- [x] **Single-column**: Submission uses single-column `manuscript` format
- [x] **Fonts**: Using default libertine font set (no substitutions)
- [x] **Margins**: No manual margin adjustments
- [ ] **Page limit**: Check venue-specific page limits (e.g., KDD: 8 pages + refs)
- [x] **Bibliography style**: `ACM-Reference-Format` (not `plainnat`)

## 2. ACM CCS Concepts (Required)

- [x] **CCS codes included**: Generated from [ACM CCS Tool](https://dl.acm.org/ccs)
- [x] **Primary concepts**:
  - Computing methodologies → Machine learning (500 - high significance)
  - Computing methodologies → Reinforcement learning (500 - high significance)
- [x] **Secondary concepts**:
  - Computing methodologies → Learning paradigms (300)
  - Computing methodologies → Natural language processing (300)
  - General and reference → Evaluation (300)
  - Software and its engineering → Software creation and management (100)
- [x] **XML block**: `\begin{CCSXML}...\end{CCSXML}` included in preamble
- [x] **LaTeX descriptors**: `\ccsdesc[significance]{concept}` commands present

## 3. Keywords (Required)

- [x] **Keywords provided**: reinforcement learning, language models, benchmark, reproducibility, RLHF, GRPO, DPO, post-training
- [x] **Format**: Comma-separated in `\keywords{}` command

## 4. ACM Metadata (Post-Acceptance)

- [ ] **DOI**: `\acmDOI{10.1145/nnnnnnn.nnnnnnn}` (assigned by ACM)
- [ ] **ISBN**: `\acmISBN{978-x-xxxx-xxxx-x/xx/xx}` (assigned by ACM)
- [ ] **Conference**: `\acmConference[Short]{Full Name}{Date}{Location}`
- [ ] **Year**: `\acmYear{2026}`
- [ ] **Copyright**: `\copyrightyear{2026}`
- [ ] **Price**: `\acmPrice{15.00}`
- [ ] **Rights statement**: Added per ACM instructions after acceptance

## 5. References & Citations

- [x] **ACM Reference Format**: Using `ACM-Reference-Format.bst`
- [x] **DOIs included**: All entries with available DOIs include `doi` field
- [x] **Conference locations**: Proceedings entries include `location` field
- [x] **BibTeX format**: Follows [ACM BibTeX guidelines](http://www.acm.org/publications/authors/bibtex-formatting)
- [x] **All references cited**: No orphan bibliography entries
- [x] **Citation style**: Numbered (default) — use `\citestyle{acmauthoryear}` for author-year

## 6. Anonymization (For Double-Blind Venues)

- [x] **Anonymous document class**: `anonymous` option enabled
- [x] **Anonymization script**: `scripts/anonymize.sh` strips identifying info
- [ ] **Self-citations**: Phrased in third person ("Previous work [X]..." not "Our prior work [X]...")
- [ ] **Acknowledgments**: Removed for submission (add back for camera-ready)
- [ ] **Supplementary**: No identifying information in appendices or code links

## 7. ACM Artifact Badges

### Artifacts Available
- [x] Code permanently available on GitHub (public repository)
- [x] Model checkpoints on Hugging Face Hub
- [ ] Zenodo DOI for archival snapshot (recommended)
- [x] Not on personal web pages (uses institutional/commercial repositories)

### Artifacts Evaluated — Functional
- [x] **Documented**: README.md, REPRODUCE.md, ARTIFACT.md, inline comments
- [x] **Consistent**: Artifacts match paper claims and generate reported results
- [x] **Complete**: All 11 implementations, all analysis scripts, all configs
- [x] **Exercisable**: Docker + scripts run end-to-end; all scripts executable

### Artifacts Evaluated — Reusable
- [x] **Well-structured**: Modular directory layout with clear separation
- [x] **Quality documentation**: Model cards, statistical methodology, CCS codes
- [x] **Community standards**: Apache 2.0 license, HF model card spec, pip-installable
- [x] **Extensible**: New RL libraries follow documented implementation template

### Results Validated
- [ ] **Results Replicated**: Awaiting independent replication using author artifacts
- [ ] **Results Reproduced**: Awaiting independent reproduction without author artifacts
- [ ] **Peer-reviewed replication paper**: Required for badge (post-publication)

## 8. Reproducibility Requirements

- [x] **Source code**: Complete implementation for all experiments
- [x] **Pinned dependencies**: Exact versions in `requirements.txt`
- [x] **Docker environment**: `Dockerfile` with CUDA 12.4 + Python 3.10
- [x] **Seed management**: Deterministic seeding across all frameworks (`utils/seed.py`)
- [x] **Multi-seed evaluation**: 5 seeds per experiment with statistical testing
- [x] **Compute documentation**: GPU types, hours, costs in `COMPUTE.md`
- [x] **Statistical analysis**: rliable metrics, bootstrap CIs, significance tests
- [x] **Reproduction commands**: Step-by-step in `REPRODUCE.md`

## 9. TAPS Submission (Post-Acceptance)

After acceptance, prepare for [The ACM Publishing System (TAPS)](https://www.acm.org/publications/taps-production-workflow):

- [ ] Switch to `\documentclass[sigconf]{acmart}` (two-column)
- [ ] Add ACM rights statement and DOI
- [ ] Verify page limits in two-column format
- [ ] Upload source files (LaTeX .zip) to TAPS
- [ ] Review auto-generated PDF and HTML5 proofs
- [ ] Verify CCS concepts render correctly in DL metadata
- [ ] Approve final publication

## 10. Venue-Specific Considerations

### KDD (Datasets & Benchmarks Track)
- Page limit: 8 pages + references + optional appendix
- Abstract deadline: Feb 1, 2026 / Paper: Feb 8, 2026 (Cycle 2)
- Notification: May 16, 2026
- Submission via: [KDD 2026 portal](https://kdd2026.kdd.org/)

### ACM Computing Surveys (CSUR)
- No page limit (survey format)
- Continuous submission
- Uses `acmsmall` template variant

### SIGIR (Reproducibility Track)
- Page limit: 4–8 pages + references
- Focus on replication studies

---

_Checklist based on [ACM Author Guidelines](https://www.acm.org/publications/authors/submissions), [ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current), and venue-specific CFPs._
