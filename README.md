# TAVERN

**Temporal Anchoring for Version Consolidation**

*A framework for Abstractive Narrative Consolidation from multiple overlapping accounts.*

**Narrative Consolidation** is a Natural Language Processing task formulated in
this doctoral work: given several long narrative accounts of the same events,
produce one continuous account that is chronologically correct, carries every
detail any source contributes, and states shared material once. It is not
multi-document summarization — the objective is unification rather than
compression, chronology is constitutive rather than secondary, and the output may
be longer than any source. The definition also excludes extraction: no system
restricted to verbatim spans can carry every detail *and* avoid repeating what
the accounts share, so abstraction is what remains once the objectives are stated.

**TAVERN** is the framework that performs the task end to end, in six stages:

| # | Stage | What it contributes |
|---|---|---|
| 1 | Pre-processing | documents, verses, the pericope layer |
| 2 | Temporal annotation | ISO 24617-1:2012 (ISO-TimeML) events, times, links |
| 3 | Anchoring and alignment | the induced timeline, event clusters |
| 4 | Relational graph | the TAEG, typed inter- and intra-document relations |
| 5 | Guided abstractive generation | one fused version per event, in order |
| 6 | Evaluation | held-out chronology, reference consolidation, metrics |

The framework's governing design principle is the separation of **temporal
planning** from **surface realisation**: the chronology of the output is settled
symbolically, before any text is generated.

Three works in this doctorate share the task and the sources. **TAEG** (JBCS)
formulates Narrative Consolidation and solves version selection extractively.
**NSNC** (IJCNN) generalises the selection to neuro-symbolic abstractive fusion.
Both take an external, ready-made chronology (Aschmann's harmony of the Gospels)
as an input to alignment. This repository implements the stage that removes that
dependency: the temporal backbone is **induced** from an ISO-TimeML annotation of
the sources, and the harmony is held out and read only at evaluation.

Annotation is one stage of six. It is the stage newly implemented here, which is
why most of the code added in this repository lives in Stage 2 and Stage 3 — not
because it is the framework's centre.

Case study: the four canonical Gospels over the Passion Week.

> PhD thesis — Roger Antonio Finger, UNISINOS
> *TAVERN: Temporal Anchoring for Version Consolidation — A Framework for
> Abstractive Narrative Consolidation from Multiple Overlapping Accounts*
> Advisor: Prof. Dr. Gabriel de Oliveira Ramos

---

## Why this repository changed

Earlier revisions of TAVERN itself — not TAEG or NSNC, which never claimed
otherwise — also took the Aschmann chronology as an input to alignment. That
made Kendall's τ — the ordering metric — **1.000 by construction** in every
configuration ever reported, because the system was never asked to exercise the
competence the metric measures. The chronology is now reachable only from Stage 6, enforced in code:

```python
# tavern/config.py
def assert_no_chronology_import() -> None:
    """Raises if a chronology load originates in stages 1–5."""
```

τ becomes a measurement. It is **0.9274**. Against a null system that uses
**no annotation at all** — N1, positional interleaving of the four documents
in raw verse order — τ is 0.8140. So 0.9274 closes **61.0%** of the distance
between "no annotation" and the curated chronology (1.000). Report both
numbers together; τ alone hides that the useful range is only 0.186 wide.

This is the `ancoragem` configuration (commit-tagged `ancoragem-20260831`),
the primary configuration this README reports. The earlier `canonical-20260827`
run (τ = 0.9155, 54.6%) is retained throughout as the **"before" state**: the
coreference score claimed to use the TIMEX3 inventory's normalisation and the
anchor chains, but that normalisation never actually reached the score, and a
hand-picked two-word stop-list stood in for entity discrimination. Addendum 9
implemented the mechanism the thesis already claimed. Every result below is
reported both ways.

The previous four-stage pipeline is preserved under `legacy/` rather than
deleted.

---

## Architecture

- **Stage 1** — token layer, 91 pericopes, entity chains.
- **Stage 2** — ISO-TimeML Layer A (conformant `<EVENT>`/`<TIMEX3>`/`<SIGNAL>`/
  `<TLINK>`/`<SLINK>`/`<ALINK>`/`<MLINK>`) and Layer B, derived from it (modal
  paths, veridicality partition, closure under Allen's algebra).
- **Stage 3** — event units, a shared day axis, progressive profile alignment
  across documents, a weighted tournament, the typed cross-document event graph.
- **Stage 4** — relational graph attention over the typed graph.
- **Stage 5** — micro-abstractive fusion, one paragraph per candidate
  canonical event, in induced order, each checked against the accounts it
  was built from — keep every detail, invent none, state each once — and
  re-asked or handed to a deterministic union when it fails. The third axis
  compares 5-grams and so catches literal repetition only; the
  paraphrase-robust measure (`fidelity.excess_repetition`) is reported but
  does not gate.
- **Stage 6** — the only place the harmony and the reference may be read.

The wall: the Aschmann chronology and the Golden Sample are reachable only from Stage 6, enforced by `config.assert_no_chronology_import()`.

---

## Results

Measured by `run_experiments.py --all --backbone ollama --backbone-model
gemma4:26b --ollama-repeat-penalty 1.1` on the digest-pinned corpus. Full
numbers in `outputs/<tag>/results.json`; the discussion belongs to the thesis, not here.

| Measure | canonical (before) | **ancoragem (primary)** |
|---|---|---|
| τ · pairwise · coverage | 0.9155 · 0.9577 · 0.8512 (143/168) | **0.9274 · 0.9637 · 0.8869 (149/168)** |
| Clusters | 249 | **289** |
| Purity | 30.8% | **44.4%** |
| B-cubed F1 | 0.498 | **0.519** |
| R-L, extractive (induced order + grouping) | 0.594 | **0.662** |
| R-L, abstractive end-to-end (Ollama) | 0.497 | **0.622** |
| Content coverage (overall / multi-source) | 91.2% / 89.7% | **98.8% / 98.3%** |
| Selection accuracy vs. floor | 0.2973 < 0.3446 (74) | **0.3600 > 0.3411 (75)** |
| Errors (not-detected/not-aligned/over/under/transposed/day-boundary/leaked) | 0/21/25/87/31/1/0 | **0/20/23/91/29/2/0** |

The abstractive row is the only one the Stage 5 fusion work moved (R-L 0.566
→ 0.604 with the fixes, → 0.622 on gemma4:26b; content coverage 93.3% →
98.8%). Everything else in this table is
Stage 3 or Stage 4. τ, pairwise, coverage, clusters, purity and B-cubed were
re-measured on this artefact and reproduce to the digit; selection accuracy and the
error-taxonomy line are carried over from the run before the Stage 5 work,
because re-measuring them requires `--downstream`, which would overwrite the frozen artefact and break the
digest the human evaluation is tied to. The point stands either way: the
faithfulness of the fusion and the quality of the induced chronology are
separable, and here they were separated by measurement rather than by
argument. `ancoragem` keeps its name across that change; the pre-fix
artefacts are recoverable from `ancoragem-20260831`.

**What these numbers do not say:**

- τ has a floor: N1 (no annotation at all) gives 0.8140 at K = 169 windows, the
  curated event count — a granularity target supplied from outside, not tuned
  against N1's own τ; at K = 289 it gives 0.8259. ancoragem closes 61.0% of the
  range to the ceiling.
- τ is protected by construction: `removed_arcs` is 0 with the induced
  grouping, 614 with the curated one, and τ falls to 0.6296 under that grouping.
- Purity is 44.4% against an 89.5% ceiling, not against 100%.
- 26 verses are claimed by two curated events each; 4 events are invisible
  to any verse-indexed instrument, which is why `gold_clusters` reads 165.
- Selection accuracy is 0.3600 against a floor of 0.3411 **over the same
  subset** — the floors over 95 and over 75 events are not interchangeable.
- Stage 3's two halves do not separate: fixing either alone **makes the system
  worse** (B = 0.4916, C = 0.4794, against A = 0.6620).
- The architecture's own ceiling is R-L 0.7948, and the pre-registered bar
  was 0.7954 — set above the achievable maximum.
- `lexical_baseline` wins: recall@1 0.5126 against 0.4369 for the annotated score.
- The absolute-day projection is 38% populated (42/112 day, 46/112 part).
- Three of the five ablations improve some headline metric, because the
  reference covers only 87.5% of the sources' vocabulary and restricting
  output is rewarded.
- The pre-registered criterion is **not met**: 0.7605 against 0.7954.

### Two corrections to the prior work

The "prior work" is TAEG (Algorithm 1, the JBCS baseline,
`tavern/baselines/__init__.py`, [github.com/neemias8/TAEG](https://github.com/neemias8/TAEG)).
Against the reference's own event numbering (not text similarity), TAEG's
reported centrality-based selection accuracy of 0.489 becomes **0.337** — the
analytical floor; only its length heuristic (0.632) beats chance. Event #28
cites Luke 13:34–35, outside the Luke 19–24 scope, so 363 citations resolve
to 362 versions, which shifts the version distribution and the floor.

The published degradation curve reproduces to within 0.02 of ROUGE-L at
every level, which is what makes these comparison points usable.

---

## What the measurements show, and what they cannot

Four cautions belong next to the tables above. They are not disclaimers; each is a measured result.

**Kendall's τ has a floor here, and it is high.** A null model that reads no annotation at all — it cuts each document into positional windows and interleaves them by position — reaches τ = 0.8140 on this corpus. The interpretable band is 0.186 wide, not 2.0. A τ of 0.9274 read at face value overstates the contribution by roughly a factor of three. `scripts/null_model_n1.py` reproduces it.

**The architecture protects its own metric.** Progressive profile alignment is monotone by construction, so the ordering graph is acyclic, the feedback arc set never fires, and τ cannot register a whole class of failure. This is not specific to this system: any monotone aligner has the property. The decisive comparison is in the repository — the same ordering machinery removes 0 arcs under the induced clustering, 614 under the curated one, and 4,203 under a non-monotone agglomerative clustering of the same corpus, where τ falls to 0.7812, *below the null model*. Run it with `python run_experiments.py --legacy-agglomerative`.

**Reproducibility is not significance.** A ten-seed sweep of the selection metric gives sd = 0.0042. Like the selection-accuracy row above, this sweep predates the Stage 5 work and was not re-run on the frozen artefact, for the same reason; the argument it makes is about denominators and does not depend on the exact value. The sampling distribution that matters is over the 75 events, not over seeds, and its null standard deviation is 0.0539 — thirteen times larger. Reporting "0.3613 ± 0.0042" would be formally correct and materially misleading. On the honest denominator, z = 0.37 and P(a random selector scores at least as high) = 0.31: the result is reproducible and indistinguishable from chance at the same time. `scripts/seed_sweep_selection.py` and `scripts/selection_significance.py`.

**The pre-registered criterion is reported as failed.** The criterion was set at substitution level — the induced version selection projected into the curated segmentation — and at that level the pipeline reaches 0.7605 under the longest-account rule (0.6655 under TAEG's) against the bar of 0.7954, a bar that, as it turned out, sat above the architecture's own ceiling of 0.7948. End-to-end ROUGE-L is a different quantity and is 0.622; the two are not comparable and must not be read against the same bar. On the same backbone an extractive configuration reaches ROUGE-L 0.662 against fusion's 0.622, while covering less of the sources' content-word vocabulary than the fusion does: 82.3% against 99.1% (the reference consolidation itself covers 87.5%; 1,693 content-word types in the sources, scikit-learn's `ENGLISH_STOP_WORDS` removed, no lemmatization or frequency/length cutoff — `scripts/verify_for_thesis.py`). The extractive figure moves by about a percentage point across independently seeded runs, since Stage 4's GNN selects the verbatim account per cluster; the reference and fusion figures do not move. The extractive configuration does not satisfy the task definition, so the two numbers do not compare two candidate systems; they compare a system that solves the task with one that does not, under a metric that rewards the latter.

**Faithfulness was a code defect; fusion was a capacity limit.** The Stage 5 fixes took the abstractive configuration from 104 of 289 events failing the fusion premise to 11, and 0 events now lose a detail the accounts carry or state material no account carries, against 52 and 49 before. Compression did not move with them: on gemma3:4b the fused paragraph was at the median exactly as long as `UnionFuser`'s output for the same accounts, and by `fidelity.excess_repetition` — which counts how often a fusion restates a content term beyond any single account — the model (0.115) was indistinguishable from that deterministic baseline (0.125). Three redrafted prompts were measured against the production one on the worst-repeating episodes and every one of them bought less repetition with lost detail and invention, so the prompt was not the binding constraint. Changing only the backbone, to gemma4:26b, moved excess repetition to 0.067, halved the events above 0.10 (87 → 63), and cut the fallback rate from 30 of 155 to 4 — 97.4% abstractive by event against 80.7%. `curation.csv`'s `fusion` column states the path per event; `scripts/check_text_quality.py` and `scripts/fusion_prompt_ab.py` are the instruments.

Nothing above is reconstructed after the fact. Each is in the thesis with the run that produced it.

---

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Two dependencies need a note. `rouge-score==0.1.2` fails to build against recent setuptools — install with `--no-build-isolation`, or copy the `rouge_score` package directly into site-packages. `torch` is needed only for Stage 4; without it the pipeline falls back to the unpropagated aggregation.

ROUGE-L over a ~16k-token reference cannot use `rouge_score`'s own longest-common-subsequence table, which is quadratic in pure Python and takes minutes per pair. `content_metrics._lcs_length` therefore uses `pylcs` when it is installed and otherwise a bit-parallel fallback (Crochemore, Iliopoulos, Pinzon and Reid, 2001) that packs a row of the table into one arbitrary-precision integer: 0.3 s on the reference, and `verify_fast_path` asserts either path reproduces `rouge_score` exactly (it does, to 0.0). `pylcs` is optional and commented out in `requirements.txt` because it is a compiled extension without a wheel for every interpreter.

## Run

### Everything at once

```bash
ollama pull gemma4:26b       # once
python run_all.py            # or, on Windows PowerShell:  .\run-all.bat
```

On a fresh clone, check out the branch first — `git clone` leaves you on the repository's default branch, which does not contain any of this:

```powershell
git fetch <remote-or-bundle> main:tavern-thesis-framework
git checkout tavern-thesis-framework
pip install -r requirements.txt
python run_all.py
```

`run_all.py` checks the environment and the corpus digests, measures **both** configurations — extractive, for comparability with the degradation curve, and abstractive, which is what the framework is for — regenerates the curation sheets, and packages everything into one `tavern_results_<stamp>.zip`.

The generation run is 155 model calls — one per multi-source cluster; the 134 single-witness events are emitted verbatim without one, since with a single account there is nothing to fuse — plus one retry for every fusion the faithfulness guard rejects, which was 13 on the reported run, for 168 in total. It is **cached to disk as it goes**, in a single file shared across every tag and config (`outputs/fusion_cache.jsonl`, keyed by backbone + model + `repeat_penalty` + the prompt version + the repair pass + the exact source texts — see `CachedFuser` in `stage5_generation/backbones.py`), so an interrupted run resumes on the same command, and re-running `--ablations` under a different configuration reuses whatever it shares with a prior run instead of regenerating from scratch. The reported run took 51 min of generation on 8 CPU threads with no GPU offload; expect that order, depending on the accelerator and on how much of the cache is already warm.

```bash
python run_all.py --backbone union            # dry run, no model needed
python run_all.py --skip-extractive           # if that run already exists
python run_all.py --backbone primera --model allenai/PRIMERA
```

### One stage at a time

```bash
# stages 1–5 only. Reads no chronology and no reference.
python main.py --backbone ollama --backbone-model gemma4:26b

# the measured tables  (~12 min on 2 cores, plus generation if abstractive)
python run_experiments.py --all --tag main --backbone extractive

# subsets
python run_experiments.py --timeline --errors
python run_experiments.py --ablations
```

Ablations are switches, not edits:

```bash
python main.py --no-veridicality --no-scaffold --cascade 1
python main.py --projection absolute --year 33
```

### Outputs

Under `outputs/<tag>/`:

| Path | Contents |
|---|---|
| `annotation/<book>.tml` | conformant stand-off ISO-TimeML, one per Gospel |
| `annotation/<book>.tokens.xml` | the token layer the annotation points into |
| `annotation/<book>.json` | derived projection: modal paths, closed network |
| `stage3/timeline.json` | induced order, clusters, conflicts |
| `consolidated.txt` | the consolidated narrative |
| `consolidated_with_markers.txt` | the same, with event markers for auditing |
| `results.json` | every measured figure (`run_experiments.py`) |

Both `outputs/canonical/` and `outputs/ancoragem/` are kept: canonical as the pre-Addendum-9 "before" state, ancoragem as the primary configuration.

No information exists only in the JSON projection: the `.tml` documents are sufficient to reproduce the pipeline, which is what makes the annotation a resource rather than an intermediate representation.

---

## Data

> **Third-party material.** `data/` contains the New International Version text of the four Gospels for the Passion Week (© Biblica, Inc.) and Aschmann's harmony of the Gospels. Neither is the author's work and neither is licensed for redistribution by this repository's licence. They are present because the pipeline is digest-pinned to them and the experiments are not reproducible without the exact files. The thesis's own annotation is **stand-off**: every annotation artefact is keyed on `book:chapter:verse` and carries no verse text, so results, annotations and ratings can be published without the sources. If you intend to redistribute anything derived from this repository, take the stand-off artefacts and supply your own copy of the text.

1,245 verses across the four Gospels (Passion Week scope), the Aschmann harmony (held out, Stage 6 only), and the Golden Sample reference (Stage 6 only). Identified by content digest and verified on every run.

Digests, known input defects, and the half-verse citation ambiguity behind the purity ceiling: [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

---

## Layout

```
TAVERN/
├── main.py
├── run_experiments.py
├── scripts/
├── tavern/
│   ├── config.py
│   ├── pipeline.py
│   ├── stage1_preprocessing/
│   ├── stage2_temporal_annotation/
│   ├── stage3_anchoring_alignment/
│   ├── stage4_gnn/
│   ├── stage5_generation/
│   ├── stage6_evaluation/
│   └── baselines/
├── avaliacao_humana/
├── legacy/
├── CLAUDE.md
└── DATA_PROVENANCE.md
```

## Not implemented here

- `bart`/`pegasus`/`primera`/`instruct` are written but unrun on the induced timeline; their curated-timeline numbers are quoted from NSNC.
- BERTScore is wired but not run; the figures quoted come from the published work.
- No intrinsic annotation evaluation exists (no manual reference); the six consistency checks substitute.
- A half-verse-precise key — caps purity/B-cubed at ~89.5%, costs ~0.031 of R-L against the curated ceiling; not fixed (see `DATA_PROVENANCE.md`).
- The length asymmetry in `predicate_similarity`'s cosine — no compensation for a short account competing against a longer wrong candidate.
- `class_agreement`'s and `modal_compatibility`'s weights — measured net-harmful / near-zero, not reweighted.
- The absolute-day projection is only ~38–41% populated (42/112 day, 46/112 part) — declared future work, not fixed here.

---

## Citation

```bibtex
@article{finger2026narrative,
  author  = {Finger, Roger Antonio and Cortes, Eduardo Gabriel and
             Rigo, Sandro José and Ramos, Gabriel de Oliveira},
  title   = {Narrative Consolidation: Formulating a New Task for
             Unifying Multi-Perspective Accounts},
  journal = {Journal of the Brazilian Computer Society},
  volume  = {32},
  number  = {1},
  year    = {2026},
  doi     = {10.5753/jbcs.2026.7717},
  note    = {TAEG, Algorithm 1: https://github.com/neemias8/TAEG}
}

@inproceedings{finger2026neurosymbolic,
  author    = {Finger, Roger Antonio and Cortes, Eduardo Gabriel and
               Rigo, Sandro José and Ramos, Gabriel de Oliveira},
  title     = {Neurosymbolic Narrative Consolidation: Grounding Abstractive
               MDS and LLMs with Temporal Event Graphs},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {NSNC: https://github.com/neemias8/Neuro-Symbolic-Narrative-Consolidation ;
               Artefacts: Zenodo 10.5281/zenodo.19262044}
}
```

## References

- ISO 24617-1:2012. *Language resource management — Semantic annotation framework — Part 1: Time and events.*
- Pustejovsky, J. et al. (2010). ISO-TimeML: An International Standard for Temporal Annotation.
- Allen, J. F. (1983). Maintaining Knowledge about Temporal Intervals.
- Eades, P., Lin, X. & Smyth, W. F. (1993). A Fast and Effective Heuristic for the Feedback Arc Set Problem.
- Schlichtkrull, M. et al. (2018). Modeling Relational Data with Graph Convolutional Networks.
- Veličković, P. et al. (2018). Graph Attention Networks.
- Kendall, M. G. (1938). A New Measure of Rank Correlation.
- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
- Aschmann, P. Chronology of the Four Gospels.

## Acknowledgements

EMBRAPII · AKCIT · CNPq 313845/2023-9 · Positivo Tecnologia S/A.
Reference consolidation: Cunha (2025); Cunha & Sena (2026).

## License

Academic research project — UNISINOS. This licence covers the code and the stand-off annotation artefacts. It does **not** cover the New International Version text or Aschmann's harmony of the Gospels, present in `data/` for reproducibility — see the third-party notice under [Data](#data).
