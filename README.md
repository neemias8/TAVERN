# TAVERN

**Temporal Anchoring for Version Consolidation in Abstractive Narrative Summarization**

TAVERN consolidates several long, overlapping narrative documents into one
chronologically ordered account — and **induces** the chronology from the text
rather than receiving it. The temporal backbone comes from an ISO 24617-1:2012
(ISO-TimeML) annotation of the sources; the curated harmony that earlier work
consumed as input is held out here as an evaluation reference.

Case study: the four canonical Gospels over the Passion Week.

> PhD thesis — Roger Antonio Finger, UNISINOS
> Advisor: Prof. Dr. Gabriel de Oliveira Ramos

---

## Why this repository changed

Earlier revisions took the Aschmann chronology as an input to alignment. That
made Kendall's τ — the ordering metric — **1.000 by construction** in every
configuration ever reported, because the system was never asked to exercise the
competence the metric measures. The chronology is now reachable only from
Stage 6, enforced in code:

```python
# tavern/config.py
def assert_no_chronology_import() -> None:
    """Raises if a chronology load originates in stages 1–5."""
```

τ becomes a measurement. It is **0.9163**.

The previous four-stage pipeline is preserved under `legacy/` rather than
deleted.

---

## Architecture

```
                        Aschmann harmony ─── HELD OUT ───┐
                        Golden Sample    ─── HELD OUT ───┤
                                                         ▼
Matthew ─┐   ┌────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
Mark   ──┤   │ 1. Pre-    │  │ 2. Temporal  │  │ 3. Anchoring │  │ 6. Eval  │
Luke   ──┼──▶│  processing│─▶│  Annotation  │─▶│   Alignment  │  │          │
John   ──┘   │  tokens,   │  │  ISO-TimeML  │  │   Graph      │  └────▲─────┘
             │  pericopes │  │  A: conform. │  │              │       │
             └────────────┘  │  B: derived  │  └──┬────────┬──┘       │
                             └──────────────┘     │        │          │
                                          (G, T̂) │        │ T̂        │
                                    ┌─────────────▼──┐  ┌──▼────────────────┐
                                    │ 4. R-GAT       │─▶│ 5. Micro-abstract.│
                                    │    over the    │  │    fusion, in T̂  │
                                    │    typed graph │  │    order          │
                                    └────────────────┘  └───────────────────┘
```

**Stage 1** — parses both XML schemas present in the resource, builds the token
layer the stand-off annotation points into (punctuation kept: the standard
admits it as a `<SIGNAL>` target), attaches 91 titled pericopes, resolves
entity chains.

**Stage 2** — two layers.
*Layer A* is conformance: `<EVENT>` (verbal, nominal and adjectival triggers,
seven classes), `<TIMEX3>` with the biblical domain profile, `<SIGNAL>`,
`<TLINK>` through a five-level evidence cascade, `<SLINK>`, `<ALINK>`,
`<MLINK>`, `<CONFIDENCE>`. Three domain problems are solved *inside* the
standard: no document creation time (a three-level anchoring hierarchy of empty
`<TIMEX3>` elements and `@anchorTimeID` chains), a luni-solar calendar with
temporal hours and a sunset day boundary (a declared profile), and the
narrative/reported distinction (derived from the `<SLINK>`s the standard already
requires).
*Layer B* derives, from Layer A alone: modal context paths, the veridicality
partition, and closure under Allen's interval algebra with conflict detection.

**Stage 3** — event units, a shared day axis from the anchorable expressions,
progressive profile alignment across documents, a weighted tournament resolved
by Eades–Lin–Smyth, and the typed cross-document event graph.

**Stage 4** — relational graph attention with the edge's confidence and its
asserted/derived flag inside the attention computation.

**Stage 5** — micro-abstractive fusion: one paragraph per candidate canonical
event, in induced order. Chronology is a property of the loop, so ordering
errors are attributable to Stage 3.

**Stage 6** — the only place the harmony and the reference may be read.

---

## Results

Measured by `run_experiments.py --all` on the digest-pinned corpus.

### The induced timeline

| Measure | Value |
|---|---|
| **Kendall's τ** | **0.9163** |
| Pairwise ordering accuracy | 0.9581 |
| Coverage | 0.8512 (143 of 168 events) |
| Candidate canonical events induced | 248 (harmony has 169) |
| *Reference:* curated chronology | 1.000 **by construction** |
| *Reference:* LexRank, no temporal structure | 0.320 |

### Annotation

5,239 `<EVENT>` (4,235 verbal, 419 nominal, 585 states) · 152 realised
`<TIMEX3>` plus 886 empty anchoring elements · 746 `<SIGNAL>` · 1,209 asserted
and 3,469 closure-derived `<TLINK>` · 4,524 `<SLINK>` · 43 `<ALINK>` ·
7 `<MLINK>` · 1,945 timeline-eligible and 3,294 subordinated events.

**All twelve code-level conformance constraints pass on all four documents.**

Evidence cascade: level 1 (explicit signal) 512 · level 2 (temporal expression)
73 · level 3 (aspectual predicate) 33 · level 4 (narrative progression) 591.
Anchoring coverage 48.4 %.

### Internal consistency

Six checks, run on every execution, in place of the intrinsic annotation
evaluation no reference permits.

| Check | Result |
|---|---|
| Schema validity, C1–C12, accessibility | **pass** |
| Closure consistency (no intra-document unsatisfiable cycle) | **pass** |
| Anchoring coverage | 0.239 *(measurement)* |
| Normalisation coverage | 1.000 |
| Partition soundness (no subordinated event on the timeline) | **pass**, 0 leaked |
| Known-conflict regression | **3/3** |

The last row matters: the three divergences the harmonisation literature
documents — the timing of the fig tree, the relation of the crucifixion to the
Passover, the sequence of the cockcrow — are all recovered from
**unsatisfiability alone, with no threshold**.

### Downstream, and the criterion that was not met

The success criterion was fixed in advance by a published robustness analysis:
ROUGE-L F1 at or above the −10 % point of the timeline-degradation curve.

| Timeline | ROUGE-L | ROUGE-1 |
|---|---|---|
| Curated, complete | 0.831 | 0.922 |
| Curated, −10 % ← **the bar** | **0.787** | 0.878 |
| Curated, −25 % | 0.704 | 0.787 |
| Curated, −50 % | 0.549 | 0.610 |
| **Induced ordering, curated segmentation** | **0.615** | 0.927 |
| **Induced end to end** | **0.595** | 0.927 |

**The criterion is not met**, and the thesis reports it as failed rather than
reinterpreting it. Three things locate the shortfall:

- **Nothing is lost.** No canonical event goes undetected, and ROUGE-1 (0.927)
  exceeds the complete curated timeline's (0.922).
- **Nothing is grossly misplaced.** Exactly **one** event crosses a day
  boundary in the wrong direction. The anchor scaffold gets the week right.
- **The errors are local.** 30 transpositions of events on the *same day*.
  ROUGE-L is brutally non-linear in these: over the 143 matched events the
  longest monotone chain is 94 (65.7 %), which is where ≈0.6 comes from.

### What the ablations say

| Configuration | τ | Coverage | ROUGE-L |
|---|---|---|---|
| Full | 0.9163 | 0.8512 | 0.503 |
| − anchor scaffold | 0.9035 | 0.8393 | 0.448 |
| − veridicality partition | 0.9295 | 0.8393 | 0.537 |
| − closure | 0.9163 | 0.8512 | 0.512 |
| − graph propagation | 0.9163 | 0.8512 | 0.571 |
| cascade levels {1} only | 0.9163 | 0.8512 | 0.595 |

Two of these contradict predictions the thesis recorded in advance, and are
reported rather than defended.

**The cascade does not move τ — and not because the relations are
uninformative.** ISO-TimeML defines no relation *between* documents, so the
cross-document merge cannot run through `<TLINK>`s on any implementation of the
standard; it runs through the `<TIMEX3>` normalisation and the anchor chains,
which is what the scaffold is built from — and the scaffold is the one component
whose removal degrades τ, coverage and ROUGE-L together. Within a document the
links *could* correct the narrative order, and here they never need to: of the
371 cluster pairs carrying relational evidence, **371 agree with the induced
order and 0 contradict it**. The Evangelists narrate in sequence. That is a
property of the corpus, and it is the sharpest limit on the generality of the
result: on narrative that uses flashback, those 371 constraints would be the
only defence against the wrong answer, and this corpus cannot distinguish a
mechanism that works from one that is never tested.

**Removing the veridicality partition slightly raises τ.** So the partition is a
correctness requirement on the annotation — check 5 fails without it — rather
than an accuracy improvement on the ordering.

### Two corrections to the prior work

1. The reference consolidation **numbers its own events**, so its per-event
   segmentation is recoverable exactly rather than by text similarity. Against
   the exact segmentation, the centrality-based selection previously reported at
   0.489 scores **0.337** — at the analytical random floor of 0.3474. The length
   heuristic survives (0.632 against 0.648) and is the only reference system on
   this resource demonstrably above chance at selecting versions.
2. The chronology cites Luke 13:34–35 for event #28, outside the Luke 19–24
   scope. There are 363 citations but **362 resolvable versions**, which moves
   the version distribution and the random floor.

The published degradation curve reproduces to within 0.02 of ROUGE-L at every
level, which is what makes these comparison points usable.

---

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Two dependencies need a note. `rouge-score==0.1.2` fails to build against recent
setuptools — install with `--no-build-isolation`, or copy the `rouge_score`
package directly into site-packages. `pylcs` supplies the compiled
longest-common-subsequence used for ROUGE-L; the reference is ~16k tokens and
the pure-Python table takes minutes per pair. `torch` is needed only for
Stage 4; without it the pipeline falls back to the unpropagated aggregation.

## Run

### Everything at once

```bash
ollama pull gemma3:4b        # once
python run_all.py            # or: run-all.bat  on Windows
```

`run_all.py` checks the environment and the corpus digests, measures **both**
configurations — extractive, for comparability with the degradation curve, and
abstractive, which is what the framework is for — regenerates the curation
sheets, and packages everything into one `tavern_results_<stamp>.zip`.

The generation run is 248 model calls and is **cached to disk as it goes**
(`outputs/<tag>/fusion_cache.jsonl`), so an interrupted run resumes on the same
command instead of starting over. Expect 40 min to a few hours end to end,
depending on the accelerator.

```bash
python run_all.py --backbone union            # dry run, no model needed
python run_all.py --skip-extractive           # if that run already exists
python run_all.py --backbone primera --model allenai/PRIMERA
```

### One stage at a time

```bash
# stages 1–5 only. Reads no chronology and no reference.
python main.py --backbone ollama --backbone-model gemma3:4b

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

No information exists only in the JSON projection: the `.tml` documents are
sufficient to reproduce the pipeline, which is what makes the annotation a
resource rather than an intermediate representation.

---

## Data

Identified by content digest and verified on every run — see
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

| File | Role |
|---|---|
| `data/EnglishNIV{Matthew40,Mark41,Luke42,John43}_PW.xml` | 1,245 verses, 25,893 words |
| `data/NIV_*_PW_with_pericopes.xml` | pericope layer only (91 units) |
| `data/ChronologyOfTheFourGospels_PW.xml` | Aschmann harmony — **held out** |
| `data/Golden_Sample.txt` | reference consolidation — **Stage 6 only** |

Two defects in the inputs are repaired in code and reported, not silently: the
pericope files stop short of the end of each book (Mark 16:9–20 and
Luke 24:13–53 are cited by the harmony, so the resurrection events would
otherwise lose all their sources), and event #28's citation falls outside scope.

The NIV text is under copyright. The annotation is distributed in **stand-off**
form, keyed on `book:chapter:verse` and token offsets, so a recipient holding a
licensed copy of the source text can reconstruct the full input.

---

## Layout

```
TAVERN/
├── main.py                     stages 1–5
├── run_experiments.py          every measured table (Stage 6 lives here)
├── tavern/
│   ├── config.py               paths, digests, all ablation switches
│   ├── pipeline.py             stage driver
│   ├── stage1_preprocessing/   corpus, segmentation, pericopes, coref
│   ├── stage2_temporal_annotation/
│   │   ├── model.py enums.py   elements; closed value sets; conformance
│   │   ├── event_tagger.py timex_tagger.py timex_normalizer.py
│   │   ├── biblical_calendar.py        the domain profile
│   │   ├── signal_tagger.py
│   │   ├── link_inference/     tlink, slink, alink, mlink
│   │   ├── veridicality.py closure.py            Layer B
│   │   └── serializer.py reader.py validator.py
│   ├── stage3_anchoring_alignment/
│   │   ├── local_timeline.py   event units, local partial order
│   │   ├── scaffold.py         the shared day axis
│   │   ├── event_coref.py      profile alignment, episode merge, conflicts
│   │   ├── global_timeline.py  tournament, Eades–Lin–Smyth, topological sort
│   │   └── graph.py            the typed cross-document event graph
│   ├── stage4_gnn/             R-GAT, and the unpropagated baseline
│   ├── stage5_generation/      micro-abstractive fusion
│   ├── stage6_evaluation/      metrics, τ, checks, conflicts, error taxonomy
│   └── baselines/              the published ladder, the degradation curve
├── legacy/                     the previous four-stage pipeline, preserved
├── CLAUDE.md                   working notes and the non-obvious pitfalls
└── DATA_PROVENANCE.md          digests and the input defects
```

## Consolidation output, and which backbone made it

TAVERN is an **abstractive** framework: it fuses the parallel accounts of each
event rather than selecting one, because the task's Completeness and
Representativeness objectives require every version's detail to survive into the
consolidation. Selecting one account discards the rest by construction.

`--backbone` picks the fuser. All of them fuse **per event**, in induced order,
so chronology is a property of the loop and no backbone can violate it.

| Backbone | Abstractive | Needs a model | Notes |
|---|---|---|---|
| `union` | no | no | deterministic; every sentence survives unless another covers it. Detail-preserving, but generates no new wording, so seams show |
| `extractive` | no | no | emits one account verbatim; exists **only** so the numbers are comparable with the degradation curve, whose rows are extractive |
| `ollama` | **yes** | `ollama pull` | instruction-tuned, conflict-aware; easiest path on a workstation |
| `instruct` | **yes** | HF download | same, in-process |
| `bart`, `pegasus`, `primera` | **yes** | HF download | the IJCNN backbones, with that paper's controls: no prompt, `<doc-sep>` for PRIMERA, single-document PEGASUS checkpoint |

```bash
ollama pull gemma3:4b
python main.py --tag gemma --backbone ollama --backbone-model gemma3:4b
python scripts/make_curation.py outputs/gemma/curation.json consolidations/gemma/
```

The consolidation committed under [`consolidations/`](consolidations/) is the
`union` output, produced where no model was reachable. It is a **floor, not the
intended output**, and `consolidations/README.md` says so on its first line.

| Backbone | R-1 | R-2 | R-L | METEOR | Length |
|---|---|---|---|---|---|
| `extractive` | 0.927 | 0.802 | 0.595 | 0.519 | 79,921 |
| `union` | 0.916 | **0.819** | 0.577 | **0.551** | 94,101 |

Union trades a little ROUGE-1 precision for higher ROUGE-2 and METEOR: it keeps
more of the reference's actual phrasing because it keeps more of the sources.
That is the trade the task's objectives ask for.

### For expert curation

`consolidations/curation.md` and `.csv` lay out, per event, every source account
beside the consolidation derived from it, with verse addresses, the scaffold's
day index, a conflict flag, and a blank verdict block asking three separate
questions — **faithful**, **complete**, **placement**. They come apart in
practice: no canonical event is undetected, exactly one is displaced across a
day boundary, and 30 are transposed with a same-day neighbour.

## Not implemented here

Stated so the repository is not read as claiming more than it does.

- **The abstractive backbones are written but unrun.** The code is here with the
  published generation controls; the container it was last run in had no access
  to a model, so no abstractive figures are reported. The IJCNN results for
  BART / PEGASUS / PRIMERA / Gemma-3 under a *curated* timeline are quoted from
  that paper and reproduced in
  [Neuro-Symbolic-Narrative-Consolidation](https://github.com/neemias8/Neuro-Symbolic-Narrative-Consolidation).
- **BERTScore.** Wired but not run in the reported tables; the figures quoted
  for it come from the published work, computed without baseline rescaling.
- **Intrinsic annotation evaluation.** No manually annotated reference exists,
  so span-level F1, attribute accuracy and inter-annotator agreement are
  unavailable. This is the largest limitation of the evaluation and the six
  internal checks detect incoherence, not error.

## Notes for anyone extending this

Four things cost real time to discover; `CLAUDE.md` has the full list.

- Quotation scope must be computed **per document, not per sentence**. The
  translation reopens a quotation mark at each paragraph of a long speech and
  closes only at the end; a counter that toggles on every mark inverts its state
  and puts narration inside the quotation. That error let 885 of 1,585
  discourse-block events reach the timeline. Check 5 is what catches it.
- Alignment must be **one progressive profile**, not six pairwise runs.
  Independent pairwise alignments disagree, and merging them transitively builds
  clusters that violate the documents' own orders.
- **A monotone profile cannot represent a disagreement about order.** Conflicts
  are recovered *before* consistency is imposed. If the conflict count is ever
  0, that is the bug, not a clean corpus.
- Allen's composition table is **generated from the endpoint algebra**, not
  transcribed, and verified against Allen (1983). 169 hand-typed cells is 169
  chances to be wrong.

---

## Citation

```bibtex
@article{finger2026narrative,
  author  = {Finger, Roger Antonio and Cortes, Vinicius and
             Rigo, Sandro José and Ramos, Gabriel de Oliveira},
  title   = {Narrative Consolidation: Formulating a New Task for
             Unifying Multi-Perspective Accounts},
  journal = {Journal of the Brazilian Computer Society},
  volume  = {32},
  number  = {1},
  year    = {2026},
  doi     = {10.5753/jbcs.2026.7717}
}

@inproceedings{finger2026neurosymbolic,
  author    = {Finger, Roger Antonio and Cortes, Vinicius and
               Rigo, Sandro José and Ramos, Gabriel de Oliveira},
  title     = {Neurosymbolic Narrative Consolidation: Grounding Abstractive
               MDS and LLMs with Temporal Event Graphs},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {Artefacts: Zenodo 10.5281/zenodo.19262044}
}
```

## References

- ISO 24617-1:2012. *Language resource management — Semantic annotation
  framework — Part 1: Time and events.*
- Pustejovsky, J. et al. (2010). ISO-TimeML: An International Standard for
  Temporal Annotation.
- Allen, J. F. (1983). Maintaining Knowledge about Temporal Intervals.
- Eades, P., Lin, X. & Smyth, W. F. (1993). A Fast and Effective Heuristic for
  the Feedback Arc Set Problem.
- Schlichtkrull, M. et al. (2018). Modeling Relational Data with Graph
  Convolutional Networks.
- Veličković, P. et al. (2018). Graph Attention Networks.
- Kendall, M. G. (1938). A New Measure of Rank Correlation.
- Lin, C.-Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.
- Aschmann, P. Chronology of the Four Gospels.

## Acknowledgements

EMBRAPII · AKCIT · CNPq 313845/2023-9 · Positivo Tecnologia S/A.
Reference consolidation: Cunha (2025); Cunha & Sena (2026).

## License

Academic research project — UNISINOS. The NIV source text is under copyright
and is not redistributed; the annotation is stand-off.
