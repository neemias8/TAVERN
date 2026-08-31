# TAVERN

**Temporal Anchoring for Version Consolidation in Abstractive Narrative Summarization**

TAVERN consolidates several long, overlapping narrative documents into one
chronologically ordered account — and **induces** the chronology from the text
rather than receiving it. Two earlier works in this same PhD share the task
and the sources: **TAEG** (JBCS) introduces Narrative Consolidation and solves
it **extractively** — one source account selected per event; **NSNC** (IJCNN)
generalises that to **abstractive**, neuro-symbolic fusion. Both take the same
external, ready-made chronology (Aschmann's harmony of the Gospels) as a given
input to alignment. TAVERN is abstractive like NSNC, but holds that same
chronology out and reads it only at evaluation — the temporal backbone instead
comes from an ISO 24617-1:2012 (ISO-TimeML) annotation of the sources.

Case study: the four canonical Gospels over the Passion Week.

> PhD thesis — Roger Antonio Finger, UNISINOS
> Advisor: Prof. Dr. Gabriel de Oliveira Ramos

---

## Why this repository changed

Earlier revisions of TAVERN itself — not TAEG or NSNC, which never claimed
otherwise — also took the Aschmann chronology as an input to alignment. That
made Kendall's τ — the ordering metric — **1.000 by construction** in every
configuration ever reported, because the system was never asked to exercise the
competence the metric measures. The chronology is now reachable only from
Stage 6, enforced in code:

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

This is the `ancoragem` configuration (tag-`ancoragem`, commit-tagged
`ancoragem-20260831`), the primary configuration this README reports as of
Addendum 9/10. The earlier `canonical-20260827` run (τ = 0.9155, 54.6%) is
retained throughout as the **"before" state**: the coreference score claimed
to use the TIMEX3 inventory's normalisation and the anchor chains (thesis
Achado 4), but that normalisation never actually reached the score, and a
hand-picked two-word stop-list stood in for entity discrimination. Addendum
9 implemented the mechanism the thesis already claimed; every comparison
below reports both, canonical first, so the fix is visible as a fix, not a
silent replacement.

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

Measured by `run_experiments.py --all --tag ancoragem --backbone ollama
--backbone-model gemma3:4b --ollama-repeat-penalty 1.1` on the digest-pinned
corpus (published configuration: `no_anchor_credit=False`, the default). This
is the primary run — commit-tagged `ancoragem-20260831` — and every number
below comes from that one execution unless marked **canonical (before)**,
which cites the prior run — commit-tagged `canonical-20260827` — kept as the
pre-fix state (see "Why this repository changed").

### The induced timeline

| Measure | canonical (before) | **ancoragem (primary)** |
|---|---|---|
| **Kendall's τ** | 0.9155 | **0.9274** |
| Pairwise ordering accuracy | 0.9577 | **0.9637** |
| Coverage | 0.8512 (143 of 168) | **0.8869 (149 of 168)** |
| Candidate canonical events induced | 249 (harmony has 169) | **289** |
| *Null:* N1, positional interleaving, **no annotation at all** | 0.8140 | 0.8140 |
| *Reference:* curated chronology | 1.000 **by construction** | 1.000 **by construction** |

τ = 0.9274 closes 61.0% of the distance from N1 (0.8140) to the curated
ceiling (1.000), against 54.6% before the fix — **the floor itself did not
move**; N1 is a property of the corpus and the null model, not of this
configuration. Report τ next to N1, never alone.

**τ is protected by construction, and this is measured, not asserted, in
both configurations.** The induced clustering is monotone by design
(progressive profile alignment), so the document-order votes in the global
tournament never conflict: `removed_arcs = 0` in both canonical and
ancoragem. Feed the *oracle* (curated) clustering into the identical
tournament instead — same algorithm, same code, only the grouping changes —
and `removed_arcs = 614`, and τ on that configuration **falls to 0.6296**,
below either fully-induced baseline. This cross-check number is unchanged by
Addendum 9's fix, because it depends only on `global_timeline.induce()` and
the oracle clustering, neither of which the fix touched. The correct
grouping exposes real cross-document ordering disagreements between the
Gospels that the induced grouping's monotonicity structurally cannot: τ is
high partly because the alignment never lets the tournament see the
disagreements it would have to resolve.

### Annotation

5,239 `<EVENT>` (4,235 verbal, 419 nominal, 585 states) · 152 realised
`<TIMEX3>` plus 886 empty anchoring elements · 746 `<SIGNAL>` · 1,208 asserted
and 3,460 closure-derived `<TLINK>` · 4,531 `<SLINK>` · 43 `<ALINK>` ·
7 `<MLINK>` · 1,944 timeline-eligible and 3,295 subordinated events.

**All twelve code-level conformance constraints pass on all four documents.**

Evidence cascade: level 1 (explicit signal) 512 · level 2 (temporal expression)
73 · level 3 (aspectual predicate) 33 · level 4 (narrative progression) 590.
Anchoring coverage (share of asserted `<TLINK>`s at evidence level 1–2) 48.4%.
This is a *relation*-level figure, not a claim that 48.4% of units have a
scaffold position — every unit does, because `Scaffold._solve_document`
interpolates one unconditionally. The unit-level figure that phrase invites a
reader to infer is a different, much lower number: only 10.1% of the 567
units (12.4% of the 410 timeline-eligible ones) are *observed* — pinned by
one of their own anchors rather than interpolated from a neighbour's.

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
**unsatisfiability alone, with no threshold**, in both configurations.
Inter-document conflicts: 94 (canonical), 96 (ancoragem) — more, finer-grained
clusters give the closure network more pairs to disagree over, not a
regression.

### Cluster purity and B-cubed: read against a ceiling, never against 100%

The error taxonomy's "over-merged" count (23 clusters, ancoragem; 25,
canonical) asks whether a whole cluster crosses a curated-event boundary; the
finer question is whether every *witness* a cluster holds maps to the same
curated event. It does in **44.4%** of multi-witness clusters
(`scripts/cluster_purity.py`; B-cubed P/R/F1 = 0.711/0.409/0.519), up from
**30.8%** (B-cubed 0.623/0.414/0.498) before the fix.

Neither number is read against 100%. Verified against a perfect (oracle)
clustering, purity still only reaches **89.5%** (B-cubed F1 0.829) — a
structural ceiling, not a code defect, and **unchanged by Addendum 9**: the
oracle clustering is built directly from the chronology
(`scripts/oracle_roundtrip.py:build_oracle`), so this ceiling is a property
of the corpus and the harmony's own citation granularity, not of the induced
system. `local_timeline.segment`'s unit boundary opens only between whole
verses; Aschmann's harmonisation occasionally individuates at the
*half*-verse level (`"14:66-68a"` for one event, `"14:68b"` for the next). 26
verse keys are claimed by two curated events this way, touching 40 of 169
events; **10 of those 40 have no verse exclusive to them at all** and so
cannot be individuated as distinct objects under any clustering, however
correct (events 47, 93, 99, 119, 132, 133, 155, 156, 157, 163 — 11 counting
event 53, which has no citation in any book). Three of those ten (47, 93,
157) never win *any* of their cited verses at all — an earlier-listed
competing event always claims them first — so together with event 53 they
are not merely hard to individuate but invisible to any verse-keyed
instrument (never a `bcubed` gold cluster; this is why `gold_clusters` reads
165, not 168). Read purity as **44.4% of the 89.5% achievable — 49.6% of
ceiling** (up from 34.4%), not as "44% of 100%".

Content coverage (share of the sources' content-word vocabulary the fusion
retains, `redundancy.coverage_over_events`): the abstractive consolidation
covers **93.3%** overall / **91.7%** for multi-source events, up from 91.2%
/ 89.7%. The reference itself covers only **88.0%** of that same vocabulary
(unchanged — the reference doesn't move) and the ancoragem consolidation
**96.4%** (whole-document measure; canonical was 95.1%) — the reference is
the longest curated account per event, which is extractive in kind, so a
fusion is penalised under every reference-based metric for material the
reference itself does not contain. That is the second threat to validity the
thesis names, measured rather than argued, and it **widens** as the fusion
gets more complete, not narrower.

**How much predicate evidence actually reaches the score, corrected twice.**
Addendum 9 first reported 27% of multi-witness clusters (43/159) with a
predicate/TIMEX3 term in common, by intersecting terms across *all* witnesses
of a cluster at once — too strict, since 96/159 clusters hold 3–4 Gospels and
the real decision is pairwise. Corrected to pairwise, canonical's no-evidence
count is 65/159 (40.9%), not 116/159 (73.0%). But pairwise raw-intersection
is itself the wrong instrument: ordinary narrative verbs (SAY, GO, COME) make
almost any two units' predicate sets intersect regardless of whether they
discriminate, which is exactly what IDF weighting suppresses — the actual
signal reaching `score()` is the *weighted cosine*, not set overlap.
`scripts/predicate_evidence_fraction.py` reports that directly, pooled over
every cross-book pair in every multi-witness cluster, as its 0.40-weighted
contribution:

| | canonical (before) | **ancoragem** |
|---|---|---|
| median 0.40×cosine contribution | 0.042 | **0.101** |
| p75 | 0.161 | **0.212** |
| fraction of pairs below 0.08 contribution | 61.0% | **46.3%** |

The direction is the one Addendum 9 predicted; the exact percentiles are this
script's own, not copied from an intermediate estimate.

### Selection accuracy: from below chance to above it

The published ladder's analytical floor (0.3474) is computed over the 95
contested curated events; the induced selection accuracy is measured over
whichever events the induced clustering actually produces a matched,
≥2-document cluster for — a different, and different-sized, denominator each
time, so comparing against 0.3474 directly would be the same error N1 first
exposed for τ. Recomputed each time over the matching subset's own
version-count distribution:

| | canonical (before) | **ancoragem** |
|---|---|---|
| subset size | 74 (42 with 3 versions, 18 with 4, 14 with 2) | **75** (43 with 3, 19 with 4, 13 with 2) |
| floor over that subset | 0.3446 | **0.3411** |
| selection accuracy | 0.2973 — **below** the floor | **0.3600 — above the floor** |

Report the floor next to the accuracy, and always the matching one.

### Grouping and ordering in Stage 3 are not separable

Crossing induced/oracle clustering against induced/oracle ordering
independently (`scripts/oracle_decomposition.py`) gives four configurations,
all measured with the same extractive/"longest" selection rule for a fair
comparison across cells that have no GNN score to use. **B and D depend only
on the oracle clustering** (built straight from the chronology,
`build_oracle`) **and are therefore identical in both configurations** — a
property of the corpus's citation granularity, not of the induced system. A
and C move:

| | grouping | ordering | canonical τ / ancoragem τ | canonical R-L / **ancoragem R-L** | selection |
|---|---|---|---|---|---|
| **A** (induced) | induced | induced | 0.9155 / **0.9274** | 0.594 / **0.662** | 0.297 (74) / **0.293 (75)** |
| **B** | oracle | induced | 0.6296 (both) | 0.492 (both) | 0.505 (95, both) |
| **C** | induced | oracle | 1.000 (both) | 0.487 / **0.479** | 0.297 (74) / **0.293 (75)** |
| **D** (oracle) | oracle | oracle | 1.000 (both) | 0.795 (both) | 0.505 (95, both) |

Canonical: D − A = 0.201, D − B = 0.303, D − C = 0.308 — sum 0.611 against a
real total of 0.201, residual 0.410. **Ancoragem: D − A = 0.133** (the
induced system alone closes markedly more of the gap to the ceiling than
before), D − B = 0.303 (unchanged), **D − C = 0.315** — sum 0.618 against a
real total of 0.133, **residual 0.486, if anything wider than canonical's**.
Fixing either component alone still makes ROUGE-L worse than the fully
induced baseline in both configurations (B and C both score below A);
grouping and ordering remain co-adapted to each other's imperfections, not
independently improvable — the fix narrowed Stage 3's own gap to its ceiling
without making the two components any more separable.

D's ROUGE-L (0.7948, both configurations) is the ceiling **TAVERN's own
architecture** can reach, not the ceiling of the curated timeline (0.8259,
"Curated, complete" below): the 0.031 gap is the cost of the verse being the
atom of segmentation, the same granularity limit that bounds purity — and,
being a property of that granularity rather than of the coreference score,
it does not move with Addendum 9's fix either.

### Downstream, and the criterion that was not met

The success criterion was fixed in advance by a published robustness analysis:
ROUGE-L F1 at or above the −10% point of the timeline-degradation curve. The
curated-timeline rows (complete / −10% / −25% / −50%) are unchanged by
Addendum 9 — they never touch the induced system, only the curated harmony
under a random perturbation:

| Timeline | ROUGE-L | ROUGE-1 |
|---|---|---|
| Curated, complete | 0.826 | 0.919 |
| Curated, −10% ← **the bar** | **0.795** | 0.880 |
| Curated, −25% | 0.708 | 0.791 |
| Curated, −50% | 0.546 | 0.607 |
| Induced order, curated segmentation, TAEG rule | 0.624 (canonical) → **0.675** | 0.941 → **0.930** |
| Induced order, curated segmentation, longest rule | 0.681 (canonical) → **0.760** | 0.978 (both) |
| Induced end to end (abstractive, Ollama gemma3:4b) | 0.497 (canonical) → **0.566** | 0.816 → **0.793** |

`repeat_penalty` matters here: llama.cpp's parameter of that name is not
HuggingFace's `repetition_penalty`, and reusing the HF value (1.5, correct
for the HF backbones) pushed the small model to glue words together
("...toBethphegeon theMountofOlves...", 3/249 events, 1.2%). Fixed to 1.1 for
Ollama specifically; end-to-end ROUGE-L moved from 0.331 to 0.497 on that fix
alone (`+50%` relative, canonical), and `scripts/check_text_quality.py` now
guards against the regression (0/249 corrupted on canonical, 0/289 on
ancoragem).

**The criterion is still not met** (0.566 < 0.795), and the thesis reports it
as failed rather than reinterpreting it — closer than before Addendum 9
(0.497), not closed. What locates the shortfall, in order of how much of it
each explains:

- **The reference is itself a selection** (88.0% content coverage, unchanged,
  against the fusion's 96.4% — up from 95.1%, so this threat **widens**, not
  narrows, as the fusion improves) — every reference-based metric penalises a
  fusion for material the reference doesn't contain.
- **Cluster purity is 49.6% of its own ceiling** (44.4% of 89.5%, up from
  34.4%) — more clusters that hold more than one witness now hold the
  *right* witnesses, but the induced grouping and ordering are still not
  separable, so the remaining gap cannot be closed one component at a time.
- **Nothing is lost at the event level, in either configuration.** No
  canonical event goes undetected.
- **Grouping and ordering interact, and the interaction did not shrink.** See
  the decomposition above: residual 0.486 (ancoragem) against 0.410
  (canonical) — Addendum 9 closed Stage 3's own gap to its ceiling without
  making grouping and ordering any more independently fixable.
- **The zero-annotation lexical baseline still wins.** See the recall@k
  section below: the fix closed 0.032 of a 0.108 gap, not the gap.

### The annotation's own discriminative power: recall@k of `score()` alone

Isolating the scoring function from the alignment algorithm entirely — curated
segmentation given for free, no monotone profile, no threshold, no band,
every one of a book's own curated-event spans ranked by the real
`predicate + participants + anchor + modal + class` score
(`scripts/measurement_a_recallk.py`, 634 queries, ~94 candidates each, chance
at rank 1 = 0.0106):

| | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| full score, canonical (before) | 0.405 | 0.732 | 0.836 | 0.551 |
| **full score, ancoragem** | **0.437** | **0.756** | **0.847** | **0.577** |
| **lexical baseline (zero annotation)** | **0.513** | **0.863** | **0.912** | **0.660** |
| − predicate | 0.238 | 0.508 | 0.620 | 0.369 |
| − participants | 0.398 | 0.675 | 0.767 | 0.527 |
| − anchor | 0.413 | 0.699 | 0.790 | 0.545 |
| − modal | 0.478 | 0.836 | 0.920 | 0.630 |
| − class | 0.453 | 0.770 | 0.842 | 0.596 |

(ablation rows are ancoragem's score with one term zeroed; the lexical
baseline never touches `score()`, so it is unaffected by Addendum 9 and is
listed once.)

**The zero-annotation lexical baseline (content-word TF cosine over the raw
verse text) still beats the full ISO-TimeML score, before and after the
fix.** Addendum 9 closed **0.032 of a 0.108 gap** (0.405→0.437 against a
fixed 0.513) — real movement, not the gap. Mapping words to predicate labels
still loses information the raw text retains for cross-document linking, on
this corpus; for linking events across documents, the ISO-TimeML abstraction
**costs** information rather than paying for it, and that is the clearest
single statement Stage 2's evaluation makes, reported with the same
prominence as the positive results. Predicate similarity is still the
dominant term (removing it costs the most, and costs *more* now: recall@1
drops to 0.238, a larger drop than canonical's 0.219 floor); **class is now
net-harmful** — removing it *raises* recall@1 from 0.437 to 0.453 — where it
contributed exactly 0.000 before the fix; **modal is more harmful than
before** — removing it raises recall@1 to 0.478 (canonical: 0.424). Neither
class nor modal was touched by Addendum 9; their apparent harm grew because
the *other* terms became better calibrated, making their existing noise
comparatively more visible, not because they got worse in isolation. Neither
finding is a reason to reweight the score: doing so after seeing this
ablation would be tuning against the reference the ablation itself used, and
would destroy the ablation's value. They are reported as findings for future
architecture work, not applied here.

### The projection reaches the score, but only partly

Addendum 9's mechanism (`scaffold.project_timexes`) derives an absolute day
and a within-day part for every *anchorable* `<TIMEX3>`, from `FEAST_DAY` and
`DAYPART_POSITION` — the same feast/daypart lexicon the scaffold already
resolves anchors against, never the chronology, never
`biblical_calendar.WEEKDAY_ORDER` (which carries the chronology's own day
labels and exists for the scaffold, not for this). Of the corpus's 112
anchorable `<TIMEX3>`: **42 (37.5%) get a concrete day** and 70 stay
subspecified (a non-narration expression, or an offset with nothing resolved
in scope to chain from); **46 (41.1%) get a concrete within-day part**, 66
do not. The `D:{day}`/`P:{part}` terms this produces are indexed into
`PredicateIDF` exactly like any other term, but for roughly three in five
anchorable expressions there is simply no term to index. **The entire gain
reported above came from a mechanism that is barely a third populated** —
declared here as the clearest piece of future work this addendum surfaces,
not something to close now: doing so after seeing this result would be
tuning against the very evaluation that reported it, and would invalidate
the `ancoragem` run.

### What the ablations say

| Configuration | canonical τ / cov / R-L | **ancoragem τ / cov / R-L** |
|---|---|---|
| Full | 0.9155 / 0.8512 / 0.497 | **0.9274 / 0.8869 / 0.566** |
| − veridicality partition | 0.9295 / 0.8393 / 0.526 | **0.9390 / 0.8690 / 0.568** |
| − closure | 0.9155 / 0.8512 / 0.499 | **0.9274 / 0.8869 / 0.568** |
| − anchor scaffold | 0.9035 / 0.8393 / 0.459 | **0.9013 / 0.8810 / 0.458** |
| − graph propagation | 0.9155 / 0.8512 / 0.500 | **0.9274 / 0.8869 / 0.570** |
| cascade levels {1} only | 0.9155 / 0.8512 / 0.497 | **0.9274 / 0.8869 / 0.569** |

(canonical's column corrects the ROUGE-L values previously listed here — an
earlier revision of this table predated the fix that makes the ablation
table reuse the main run's own result object instead of a second, separately
non-deterministic GNN pass; these are the values `outputs/canonical/
results.json` actually carries.)

Two of these contradict predictions the thesis recorded in advance, and are
reported rather than defended, in both configurations.

**The cascade does not move τ — and not because the relations are
uninformative.** ISO-TimeML defines no relation *between* documents, so the
cross-document merge cannot run through `<TLINK>`s on any implementation of the
standard; it runs through the `<TIMEX3>` normalisation and the anchor chains,
which is what the scaffold is built from — and the scaffold is the one component
whose removal degrades τ, coverage and ROUGE-L together. Within a document the
links *could* correct the narrative order, and here they never need to: of the
371 cluster pairs carrying relational evidence in the canonical run, **371
agree with the induced order and 0 contradict it** (not recomputed against
ancoragem's 289 clusters — Addendum 10 authorised only the oracle
decomposition re-run, and the count would change with the clustering even if
the finding almost certainly would not). The Evangelists narrate in sequence.
That is a property of the corpus, and it is the sharpest limit on the
generality of the result: on narrative that uses flashback, those constraints
would be the only defence against the wrong answer, and this corpus cannot
distinguish a mechanism that works from one that is never tested.

**Removing the veridicality partition slightly raises τ.** So the partition is a
correctness requirement on the annotation — check 5 fails without it — rather
than an accuracy improvement on the ordering.

### Two corrections to the prior work

The "prior work" here is TAEG (Algorithm 1: centrality over a
similarity-weighted event graph), the JBCS paper's own baseline, reproduced
in `tavern/baselines/__init__.py` and at
[github.com/neemias8/TAEG](https://github.com/neemias8/TAEG).

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
package directly into site-packages. `torch` is needed only for Stage 4; without
it the pipeline falls back to the unpropagated aggregation.

ROUGE-L over a ~16k-token reference cannot use `rouge_score`'s own
longest-common-subsequence table, which is quadratic in pure Python and takes
minutes per pair. `content_metrics._lcs_length` therefore uses `pylcs` when it is
installed and otherwise a bit-parallel fallback (Crochemore, Iliopoulos, Pinzon
and Reid, 2001) that packs a row of the table into one arbitrary-precision
integer: 0.3 s on the reference, and `verify_fast_path` asserts either path
reproduces `rouge_score` exactly (it does, to 0.0). `pylcs` is optional and
commented out in `requirements.txt` because it is a compiled extension without a
wheel for every interpreter.

## Run

### Everything at once

```bash
ollama pull gemma3:4b        # once
python run_all.py            # or, on Windows PowerShell:  .\run-all.bat
```

On a fresh clone, check out the branch first — `git clone` leaves you on the
repository's default branch, which does not contain any of this:

```powershell
git fetch <remote-or-bundle> main:tavern-thesis-framework
git checkout tavern-thesis-framework
pip install -r requirements.txt
python run_all.py
```

`run_all.py` checks the environment and the corpus digests, measures **both**
configurations — extractive, for comparability with the degradation curve, and
abstractive, which is what the framework is for — regenerates the curation
sheets, and packages everything into one `tavern_results_<stamp>.zip`.

The generation run is ~289 model calls (one per induced cluster) and is
**cached to disk as it goes**, in a single file shared across every tag and
config (`outputs/fusion_cache.jsonl`, keyed by backbone + model +
`repeat_penalty` + the exact source texts — see `CachedFuser` in
`stage5_generation/backbones.py`), so an interrupted run resumes on the same
command, and re-running `--ablations` under a different configuration reuses
whatever it shares with a prior run instead of regenerating from scratch.
Expect 40 min to a few hours end to end on CPU, depending on the accelerator
and on how much of the cache is already warm.

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
| `results.json` | every measured figure (`run_experiments.py`); on `outputs/canonical/`, also a `clustering_quality` key (purity, its ceiling, B-cubed, content coverage — `scripts/cluster_purity.py`) |

Both `outputs/canonical/` and `outputs/ancoragem/` are kept: canonical as the
pre-Addendum-9 "before" state, ancoragem as the primary configuration.

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
├── scripts/                    Stage 6 diagnostics, cited by name throughout
│                                Results: cluster_purity.py, oracle_roundtrip.py,
│                                oracle_decomposition.py, measurement_a_recallk.py,
│                                measurement_b_graph.py, predicate_evidence_fraction.py,
│                                test_no_evidence_floor.py, make_curation.py
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

`ollama`/gemma3:4b **has been run**, on both the canonical and the primary
`ancoragem` runs (`repeat_penalty=1.1`, fixed from the HuggingFace-inherited
1.5 that glued words together — see Results). It is the intended abstractive
configuration, not a fallback; the consolidation committed under
[`consolidations/`](consolidations/) is regenerated from `ancoragem`, not
from `canonical` or `union`.

| Backbone | R-1 | R-2 | R-L | METEOR | Length | corrupted |
|---|---|---|---|---|---|---|
| `ollama` (gemma3:4b, canonical — before) | 0.816 | 0.736 | 0.497 | 0.479 | 121,832 | 0/249 |
| **`ollama` (gemma3:4b, ancoragem)** | **0.793** | **0.734** | **0.566** | **0.477** | **128,837** | **0/289** |

The `extractive` backbone comparison row from earlier revisions of this file
is not reproduced here: it was measured under the pre-Addendum-9 Stage 3, and
Addendum 10 authorised only the oracle-decomposition re-run, not a fresh
extractive backbone pass — refreshing it is future work, not a number to
guess at.

`ollama`'s ROUGE-1 fell slightly (0.816→0.793) while its content coverage
rose (95.1%→96.4% of the sources' content-word vocabulary, against the
reference's own unchanged 88.0%) and its ROUGE-L *rose* (0.497→0.566) — the
reference-is-a-selection effect above widens exactly where the fusion gets
more complete, and the ROUGE-L improvement shows Addendum 9's fix reached
the generation stage, not just Stage 3's own internal metrics.

### For expert curation

`consolidations/curation.md` and `.csv` lay out, per event, every source account
beside the consolidation derived from it, with verse addresses, the scaffold's
day index, a conflict flag, and a blank verdict block asking three separate
questions — **faithful**, **complete**, **placement**. They come apart in
practice, in both configurations: no canonical event is undetected; two
(ancoragem) or one (canonical) are displaced across a day boundary; 29
(ancoragem) or 31 (canonical) are transposed with a same-day neighbour.

## Not implemented here

Stated so the repository is not read as claiming more than it does.

- **`bart`/`pegasus`/`primera`/`instruct` are written but unrun** on the
  induced timeline. `ollama`/gemma3:4b has been (see Results); the others need
  a HuggingFace download this environment did not have reachable. The IJCNN
  results for BART / PEGASUS / PRIMERA / Gemma-3 under a *curated* timeline
  are quoted from that paper (NSNC) and reproduced in
  [Neuro-Symbolic-Narrative-Consolidation](https://github.com/neemias8/Neuro-Symbolic-Narrative-Consolidation).
- **BERTScore.** Wired but not run in the reported tables; the figures quoted
  for it come from the published work, computed without baseline rescaling.
- **Intrinsic annotation evaluation.** No manually annotated reference exists,
  so span-level F1, attribute accuracy and inter-annotator agreement are
  unavailable. This is the largest limitation of the evaluation and the six
  internal checks detect incoherence, not error.
- **A half-verse-precise key.** Aschmann's harmonisation occasionally
  individuates below verse granularity; TAVERN's atom is the whole verse
  (`local_timeline.segment`'s `for v in verses`). This caps cluster purity and
  B-cubed at ~89.5%, not 100%, and costs ~0.031 of ROUGE-L against the
  curated ceiling (see Results). Not fixed: it would mean a half-verse key
  threaded through `Corpus`/`ReferenceParser`/`Chronology`, used everywhere,
  for a gain in reporting precision rather than in the system. Documented as
  a known property of the resource in `DATA_PROVENANCE.md`.
- **The length asymmetry in `predicate_similarity`.** Its cosine is correctly
  L2-normalised, but nothing in `score()` compensates for a short, terse
  account competing against a longer, lexically richer wrong candidate — a
  real, unfixed asymmetry, not investigated further because fixing it now
  would invalidate the `ancoragem` run.
- **`class_agreement`'s and `modal_compatibility`'s weights.** Addendum 9 did
  not touch either term, but their measured harm grew once the other terms
  were fixed: `class` moved from contributing exactly 0.000 (canonical) to
  net-harmful (ancoragem, removing it raises recall@1 by 0.016);
  `modal_compatibility` was already net-harmful and is now more so (+0.041
  on removal, against +0.019 before). Left as measured, in both states:
  reweighting after seeing an ablation would be tuning against the reference
  the ablation used.
- **The absolute-day/within-day projection is only ~38–41% populated.**
  42 of 112 anchorable `<TIMEX3>` get a concrete day, 46 a concrete part
  (`scaffold.project_timexes`); the rest stay subspecified. The entire
  `ancoragem` gain over `canonical` came through this partly-populated
  mechanism. Not extended now: doing so after measuring `ancoragem`'s result
  would be tuning against the very evaluation that reported it.

## Notes for anyone extending this

Five things cost real time to discover; `CLAUDE.md` has the full list.

- **A structural score defect is worth a failing test before the fix, not
  after.** Two absence-of-evidence bugs — `participant_similarity`'s bare
  Jaccard over the ubiquitous entities, `modal_compatibility`'s 1.0 default
  for no modal evidence — let two units with zero shared predicate, zero
  distinguishing entity and zero anchor evidence clear `MATCH_THRESHOLD`
  (0.25×1.0 + 0.10×1.0 = 0.35 ≥ 0.34).
  `scripts/test_no_evidence_floor.py` was committed failing against the
  pre-fix code (commit `bf12077`) before Addendum 9's fix landed (`9d9e0ec`)
  — the failing commit is the proof the defect existed, kept in history
  rather than folded into the fix as an invisible footnote.

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
  doi     = {10.5753/jbcs.2026.7717},
  note    = {TAEG, Algorithm 1: https://github.com/neemias8/TAEG}
}

@inproceedings{finger2026neurosymbolic,
  author    = {Finger, Roger Antonio and Cortes, Vinicius and
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
