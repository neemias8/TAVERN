# TAVERN — working notes

TAVERN induces a chronological backbone for Narrative Consolidation from
ISO 24617-1:2012 temporal annotation, instead of receiving one. The thesis is
the specification; this file is what a new session needs before touching code.

## Layout

```
tavern/
  config.py                 paths, digests, TavernConfig (all ablation switches)
  pipeline.py               stages 1–5; caches stages 1–2 across the ablation grid
  stage1_preprocessing/     corpus, token layer, 91 pericopes, entity chains
  stage2_temporal_annotation/
      Layer A: event_tagger, timex_tagger, timex_normalizer,
               biblical_calendar (the domain profile), signal_tagger,
               link_inference/{tlink,slink,alink,mlink}, serializer, reader
      Layer B: veridicality, closure  (validator covers Appendix B's C1–C12)
  stage3_anchoring_alignment/
      local_timeline  event units + local partial order
      scaffold        the shared day axis
      event_coref     progressive profile alignment + episode merge + conflicts
      global_timeline weighted tournament, Eades–Lin–Smyth, topological sort
      graph           the typed cross-document event graph
  stage4_gnn/               R-GAT, and the unpropagated baseline it is compared to
  stage5_generation/        micro-abstractive fusion; selection is over DOCUMENTS
  stage6_evaluation/        the only place the chronology may be read
  baselines/                the published ladder, the degradation curve
run_experiments.py          reproduces every measured table
```

## Run

```
python run_experiments.py --all --tag main      # ~12 min, 2 cores
python run_experiments.py --timeline --errors   # subsets
```
Results land in `outputs/<tag>/results.json` plus the `.tml` documents, the
token layer, the JSON projection and the consolidated narrative.

## Non-obvious things, all of them learned the hard way

- **`book[:2]` collides** for matthew/mark. Use `config.BOOK_CODE`.
- **Quotation scope must be computed per document, not per sentence.** The NIV
  reopens a quotation mark at each paragraph of a long speech and closes only at
  the end. A counter that toggles on every mark inverts its state and puts
  narration inside the quotation. Getting this wrong let 885 of 1,585
  discourse-block events reach the timeline; check 5 is what catches it.
- **Alignment must be a single progressive profile, not six pairwise runs.**
  Independent pairwise alignments disagree, and merging them transitively builds
  clusters that violate the documents' own orders.
- **A monotone profile cannot represent order disagreement.** Conflicts are
  recovered *before* the alignment imposes consistency, as the mutually-best
  matches outside a maximum monotone subsequence. If the conflict count is ever
  0, that is the bug, not a clean corpus.
- **ROUGE-L must not use `rouge_score`'s own LCS table.** It is a quadratic
  Python table and takes minutes on a 16k-token reference, which makes the
  ablation grid impractical. `content_metrics._lcs_length` uses `pylcs` when it
  is installed and otherwise the bit-parallel algorithm of Crochemore et al.
  (2001), which packs a DP row into one big integer — 0.3 s on the reference,
  no compiled extension needed. `verify_fast_path` asserts either path equals
  `rouge_score` exactly; it does, to 0.0.
- **Selection is over documents, emitting a contiguous span.** A canonical event
  cites a verse range per Gospel; selecting per unit fragments the account and
  costs ~0.1 of ROUGE-L.
- Allen's composition table is generated from the endpoint algebra, not typed in.
- **A structural score defect is worth a failing test before the fix, not
  after.** `participant_similarity`'s bare Jaccard over the ubiquitous
  entities and `modal_compatibility`'s 1.0 default for no modal evidence let
  two units with zero shared predicate, zero distinguishing entity and zero
  anchor evidence clear `MATCH_THRESHOLD` (0.25×1.0 + 0.10×1.0 = 0.35 ≥
  0.34). `scripts/test_no_evidence_floor.py` was committed failing against
  the pre-fix code (`bf12077`) before Addendum 9's fix (`9d9e0ec`) — the
  failing commit is the proof the defect existed, kept in history rather
  than folded into the fix as an invisible footnote.

## Consolidation output, and which backbone made it

TAVERN is abstractive by design: `--backbone` fuses **per event**, in induced
order, so chronology is a property of the loop. `union`/`extractive` need no
model; `ollama`/`instruct`/`bart`/`pegasus`/`primera` do. What's committed
under `consolidations/` is `ollama`/gemma3:4b on the `ancoragem` run (R-1
0.793, R-2 0.734, R-L 0.566, METEOR 0.477, 0/289 glued-word events) —
regenerate with `python main.py --tag t --backbone ollama --backbone-model
gemma3:4b && python scripts/make_curation.py outputs/t/curation.json
consolidations/t/`. `consolidations/curation.md`/`.csv` lay out, per event,
every source account beside its consolidation, with verse addresses, day
index, conflict flag, and a blank verdict for **faithful / complete /
placement** — no event is undetected, 2 are displaced across a day
boundary, 29 are transposed with a same-day neighbour (ancoragem).

## Where the results stand

**Primary run: `ancoragem-20260831`** (tag `ancoragem`, code commit `9d9e0ec`,
base config, Ollama gemma3:4b, `repeat_penalty=1.1`). **`canonical-20260827`
is retained as the "before" state** — the run that exposed the defect
Addendum 9 fixed, not a discarded result. Both tags stay citable; every
number below reports both where they differ.

Addendum 9's finding: the thesis's Achado 4 claims the TIMEX3 inventory's
normalisation and the anchor chains order the four documents, but the
normalisation never reached the coreference score, and a hand-picked
2-entity stop-list (`_UBIQUITOUS_ENTITIES`) stood in for entity
discrimination. `scripts/test_no_evidence_floor.py` proves it: two units
sharing zero predicate, zero distinguishing entity and zero anchor evidence
still cleared `MATCH_THRESHOLD` (0.25×1.0 + 0.10×1.0 = 0.35 ≥ 0.34) — failing
commit `bf12077`, fixed in `9d9e0ec`. The fix: `scaffold.project_timexes`
derives an absolute day/part per anchorable TIMEX3 from `FEAST_DAY`/
`DAYPART_POSITION` (never the chronology, never `WEEKDAY_ORDER`) and feeds
it into `PredicateIDF` as `D:{day}`/`P:{part}` terms; `_UBIQUITOUS_ENTITIES`
is replaced by `EntityIDF`, the same IDF construction applied to entities.

τ = 0.9274 (was 0.9155), pairwise 0.9637 (0.9577), coverage 0.8869/149/168
(0.8512/143/168), 289 clusters (249). All six consistency checks pass in
both. 96 inter-document conflicts (94), all three documented divergences
recovered in both. End-to-end ROUGE-L 0.566 (0.497, Ollama), against the
pre-registered 0.795 — **still not met**, closer than before, and Chapter 10
says so.

**τ has a floor and it changes the reading — and the floor did not move.**
N1 (positional interleaving, zero annotation) gets τ=0.8140 regardless of
configuration, since it's a property of the corpus and the null model, not
of Stage 3's scoring. 0.9274 closes 61.0% of the gap to the curated ceiling
(1.000), up from 54.6% — report both numbers, never τ alone.

**τ is protected by construction — measured, not asserted, unchanged by the
fix.** `removed_arcs` in the global tournament is 0 under either induced
(monotone-by-construction) clustering. Feed the oracle clustering into the
identical tournament and `removed_arcs = 614`, τ **falls** to 0.6296 in
both — this cross-check depends only on `global_timeline.induce()` and the
oracle clustering, neither touched by Addendum 9.

**Cluster purity moved, the ceiling did not.** 44.4% of multi-witness
clusters are pure (was 30.8%) — but the ceiling, verified against a perfect
oracle clustering, is still 89.5% (B-cubed F1 0.829): the oracle clustering
is built straight from the chronology, so this ceiling is a property of the
corpus's citation granularity, invariant to the coreference fix.
`local_timeline.segment` opens a unit boundary only between whole verses,
and Aschmann's harmonisation occasionally splits one verse between two
events (`"14:66-68a"` / `"14:68b"`). 26 verse keys, 40 events touched, 10
with no verse exclusive to them at all (3 of those 10 never win any verse at
all, invisible to any verse-keyed instrument, not just hard to individuate).
44.4/89.5 = 49.6% of ceiling now (was 34.4%).

**How much predicate evidence reaches the score — corrected twice, and the
second correction is the one that goes to the thesis.** First measurement
(27%, all-witness intersection) was too strict for 3–4-document clusters;
corrected to pairwise, 59.1% of clusters share *some* raw predicate/TIMEX3
term. But raw intersection is itself the wrong instrument — common verbs
(SAY, GO, COME) make almost any pair intersect regardless of discrimination,
which IDF weighting suppresses. `scripts/predicate_evidence_fraction.py`
reports the actual signal: the pairwise IDF-weighted cosine's 0.40-weighted
contribution, pooled over every cross-book pair. Median contribution 0.042
→ 0.101, fraction of pairs below 0.08 contribution 61.0% → 46.3%. Same
direction as predicted, not the same exact percentiles as any intermediate
estimate — report the instrument, not a copied number.

**Grouping and ordering are not separable — and the fix did not separate
them.** Crossing induced/oracle on each axis (A=both induced, B=oracle
grouping, C=oracle ordering, D=both oracle) — B and D depend only on the
oracle clustering and are therefore identical before/after: canonical
D−A=0.201, D−B=0.303, D−C=0.308, sum 0.611, residual 0.410; ancoragem
D−A=0.133 (closes markedly more of Stage 3's own gap), D−B=0.303 (same),
D−C=0.315, sum 0.618, residual 0.486 — *wider*, if anything. Fixing either
component alone still makes ROUGE-L worse than the fully induced baseline in
both configurations.

**Selection accuracy crossed from below chance to above it.** Canonical:
0.2973 over 74 matched events, floor (recomputed over that subset's own
version-count distribution) 0.3446 — below. Ancoragem: **0.3600 over 75**,
floor **0.3411** — above.

**The annotation still loses to a zero-annotation lexical baseline — closer,
not closed.** Isolating `score()` from the alignment algorithm: recall@1
0.405→**0.437** for the full score, lexical baseline unchanged at 0.513 (it
never touches `score()`). The fix closed 0.032 of a 0.108 gap. Per-term
ablation on the fixed score: `class` moved from contributing exactly 0.000
to net-harmful (+0.016 on removal); `modal` was already net-harmful and is
now more so (+0.041, was +0.019) — Addendum 9 touched neither term; their
apparent harm grew because the other terms got better calibrated. Not
reweighted — that would be tuning against the reference the ablation used.

**The absolute-day/within-day projection is only partly populated, and this
is future work, not a bug.** 42/112 anchorable TIMEX3 get a concrete day
(37.5%), 46/112 a concrete part (41.1%). The entire ancoragem gain came
through a mechanism barely a third populated. Do not close this now — doing
so after measuring ancoragem's result would be tuning against the
evaluation that reported it.

Two ablations contradict the thesis's predictions and are reported, not
hidden, in both configurations.

The cascade does not move τ at all — but **not** because the relations are
uninformative. ISO-TimeML defines no cross-document relation, so the merge cannot
run through `<TLINK>`s on any implementation; it runs through the `<TIMEX3>`
normalisation and anchor chains, i.e. the scaffold, which is the one component
whose removal degrades τ, coverage and ROUGE-L together in both configurations.
Within a document, the canonical run's own count (371 relational constraints
between clusters agree with the narrative order, 0 contradict) was not
recomputed against ancoragem's 289 clusters — only the oracle decomposition
was re-run — but the finding (the Evangelists narrate in order, so the links
never need to correct anything here) is not expected to flip. Do not restate
this as "the annotation does not help".

Removing the veridicality partition slightly raises τ in both configurations,
so it is a correctness requirement (check 5) rather than an accuracy gain.

**The reference is itself a selection, and the gap widens as the fusion
improves.** It covers 88.0% of the sources' content-word vocabulary
(unchanged); the abstractive consolidation covers 96.4% now (was 95.1%).
Every reference-based metric therefore penalises a fusion for material the
reference doesn't contain — the strongest form of the thesis's second threat
to validity, measured rather than argued, and it gets *worse*, not better,
as Stage 3 improves.

**`repeat_penalty` is backend-specific, and reusing the HuggingFace value was
a real bug, not a tuning target.** llama.cpp's `repeat_penalty` (what Ollama
exposes) and HuggingFace's `repetition_penalty` share a name, not a scale;
1.5 (correct for the HF backbones, thesis Chapter 8's fixed decoding
controls) pushed gemma3:4b to glue words together
(`...toBethphegeon theMountofOlves...`, 3/249 events). Fixed to
`OLLAMA_REPEAT_PENALTY=1.1` for Ollama only; end-to-end ROUGE-L moved 0.331 →
0.497 (canonical) on that fix alone. `scripts/check_text_quality.py` guards
the regression (0/249 canonical, 0/289 ancoragem corrupted).

## Known, unfixed, and staying that way

Fixing any of these now would invalidate the `ancoragem` run and force
re-reconciling the thesis; they are findings, not open bugs to close reflexively.

- The half-verse/whole-verse granularity mismatch above (purity's 89.5%
  ceiling, and ~0.031 of ROUGE-L against the curated timeline's own ceiling)
  — a corpus property, unchanged by Addendum 9. Fixing it means a half-verse
  key through `Corpus`/`ReferenceParser`/`Chronology`, used everywhere.
- **The absolute-day/within-day projection is only ~38–41% populated**
  (42/112 day, 46/112 part) — the mechanism Addendum 9 added, not extended
  further, per the ablations/analysis above.
- `predicate_similarity`'s cosine has no length compensation for a short,
  terse account competing against a longer, richer wrong candidate.
- `class_agreement`'s and `modal_compatibility`'s weights, per the per-term
  ablation above — neither touched by Addendum 9, both measurably more
  harmful after it.
- The fusion cache is global now (keyed by backbone/model/`repeat_penalty`/
  the exact texts, `outputs/fusion_cache.jsonl`, not per-tag) — this one
  *was* fixed, because it changed nothing about what gets measured, only how
  much redundant regeneration an ablation grid does.

## The one rule

The chronology and the Golden Sample are Stage 6 only.
`config.assert_no_chronology_import()` enforces it by inspecting the call stack.
Do not add an import path from an earlier stage, and do not use the harmony's
event descriptions or verse references anywhere in stages 1–5. The "oracle
timeline" configuration (perfect grouping/ordering, used above as a
calibration ceiling) lives in `scripts/oracle_roundtrip.py` and
`scripts/oracle_decomposition.py`, Stage 6 scripts, for exactly this reason —
not a `TavernConfig` flag. `TavernConfig.use_oracle_timeline` existed once;
it was declared and never read, and has been removed rather than wired into
`pipeline.py`, which the guard would (correctly) have refused.
