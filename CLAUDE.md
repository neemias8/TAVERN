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

## Where the results stand

**Canonical run: `canonical-20260827`** (tag `canonical`, base config
— `no_anchor_credit=False`, the published default — Ollama gemma3:4b,
`repeat_penalty=1.1`). This is the single source; everything below is
measured on it or diagnostically against it, not a separate run.

τ = 0.9155, pairwise 0.9577, coverage 0.8512 (143/168), 249 clusters. All six
consistency checks pass. 94 inter-document conflicts, all three documented
divergences recovered. End-to-end ROUGE-L 0.497 (Ollama, `repeat_penalty`
fixed from 1.5 to 1.1 — see below), against the pre-registered 0.795 —
**not met**, and Chapter 10 says so.

**τ has a floor and it changes the reading.** N1 (positional interleaving,
zero annotation) gets τ=0.8140. So 0.9155 closes 54.6% of the gap to the
curated ceiling (1.000), not "92% of perfect" — report both numbers, never τ
alone.

**τ is protected by construction — measured, not asserted.** `removed_arcs`
in the global tournament is 0 under the induced (monotone-by-construction)
clustering. Feed the oracle clustering into the identical tournament and
`removed_arcs = 614`, τ **falls** to 0.6296. The induced grouping's
monotonicity keeps real cross-document ordering disagreements from ever
reaching the tournament; τ is high partly because of that, not despite it.

**Cluster purity has the same shape.** 30.8% of multi-witness clusters are
pure — but the ceiling, verified against a perfect oracle clustering, is
89.5%, not 100%: `local_timeline.segment` opens a unit boundary only between
whole verses, and Aschmann's harmonisation occasionally splits one verse
between two events (`"14:66-68a"` / `"14:68b"`). 26 verse keys, 40 events
touched, 10 with no verse exclusive to them at all. 30.8/89.5 = 34.4% of
ceiling is the number to report, not 30.8/100.

**Grouping and ordering are not separable.** Crossing induced/oracle on each
axis (A=both induced, B=oracle grouping, C=oracle ordering, D=both oracle):
D−B (0.303) + D−C (0.308) = 0.611, against the real D−A of 0.201 — a residual
of 0.410. Fixing either component alone makes ROUGE-L *worse* than the fully
induced baseline; only fixing both together helps.

**Selection accuracy is below chance, against the right floor.** 0.2973
induced, over 74 events (the ones the induced clustering actually gives a
matched ≥2-document cluster for) — not the 95 the published analytical floor
(0.3474) is computed over. Recomputed over the same 74, the floor is 0.3446.
0.2973 < 0.3446.

**The annotation loses to a zero-annotation lexical baseline.** Isolating
`score()` from the alignment algorithm (curated segmentation given for free,
no threshold, no band): recall@1 is 0.405 for the full
predicate+participants+anchor+modal+class score, **0.513 for raw content-word
cosine over the verse text**. Per-term ablation: `class` contributes exactly
0.000; `modal` is net-harmful (+0.019 recall@1 when removed). Not reweighted
— that would be tuning against the reference the ablation used.

Two ablations contradict the thesis's predictions and are reported, not hidden.

The cascade does not move τ at all — but **not** because the relations are
uninformative. ISO-TimeML defines no cross-document relation, so the merge cannot
run through `<TLINK>`s on any implementation; it runs through the `<TIMEX3>`
normalisation and anchor chains, i.e. the scaffold, which is the one component
whose removal degrades τ, coverage and ROUGE-L together. And within a document
all **371** relational constraints between clusters agree with the narrative
order, **0** contradict — the Evangelists narrate in order, so the links have
nothing to correct here. Do not restate this as "the annotation does not help".

Removing the veridicality partition slightly raises τ, so it is a correctness
requirement (check 5) rather than an accuracy gain.

**The reference is itself a selection.** It covers 88.0% of the sources'
content-word vocabulary; the abstractive consolidation covers 95.3%. Every
reference-based metric therefore penalises a fusion for material the
reference doesn't contain — the strongest form of the thesis's second threat
to validity, measured rather than argued.

**`repeat_penalty` is backend-specific, and reusing the HuggingFace value was
a real bug, not a tuning target.** llama.cpp's `repeat_penalty` (what Ollama
exposes) and HuggingFace's `repetition_penalty` share a name, not a scale;
1.5 (correct for the HF backbones, thesis Chapter 8's fixed decoding
controls) pushed gemma3:4b to glue words together
(`...toBethphegeon theMountofOlves...`, 3/249 events). Fixed to
`OLLAMA_REPEAT_PENALTY=1.1` for Ollama only; end-to-end ROUGE-L moved 0.331 →
0.497 on that fix alone. `scripts/check_text_quality.py` guards the
regression.

## Known, unfixed, and staying that way

Fixing any of these now would invalidate the canonical run and force
re-reconciling the thesis; they are findings, not open bugs to close reflexively.

- The half-verse/whole-verse granularity mismatch above (purity's 89.5%
  ceiling, and ~0.031 of ROUGE-L against the curated timeline's own ceiling).
  Fixing it means a half-verse key through `Corpus`/`ReferenceParser`/
  `Chronology`, used everywhere.
- `predicate_similarity`'s cosine has no length compensation for a short,
  terse account competing against a longer, richer wrong candidate.
- `class_agreement`'s and `modal_compatibility`'s weights, per the per-term
  ablation above.
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
