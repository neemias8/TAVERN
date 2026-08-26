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
- **ROUGE-L needs the compiled path.** `content_metrics.rouge` delegates the
  subsequence to `pylcs`; `verify_fast_path` asserts it equals `rouge_score`
  exactly. The pure-Python table takes minutes on a 16k-token reference.
- **Selection is over documents, emitting a contiguous span.** A canonical event
  cites a verse range per Gospel; selecting per unit fragments the account and
  costs ~0.1 of ROUGE-L.
- Allen's composition table is generated from the endpoint algebra, not typed in.

## Where the results stand

τ = 0.9163, coverage 0.8512. All six consistency checks pass. 91 inter-document
conflicts, all three documented divergences recovered. Downstream ROUGE-L 0.640
(induced ordering, curated segmentation) and 0.595 (end to end), against the
pre-registered 0.795 — **not met**, and Chapter 10 says so.

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

## The one rule

The chronology and the Golden Sample are Stage 6 only.
`config.assert_no_chronology_import()` enforces it by inspecting the call stack.
Do not add an import path from an earlier stage, and do not use the harmony's
event descriptions or verse references anywhere in stages 1–5.
