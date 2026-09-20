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
      fidelity      keep every detail / invent none / state each once, measured
                    against the accounts alone -- hence Stage 5, not Stage 6
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
- **`py_heideltime` on Windows silently returns nothing.** Its own
  `_write_config_props()` writes the TreeTagger path with backslashes into a
  Java `.properties` file, which eats them as escape sequences, so
  `C:\ProgramData\anaconda3\...` is read back with no separators at all and
  TreeTagger's path resolution breaks — no exception, `stderr` is captured
  and discarded by the library itself. `tools/heideltime_agreement.py`
  monkeypatches the function (forward slashes, Windows accepts them) rather
  than touching the installed package; also needs `pip install emoji`
  (undeclared dependency) and must run per chapter, not per verse (each
  call starts a JVM) — see that file's module docstring for the rest.
- **A two-sided requirement needs a guard on both sides, or its gradient
  points at the failure it exists to prevent.** "Keep every detail" and
  "state it once" are one requirement; a check on recall alone makes
  concatenation the optimal answer, and a check on brevity alone makes
  dropping a witness optimal. `fidelity.check` measures three axes at once
  for that reason, and the third — repetition — is compared against the
  accounts' own repetition rather than absolutely, since this corpus
  narrates a command and then its execution in the same words. Verified the
  hard way: the two-axis version had gemma3:4b answer a dropped-detail
  rejection by pasting three accounts end to end at recall 0.91, passing.
- **A cache key must digest everything that determines the output, and a
  docstring saying so is not a test.** `CachedFuser._key` covered the
  backbone, the model, `repeat_penalty` and the accounts, claimed in prose
  to cover the prompt, and did not — so the first prompt edit would have
  replayed the old generations under a new label. `PROMPT_VERSION` is now
  in the key. Bump it whenever the prompt text changes.

## Consolidation output, and which backbone made it

TAVERN is abstractive by design: `--backbone` fuses **per event**, in induced
order, so chronology is a property of the loop. `union`/`extractive` need no
model; `ollama`/`instruct`/`bart`/`pegasus`/`primera` do. What's committed
under `consolidations/` is `ollama`/gemma4:26b on the `ancoragem` run (R-1
0.824, R-2 0.800, R-L 0.622, METEOR 0.535, 0/289 glued-word events, digest
`c903b9fd`) — regenerate with `python main.py --tag t --backbone ollama
--backbone-model gemma4:26b && python scripts/make_curation.py
outputs/t/curation.json consolidations/t/`. `consolidations/curation.md`/`.csv`
lay out, per event, every source account beside its consolidation, with verse
addresses, day index, conflict flag, the **fusion path**, and a blank verdict
for **faithful / complete / placement** — no event is undetected, 2 are
displaced across a day boundary, 29 are transposed with a same-day neighbour
(ancoragem).

**Read the `fusion` column before calling a paragraph abstractive.** Of the
289, 134 are single-account events emitted verbatim; of the 155
multi-account ones, 142 are the backbone's first attempt, 9 its strict
re-ask, and **4 are `UnionFuser`'s deterministic output** after the
backbone failed the guard twice. The configuration is 97.4% abstractive by
event, not 100%, and the artifact says so per event rather than leaving it
to the backbone's name. On gemma3:4b the same three counts were 107 / 18 /
**30**, i.e. 80.7%: the fallback rate is a property of the model, and it is
the cheapest single indicator of whether a backbone can do this task.

**Regenerate `consolidations/` and `human_eval/` from ONE run, and record
the digest.** Stage 4's GNN is not reproducible across runs (§9.6), so a
second `pipeline.run` of the identical config selects a different account on
a handful of clusters, which changes the fusion inputs, which changes the
text. Measured here: two runs of the same config gave digests `63d19e65` and
`ab6fa2e4`, R-L 0.6045 vs 0.6043, content coverage 0.9864 vs 0.9870.
`scripts/verify_for_thesis.py` prints the per-event digest; if
`consolidations/`, `human_eval/` and `outputs/ancoragem/` do not share it,
they are three different runs and the human evaluation is not of the
committed artifact.

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
recovered in both. End-to-end ROUGE-L **0.622** on gemma4:26b (0.604 on gemma3:4b after
Addendum 18's Stage 5 fixes, 0.566 before them, 0.497 canonical), against the
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
improves.** It covers 87.5% of the sources' content-word vocabulary
(unchanged); the abstractive consolidation covers **99.1%** now (98.7% on
gemma3:4b after Addendum 18, 96.4% before it, 95.1% canonical), against the extractive configuration's
82.3% — `scripts/verify_for_thesis.py`'s fixed definition, 1,693 types.
Every reference-based metric therefore penalises a fusion for material the
reference doesn't contain — the strongest form of the thesis's second threat
to validity, measured rather than argued, and it gets *worse*, not better,
both as Stage 3 improves and as the fusion stops dropping detail.

**`repeat_penalty` is backend-specific, and reusing the HuggingFace value was
a real bug, not a tuning target.** llama.cpp's `repeat_penalty` (what Ollama
exposes) and HuggingFace's `repetition_penalty` share a name, not a scale;
1.5 (correct for the HF backbones, thesis Chapter 8's fixed decoding
controls) pushed gemma3:4b to glue words together
(`...toBethphegeon theMountofOlves...`, 3/249 events). Fixed to
`OLLAMA_REPEAT_PENALTY=1.1` for Ollama only; end-to-end ROUGE-L moved 0.331 →
0.497 (canonical) on that fix alone. `scripts/check_text_quality.py` guards
the regression (0/249 canonical, 0/289 ancoragem corrupted).

**Addendum 16: 155 `<TLINK>` were conformance violations, fixed, and the
`ancoragem` artifacts were replaced in place.** `IS_INCLUDED` is reserved by
the norm (thesis Section 2.3.8) for event-time; `closure.py`'s reverse Allen
map emitted it for event-event relations too, since the closure network's
nodes are always events but the map didn't know that. 155 event-event
`IS_INCLUDED` → `DURING` (0 → 151), plus 4 inverted event-time links found
along the way ("time IS_INCLUDED event", backwards). All four Gospels now
pass 12/12. Verified as a null change *before* touching anything: a
pre-registered diff against `ancoragem-20260831`'s own `results.json` found
differences only in two GNN-dependent ablation rows, and a **control** (two
independent runs of the fixed code, same config, diffed against each
other) reproduced noise of the same or larger magnitude in those same
rows — confirming the variation is Stage 4's own run-to-run GNN noise, not
the fix. The tag `ancoragem-conformance-20260911` points at the fixing
commit; `ancoragem-20260831` is untouched and stays citable as the
pre-fix state. `outputs/ancoragem/` itself (gitignored, not the git tag)
was overwritten with the corrected run's artifacts — byte-identical
`consolidated.txt` (fusion-cache hit), identical τ/coverage/clusters/
purity/B-cubed/end-to-end R-1-R-2-R-L-METEOR to the pre-fix numbers
already in this file and the README.

**The control also gives thesis Section 9.6 a number it's currently
missing.** §9.6 says GNN-involving figures are "reported as the mean over a
stated number of seeded runs" — true (`mean_over_seeds` does average
`cfg.seeds = (13, 42, 1337)`), but two full re-runs of the identical
config+seeds still don't reproduce each other: ROUGE-family metrics moved
by up to ~0.006, selection accuracy by up to one event out of 75 (0.0133),
purely from `index_reduce_(reduce="amax")`'s documented CPU
non-determinism, which the seed does not control. Exactly reproducible
across both control runs: Kendall's τ, coverage, cluster count, and the
`- graph propagation` ablation (no GNN training at all) — confirming the
noise is specifically Stage 4's, not a general property of the pipeline.
The abstractive end-to-end headline row (what Chapter 10 actually reports)
was also exactly reproducible both times, via the same fusion-cache hit
that keeps `consolidated.txt` byte-identical. §9.6 needs a sentence saying
seeding does not eliminate this — the exact figures above are the
envelope to cite.

**Addendum 18: the fusion was neither keeping every detail nor removing
redundancy, and nothing measured either.** Stage 5 had one instrument,
`text_quality.is_glued`, which catches a decoding artefact. Both sides of
the task's own premise (§8.1) were unmeasured, and an audit of the
`ancoragem` artifacts found the premise violated on **104 of 289 events**:
52 dropped a detail the accounts carry (E002 loses Matthew's donkey, the
prophecy and the cloaks), 49 stated material no account contains, 26
restated their own wording. The inventions were not subtle — E068 grafts an
aviation accident onto "Pray that your flight will not take place in
winter", E165 fabricates two witnesses complete with "Account 2:" headers.
ROUGE does not punish any of this, which is why it survived four addenda.

Four causes, all in Stage 5, all mechanical:

- **134 of 289 events have one account**, where there is nothing to fuse,
  and `consolidate` called the generator anyway — on 33 of them the model
  continued past the source. 37 of those 134 were additionally told "these
  accounts DISAGREE", there being one.
- **The prompt stated two objectives in tension** ("keep every detail ... do
  not omit" against "state it once") with no method and no example. A 4B
  model under greedy decoding reconciles them by splicing.
- **`num_predict` was `DECODING["max_new_tokens"]`**, a HuggingFace constant
  passed to llama.cpp — truncating 9 events mid-word and leaving 250 tokens
  of room on a one-line account.
- **`CachedFuser._key` did not digest the prompt**, though its docstring
  claimed it did. Editing the prompt would have silently replayed the old
  generations. `PROMPT_VERSION` is now in the key.

The fixes: single-account short-circuit in `consolidate`; `build_prompt`
stating the method, one worked merge, and the prohibitions; a per-cluster
`num_predict` with stop sequences; and `fidelity.check` + `RepairingFuser` —
three axes (detail recall, novel content, internal repetition against the
sources' own), one strict re-ask naming the failure, then `UnionFuser`.

**The third axis is the load-bearing one.** Built with two (recall and
invention) the guard's gradient points at the defect it exists to stop:
told it had dropped detail, gemma3:4b answered by pasting all three accounts
of E002 end to end — 309 tokens, recall 0.91, passing. Redundancy is
compared against the accounts' own repetition, never absolutely: this corpus
narrates a command and then its execution in the same words.

Result, same tag, Stage 3 untouched and verified so — τ 0.9274, pairwise
0.9637, coverage 0.8869, 289 clusters, `removed_arcs` 0, 96 conflicts, all
to the digit. 286 of 289 texts changed. **104 → 11 premise failures, and
all 11 are the union fallback's own repetition**: the abstractive path is
0/259. R-1 0.793→0.807, R-2 0.734→0.782, R-L 0.566→0.604, METEOR
0.477→0.525; content coverage 93.3%→98.3% overall. 50 min cold on 8 CPU
threads, 194 model calls instead of 289.

**What it did NOT buy is compression, and that is the finding.** Verbatim
copy *rose* (0.897→0.979 median over multi-source events), and the fused
paragraph is at the median exactly the length of `UnionFuser`'s output for
the same accounts — only 37 of 155 come in below 0.95 of it. By the
cross-source instrument, events with *some* residual restatement went
27.7%→37.4% even as the severe ones fell 28→15. gemma3:4b now retains
faithfully and merges rarely; where the overlap is lexically literal the
guard catches it and the re-ask fixes it, where the Evangelists say the same
thing in different words neither fires. That is a capacity limit, not a
prompt one — it is what a larger local model would be bought for, and the
measurement to compare it against is in `scripts/check_text_quality.py`.

Two protocol consequences, both in `scripts/make_human_eval.py`:

- **The pre-registered control items were broken and nobody could have
  noticed.** §9.3.5's 4 controls are single-account events where all three
  conditions are "identical by construction"; before this fix **133 of 134
  differed**, so a control was three-way distinguishable and measured
  nothing. Check `controls_by_fusion_path` reads `{"single": N}`.
- **The abstractive condition is not uniformly the backbone** — 9 of the 32
  sampled comparison items are union fallbacks, over-represented against
  the population's 19% because a fallback is more likely to differ from the
  other two conditions. The key now carries `fusion` per item; report the
  condition with and without them.

**Addendum 19: the prompt was not the constraint; the model was. And the
bigger model is not slower.** Addendum 18 fixed faithfulness and left
redundancy untouched — which nothing noticed, because
`internal_redundancy` compares 5-grams and the fusion's repetition is
paraphrastic. In E060 the same prophecy arrives three times, once per
Evangelist: "not one stone here will be left on another; every one will be
thrown down" beside "not one stone will be left on another; every one of
them will be thrown down" — Jaccard 1.00 over content terms, almost no
shared 5-gram. The event passed at 0.24 against a 0.25 threshold while being
unreadable. `fidelity.excess_repetition` is the missing instrument: content
terms the fusion states beyond the ceiling any single account sets. By it,
gemma3:4b's fusion (median 0.115) was **indistinguishable from `UnionFuser`**
(0.125) — the model had never been merging at all.

Three redrafted prompts (`scripts/fusion_prompt_ab.py`, kept) were measured
against the production one over the eight worst episodes. Every one bought
less repetition with lost detail and invention: the most aggressive dropped
excess repetition to 0.285 but recall from 1.00 to 0.74 and pushed invented
content from 0.03 to 0.20. `UnionFuser`, which has no prompt, repeated less
than any of them. **Do not redraft the prompt again without re-running that
grid** — it is seven minutes and it has already answered this question once.

Changing only the backbone, prompt untouched: **gemma4:26b** gives excess
repetition 0.067 (over all 155 multi-source events, against 0.115), events
above 0.10 down 87 → 63, events compressing below `UnionFuser`'s length
37 → 69, union fallbacks **30 → 4** (97.4% abstractive by event, against
80.7%), R-L 0.604 → 0.622, content coverage 98.7% → 99.1%. One event now
loses detail where none did, which is the price of real compression and is
reported rather than tuned away.

Two things about that model that the arithmetic gets wrong:

- **It is not slower.** 15.65 tok/s against gemma3:4b's 16.3, because it is
  sparse: an 18 GB file that loads 9.5 GB resident. The dense estimate
  (bandwidth ÷ weight bytes) predicted ~5x slower and was simply the wrong
  model of the machine. A full generation pass costs about what the 4b cost.
- **It is a reasoning model, and `/api/generate` discards its output.** It
  spends the whole `num_predict` budget in a `thinking` field the endpoint
  drops, returning `response: ""` with `done_reason: "length"` and no error.
  Every event would have come back empty, failed the guard twice and landed
  in the union fallback — a run that looks mediocre rather than broken.
  `OllamaFuser` now sends `"think": False`. Check this first for any new
  backbone: an empty-response rate is the symptom.

Still open, and deliberately not closed here: `excess_repetition` is
implemented but **not wired into `check()`**, because doing so changes which
events fall back, and `MIN_DETAIL_RECALL = 0.88` was calibrated against a
model that copied. A model that genuinely compresses dips below it on hard
episodes. Neither should be adjusted to make the current backbone pass.

`scripts/suspect_positions.py` is new and unrelated in cause: §9.3.5's Part
B stratifies 30 adjacent pairs half-and-half between the monotone
subsequence and suspected transpositions, `make_human_eval.py --suspect`
consumed that list, and **nothing produced it** — `error_analysis.analyse`
counts the 31 inversions (29 same-day, 2 across a day boundary) and keeps
one example string, discarding which clusters they were. Without it Part B
came out `{"monotone": 30}` and the stratum-weighted estimate — the only
figure comparable with the pairwise τ — could not be computed at all.

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
  `PROMPT_VERSION`/the repair pass/the exact texts,
  `outputs/fusion_cache.jsonl`, not per-tag) — this one *was* fixed, because
  it changed nothing about what gets measured, only how much redundant
  regeneration an ablation grid does.
- **gemma3:4b retains rather than merges** (Addendum 18): median fusion
  length equals `UnionFuser`'s, 37/155 compress below 0.95 of it, and 30 of
  155 the backbone could not fuse at all. Do not close this by reweighting
  the guard or by re-tuning the prompt against the events it fails — that is
  tuning against the instrument that reported it. It is a capacity question,
  and the honest way to answer it is a larger local model measured with the
  same guard. Benchmarked on this machine (8 CPU threads, no GPU offload,
  100% CPU): gemma3:4b decodes at 16.3 tok/s, a 26–27B at Q4 would be ~5x
  slower, so ~3 h per generation pass against 50 min — feasible once for the
  headline run, not inside the ablation grid.
- **`class_agreement` and `modal_compatibility` were not touched by Addendum
  18 either.** The per-term ablation predates the fusion fixes and was not
  re-run against them; nothing in Stage 5 can move a Stage 3 score, but the
  numbers cited are canonical/ancoragem-pre-18 and should be labelled as
  such if quoted next to the new ROUGE figures.

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
