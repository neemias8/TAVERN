# Data provenance

Every experiment runs on the files in `data/`, identified by content digest.
`tavern.config.verify_corpus()` checks them and raises on mismatch; the check
runs at the start of `run_experiments.py`.

| File | MD5 | Role |
|---|---|---|
| `ChronologyOfTheFourGospels_PW.xml` | `23cf52ee7c597b2aa190dc243d13f969` | Aschmann harmony. **Held out.** Reachable only from `tavern/stage6_evaluation/`. |
| `EnglishNIVMatthew40_PW.xml` | `8dea0ab56468c26dcb45c3160148c837` | Matthew 21–28, 389 verses |
| `EnglishNIVMark41_PW.xml` | `003ba2b7cf7ca1b1af959f5352056ca8` | Mark 11–16, 253 verses |
| `EnglishNIVLuke42_PW.xml` | `0afe52402eb56d591406209c343ea4fb` | Luke 19–24, 285 verses |
| `EnglishNIVJohn43_PW.xml` | `2c7f59099ba90ef34666d23c6c480e6a` | John 12–20, 318 verses |
| `Golden_Sample.txt` | `dca94bbd4697c381c5d0f6859a658142` | Reference consolidation, 84,659 chars. **Stage 6 only.** |

Derived, not digest-pinned: `NIV_*_PW_with_pericopes.xml` supply the pericope
layer only. Verse text always comes from the digest-pinned files.

## Corpus facts, as measured

- 1,245 verses, 25,893 words, 91 pericopes
- 169 canonical events; 363 citations, **362 resolvable versions**
- Version distribution: 1 event with none, 73 with one, 20 with two, 51 with
  three, 24 with four → 95 contested
- Analytical random floor for selection: 0.3474 over 95 events

## Two defects in the input, handled in code

1. **The pericope files stop short of the end of each book** (1,179 verses
   against 1,245). `stage1_preprocessing/pericopes.py` extends the final
   pericope of each Gospel to the last verse present in the verse files and
   reports the repair. Mark 16:9–20 and Luke 24:13–53 are cited by the
   chronology, so without this the resurrection events would lose all sources.

2. **Event #28 cites Luke 13:34–35**, outside the Luke 19–24 scope. The
   citation does not resolve; the event keeps one version instead of two. The
   scope is not widened, because the passage is a parallel to Matthew 23:37–39
   spoken on another occasion.

## A known ambiguity: half-verse citations

Aschmann's harmonisation occasionally individuates events below verse
granularity — one verse split between two adjacent events, e.g. Mark 14 cites
`"66-68a"` for one event and `"68b"` for the next. The corpus's own verse
addressing (`book:chapter:verse`) has no key for "half a verse": `Corpus`'s
text extraction (`ReferenceParser`/`VerseSplitter`) can select the right
*text* for each half, but the verse *key* itself — what
`Chronology.verse_keys` and everything built on it uses to say "this belongs
to event N" — is necessarily whole-verse. Two adjacent events that split a
verse this way both carry that verse's key.

Measured: **26 verse keys are claimed by two curated events**, touching **40**
of the 169 curated events. **10 of those 40 have no verse exclusive to them
at all** (events 47, 93, 99, 119, 132, 133, 155, 156, 157, 163) and so cannot
be individuated as distinct objects under any clustering, however correct —
11 counting event 53, which has no citation in any book at all and is
already excluded from every coverage denominator in the codebase. (An
earlier estimate of 13 for this count, from an independent recomputation
outside this codebase, was not reconciled against a definition; a second
independent recomputation — the one this section now reports — landed on
the identical list of 10 above, so 10 is what `scripts/cluster_purity.py`
and `outputs/canonical/results.json` carry.)

A finer distinction inside those 10: **3 of them (47, 93, 157) never win
even one verse** — every verse each cites is also cited by an earlier-listed
competing event, so `verse_to_event`'s `setdefault` never assigns any verse
to them at all. Together with event 53, **4 of the 168 citable curated
events are invisible to any verse-keyed instrument** (never a `bcubed`
gold cluster, never a purity `modal` winner) — not merely hard to
individuate, absent from the map entirely. This is why `bcubed`'s
`gold_clusters` reads 165, not 168.

This is a property of the resource, verified against a perfect (oracle)
clustering rather than assumed: cluster purity and B-cubed cap out at ~89.5%,
not 100%, for this reason alone. It is not fixed in code — doing so would
mean a half-verse key threaded through `Corpus`/`ReferenceParser`/
`Chronology`, used by every stage, for a gain in reporting precision rather
than in the system being measured. `scripts/cluster_purity.py`'s module
docstring carries the same note next to the code that measures it.

## The held-out guarantee

`config.assert_no_chronology_import()` inspects the call stack and raises if a
chronology load originates in `stage1_`…`stage5_`. It is called by
`stage6_evaluation/chronology.py:load`, the only reader of that file.
