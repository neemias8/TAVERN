# Consolidations — system output for expert curation

The consolidated narrative TAVERN produces, laid out so that a human expert can
judge it. This is the **`ancoragem`** run's own output (tag `ancoragem`,
`ollama`/gemma4:26b, `repeat_penalty=1.1`), the primary configuration — not a
placeholder, and not `canonical` (kept as the pre-fix "before" state; see the
repository `README.md`).

`ancoragem` keeps its name across the Stage 5 fusion fixes and the change of
backbone: the pre-fix artefacts are the ones committed here up to
`ancoragem-20260831`, recoverable from git, and this directory holds the
current run's. Unlike the Addendum 16 conformance fix, which was verified as a
null change before it was applied, these change the text — the fixes altered
286 of 289 consolidations on gemma3:4b, and moving to gemma4:26b altered them
again. The per-event digest in `scripts/verify_for_thesis.py` is what ties
this directory, `human_eval/` and `outputs/ancoragem/` to one run; it is
currently `c903b9fd`. Regenerate with:

```bash
python run_experiments.py --all --tag ancoragem --backbone ollama --backbone-model gemma4:26b --ollama-repeat-penalty 1.1
python scripts/make_curation.py outputs/ancoragem/curation.json consolidations/
```

| File | What it is |
|---|---|
| `consolidated.txt` | the narrative on its own, to read straight through |
| `curation.md` | one section per event: every source account beside the consolidation derived from it, with verse addresses and a blank verdict block |
| `curation.csv` | the same, one row per event, for a spreadsheet |

## What is being asked of a curator

Consolidation quality is not judgeable from the finished narrative alone. To say
whether an event's paragraph is good you have to see what went into it, and the
judgement splits into three that can come apart:

- **Faithful** — does the paragraph assert anything the sources do not? A fusion
  that invents a place, a number or a motive is unfaithful even if it reads well.
- **Complete** — is any detail present in *any* account missing from the fusion?
  This is the objective the extractive configuration cannot satisfy by
  construction, since it emits one account and discards the others.
- **Placement** — is the event in the right position, and on the right day?
  Ordering errors belong to Stage 3 and are attributable to it, because the
  generation loop cannot reorder anything.

Keeping them apart matters because the system fails at them unevenly: no
canonical event is undetected, two are displaced across a day boundary, and
29 are transposed with a neighbour on the same day.

## Which backbone produced this

Stated on the first line of `curation.md`, and it changes what the artefact is
worth. **This is `ollama` / gemma4:26b, on the `ancoragem` run** — genuinely
abstractive per-event fusion following the method of the IJCNN work, with
`repeat_penalty=1.1` (llama.cpp's own semantics, not HuggingFace's — see
`stage5_generation/backbones.py`), over the coreference score Addendum 9
fixed (the TIMEX3 normalisation and the anchor chains now actually reach it).
Earlier revisions of this repository committed `union`, then `canonical`
(pre-fix) here; both are superseded as the committed artefact, not deleted.

- **`ollama` (committed, ancoragem)** — abstractive per-event fusion. Judge
  it on Faithful, Complete and Placement; end-to-end against the held-out
  reference it scores R-1 0.8244 / R-2 0.8002 / R-L 0.6224 / METEOR 0.5347
  (on gemma3:4b with the same code: R-1 0.8074 / R-2 0.7823 / R-L 0.6045 /
  METEOR 0.5248; before the Stage 5 fusion fixes: R-1 0.7930 / R-2 0.7340 /
  R-L 0.5656 / METEOR 0.4773). Regenerate with the command at the top of this
  file. 0 of 289 events show word-gluing corruption, and 0 of the 285 events
  the backbone itself produced lose a detail the accounts carry or state
  material no account carries — both checked by
  `scripts/check_text_quality.py`. Before the fixes, 104 of 289 failed that
  second check: 52 dropped detail, 49 invented it (one grafted an aviation
  accident onto Matthew 24:20), 26 restated themselves.

  **Not every paragraph is the model's.** Of the 289, 134 are single-account
  events emitted verbatim — with one witness there is nothing to fuse, and
  the account is the only faithful answer. Of the 155 multi-account events,
  142 are the backbone's first attempt, 9 its strict re-ask after the
  faithfulness guard rejected the first, and **4 are `UnionFuser`'s
  deterministic output**, the backbone having failed twice. Each event's
  `curation.csv` row states which, under `fusion`; a curator weighing "is
  this abstractive" should read that column rather than the backbone name.
  The abstractive fraction of multi-account events is 0.974. On gemma3:4b,
  with the identical prompt and guard, it was 0.813 — how often a backbone
  falls back is the cheapest single indicator of whether it can do this task.
- **`union`** — deterministic, no model, kept only as a reference floor: every
  sentence from every account survives unless another already covers it, so it
  is detail-preserving but not abstractive — the seams between accounts stay
  visible and the prose does not flow. Regenerate with `--tag union --backbone
  union` if you want it back.
- **`instruct` / `bart` / `pegasus` / `primera`** — the other backbones the
  framework supports; not what is committed here.

TAVERN is an abstractive framework by design: the whole point of fusing per
event rather than selecting per event is that complementary details from every
version reach the consolidation in continuous prose. The committed artefact
now demonstrates that, rather than standing in for it.

## Where the ordering comes from

The order of events in these files was **induced from the text**. The Aschmann
harmony is held out and is read only by the evaluation stage; the day indices
shown come from the annotation's own temporal expressions and anchor chains, not
from any harmony. Kendall's τ against the held-out harmony is 0.9274 at coverage
0.8869 (149/168 clusters registered to a day) — up from 0.9155/0.8512 (143/168)
on the pre-fix `canonical` run.

Conflicted events are flagged. Where the sources were found to disagree under
closure, the flag means the framework detected it — the three divergences the
harmonisation literature documents (the fig tree, the Passover day, the
cockcrow) are all among the 96 reported (94 on `canonical`).
