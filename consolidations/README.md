# Consolidations — system output for expert curation

The consolidated narrative TAVERN produces, laid out so that a human expert can
judge it. This is the canonical run's own output (tag `canonical`,
`ollama`/gemma3:4b, `repeat_penalty=1.1`, commit-tagged `canonical-20260827`),
not a placeholder. Regenerate with:

```bash
python run_experiments.py --all --tag canonical --backbone ollama --backbone-model gemma3:4b --ollama-repeat-penalty 1.1
python scripts/make_curation.py outputs/canonical/curation.json consolidations/
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
canonical event is undetected, exactly one is displaced across a day boundary,
and 30 are transposed with a neighbour on the same day.

## Which backbone produced this

Stated on the first line of `curation.md`, and it changes what the artefact is
worth. **This is now `ollama` / gemma3:4b** — the canonical run, genuinely
abstractive per-event fusion following the method of the IJCNN work, with
`repeat_penalty=1.1` (llama.cpp's own semantics, not HuggingFace's — see
`stage5_generation/backbones.py`). Earlier revisions of this repository
committed `union` here instead, because the container this repository was
first run in had no access to an Ollama model; that is no longer the case.

- **`ollama` (committed)** — genuinely abstractive per-event fusion. Judge it
  on Faithful, Complete and Placement; end-to-end against the held-out
  reference it scores R-1 0.8155 / R-2 0.7363 / R-L 0.4971 / METEOR 0.4792.
  Regenerate with the command at the top of this file. 0 of 249 consolidated
  events show word-gluing corruption (`stage6_evaluation/text_quality.py`
  checks this; it is what the `repeat_penalty` fix above was for).
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
from any harmony. Kendall's τ against the held-out harmony is 0.9155 at coverage
0.8512 (143/168 clusters registered to a day).

Conflicted events are flagged. Where the sources were found to disagree under
closure, the flag means the framework detected it — the three divergences the
harmonisation literature documents (the fig tree, the Passover day, the
cockcrow) are all among the 94 reported.
