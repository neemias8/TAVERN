# Consolidations — system output for expert curation

The consolidated narrative TAVERN produces, laid out so that a human expert can
judge it. Regenerate with:

```bash
python main.py --tag union --backbone union
python scripts/make_curation.py outputs/union/curation.json consolidations/
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
worth.

- **`union`** — deterministic, no model. Every sentence from every account
  survives unless another already covers it. **Detail-preserving but not
  abstractive**: it generates no new wording, so the seams between accounts stay
  visible and the prose does not flow. Judge it on Faithful and Complete;
  fluency is not what it offers.
- **`ollama` / `instruct` / `bart` / `pegasus` / `primera`** — genuinely
  abstractive per-event fusion, following the method of the IJCNN work. These
  need a model, and the container this repository was last run in had no access
  to one. On a machine with Ollama:

  ```bash
  ollama pull gemma3:4b
  python main.py --tag gemma --backbone ollama --backbone-model gemma3:4b
  python scripts/make_curation.py outputs/gemma/curation.json consolidations/gemma/
  ```

The `union` output is committed as a floor, not as the intended output of the
system. TAVERN is an abstractive framework by design: the whole point of fusing
per event rather than selecting per event is that complementary details from
every version reach the consolidation in continuous prose.

## Where the ordering comes from

The order of events in these files was **induced from the text**. The Aschmann
harmony is held out and is read only by the evaluation stage; the day indices
shown come from the annotation's own temporal expressions and anchor chains, not
from any harmony. Kendall's τ against the held-out harmony is 0.9163 at coverage
0.8512.

Conflicted events are flagged. Where the sources were found to disagree under
closure, the flag means the framework detected it — the three divergences the
harmonisation literature documents (the fig tree, the Passover day, the
cockcrow) are all among the 91 reported.
