# Archived experiment results — post-hoc Stage 3 refinement

`outputs/` is git-ignored because every run regenerates it. This folder keeps
copies of the reports cited in the FQAS 2027 submission, so readers can check
the reported numbers without rerunning the pipeline.

## Two checkouts, and why that matters

The headline comparison has its two ends in **different checkouts**, and the
files here say which:

* the **baseline** is the package the thesis was run with — `main`, commit
  `c4465bc3`;
* the **refinement** is the package on this branch, which is what produced
  the consolidated narrative.

This is not a bookkeeping detail. The refinement touches three modules, and
only one of them is behind a flag:

| module | change | gated? |
|---|---|---|
| `event_coref.py` | `REFINE_GROUPING`, `INTERPOLATED_ANCHOR_DISCOUNT` | yes |
| `scaffold.py` | `in_narration` filter on both projection passes, tolerance on pins, clear of projected days/parts | **no** |
| `biblical_calendar.py` | `half` 0 → 0.5, `several` 0 → `None`, month code `M` → `MO` | **no** |

Because two of the three apply unconditionally, *every* row of the structural
table moves with the checkout — the baseline included. Measured:
`0_baseline_unpatched` is 289 clusters / 44.44% purity against the thesis
package and 317 / 55.88% against this branch's. A table generated from one
checkout and a table generated from the other are not comparable row by row.

Every row of `refinement_report.json` therefore carries a `package_state`
field (git HEAD plus the three discriminating points above). The three
booleans identify the package; `git_head` identifies the repository revision
and does not by itself tell you which `tavern/` was in the tree.

## Files

All files are flat in this folder.

| File | Package | Content |
|---|---|---|
| `refinement_report.json` | thesis (`main`) | Structural metrics for the four configurations: clusters, multi-witness purity, cross-day mixing, B-cubed over 1,221 verses, Kendall's τ, coverage, and the cluster assignments of the four impure items of the theologian sample. |
| `curation_<config>.json` | thesis (`main`) | Cluster-to-verse assignment of each configuration (input to `scripts/cluster_purity.py`). |
| `refinement_report_branch_package.json` | this branch | The same four configurations measured against the deployed package, for the comparison in the next section. |
| `results.json` | this branch | Full-pipeline report: timeline, consistency checks, conflicts, selection, error taxonomy, end-to-end content metrics. |
| `curation.json` | this branch | Per-event record of the consolidation: sources, fusion path, fidelity verdict, scaffold day, and the consolidated paragraph. |

**Content note.** The structural files (`refinement_report*.json`,
`curation_<config>.json`) hold verse references and metrics only. `curation.json`
and `results.json` are different: `curation.json` carries the **NIV text** of
every source account and every consolidated paragraph, for all 347 events. That
is the same material already published under `consolidations/` on `main`; it is
noted here so nobody cites this folder as text-free.

## Main figures

Baseline from the thesis package, refinement from this branch:

| | baseline (`main`) | refinement (this branch) |
|---|---|---|
| clusters | 289 | 347 |
| multi-witness purity | 44.44% | **74.14%** |
| cross-day mixing | 9.15% | 2.6% |
| B-cubed P / R / F1 | 0.7114 / 0.4090 / 0.5194 | 0.8034 / 0.4266 / 0.5572 |
| Kendall's τ | 0.9274 | **0.9383** |
| pairwise accuracy | 0.9637 | 0.9691 |
| coverage | 0.8869 (149/168) | **0.9345 (157/168)** |

End to end, abstractive fusion with `gemma4:26b`: R-1 0.8310, R-2 0.8082,
**R-L 0.6665**, METEOR 0.5412.

The structural progression, all four rows against the thesis package
(`refinement_report.json`):

| config | clusters | purity | day mixing | τ | matched |
|---|---|---|---|---|---|
| `0_baseline_unpatched` | 289 | 44.44% | 9.15% | 0.9274 | 149/168 |
| `1_porte4_plus_embeddings` | 309 | 60.28% | 4.26% | 0.9410 | 151/168 |
| `2_posthoc_refinement_th033` | 348 | 77.19% | 0.88% | 0.9360 | 156/168 |
| `3_posthoc_refinement_th034` | 347 | 76.32% | 1.75% | 0.9363 | 157/168 |

## Why configuration 3 reads 76.32% here and 74.14% end to end

These are **one algorithm, not two**. `scripts/posthoc_stage3_refinement.py`
reimplements the refinement by monkeypatching `event_coref`, and the package
implements it behind `REFINE_GROUPING`; the two agree exactly. Run against the
same package, configuration 3 reproduces the full pipeline to the digit:

```
347 of 347 clusters identical, purity 0.7414, tau 0.9383,
coverage 0.9345, B-cubed F1 0.5572 — every field
```

The gap between 76.32% and 74.14% is `scaffold.py` and `biblical_calendar.py`,
the two unconditional modules above, and nothing else. Compare
`curation_3_posthoc_refinement_th034.json` (thesis package) with
`curation.json` (this branch) and 16 of the 347 clusters differ; regenerate the
former against this branch's package and the difference is zero.

## Reproducing

The structural table, from this branch, with the thesis package swapped in:

```bash
F="tavern/stage3_anchoring_alignment/event_coref.py \
   tavern/stage3_anchoring_alignment/scaffold.py \
   tavern/stage2_temporal_annotation/biblical_calendar.py"
git checkout origin/main -- $F
python scripts/posthoc_stage3_refinement.py --json results/refinement_report.json
git checkout HEAD -- $F          # restore the branch package
```

The same table against the deployed package — just omit the two `git checkout`
lines and write to `refinement_report_branch_package.json`.

The end-to-end run (needs a local Ollama server with `gemma4:26b`):

```bash
python run_experiments.py --corpus --annotation --consistency --timeline \
    --downstream --errors --tag posthoc_refined \
    --backbone ollama --backbone-model gemma4:26b --ollama-repeat-penalty 1.1
```

Stage 4 is not bit-reproducible across runs (thesis §9.6), so ROUGE-family
figures move by up to ~0.006 between runs of the identical config. τ, coverage
and the cluster count are exactly reproducible; the structural table above does
not involve Stage 4 at all.

## SHA-256

These are hashes of the **git blobs** (LF line endings), not of the files as
checked out — on Windows the working tree is CRLF and `sha256sum` on the file
gives a different value. Verify with:

```bash
git show HEAD:results/<file> | sha256sum
```

```
d716b4cf18e3a0182b50e3b8cd62c047ab8a2098fc0fbad6b9d01f50588640b7  refinement_report.json
b0d70a594ee09dd5ffc45b70fba585a62f25cfa1a4470f7c9e1c556de6925bf4  refinement_report_branch_package.json
956ae9b03afea21e05a1ee801105c3a4a0a885e973e6d37d2617efd197cdd250  curation_0_baseline_unpatched.json
5d351a72ec5022fe3ff21663ae447ec1a0cba094623dd9f91923282c13f0d79d  curation_1_porte4_plus_embeddings.json
2f6ff387a5cada956bb9c567b8ec83098676eefd27ad8f01c303fb80ed849fca  curation_2_posthoc_refinement_th033.json
8ace1f90c07b5161279ab4e6602ca8d72c08704a7512e49e20d72d7681377580  curation_3_posthoc_refinement_th034.json
b8df2e882585a875ce12afc1a944479daf9d20f98ce129c5888548614086fce3  results.json
b6c7cad19ac63ebd645191579041cf44d379d7aca01d8ebab848e33eea98aff9  curation.json
```
