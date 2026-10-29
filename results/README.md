# Archived experiment results — post-hoc Stage 3 refinement

`outputs/` is git-ignored because every run regenerates it. This folder keeps
copies of the reports cited in the FQAS 2027 submission, so readers can check
the reported numbers without rerunning the pipeline. All files hold verse
references and metrics only; no Bible text is included.

## `posthoc_stage3/`

Produced by:

```
python scripts/posthoc_stage3_refinement.py
```

| File | Content |
|---|---|
| `refinement_report.json` | Structural metrics for the four configurations (`0_baseline_unpatched`, `1_porte4_plus_embeddings`, `2_posthoc_refinement_th033`, `3_posthoc_refinement_th034`): clusters, multi-witness purity, cross-day mixing, B-cubed over 1,221 verses, Kendall's τ, coverage, and the cluster assignments of the four impure items of the theologian sample. |
| `curation_<config>.json` | Cluster-to-verse assignment of each configuration (input to `scripts/cluster_purity.py`). |

Main figures (`3_posthoc_refinement_th034` vs. `0_baseline_unpatched`): purity
44.44% → 76.32%, cross-day mixing 9.15% → 1.75%, B-cubed precision
0.7114 → 0.8055, τ 0.9274 → 0.9363 (structural run).

The end-to-end figures of the paper (τ 0.9383 over 157/168 events, abstractive
ROUGE-L 0.6665, Stage 5 statistics) come from a full pipeline run with
`gemma4:26b` (`results.json`, `curation.json`). That run needs a local Ollama
server; its outputs will be added under `posthoc_stage3/end_to_end/`.

Regenerated on 2026-10-29 on this branch (spaCy 3.8, `en_core_web_sm`); `main`
gives identical files. SHA-256:

```
ab5830a2982e5d982856b9077736264a7cb35a247f7a453f3dca06bf26bccdb9  posthoc_stage3/refinement_report.json
956ae9b03afea21e05a1ee801105c3a4a0a885e973e6d37d2617efd197cdd250  posthoc_stage3/curation_0_baseline_unpatched.json
5d351a72ec5022fe3ff21663ae447ec1a0cba094623dd9f91923282c13f0d79d  posthoc_stage3/curation_1_porte4_plus_embeddings.json
2f6ff387a5cada956bb9c567b8ec83098676eefd27ad8f01c303fb80ed849fca  posthoc_stage3/curation_2_posthoc_refinement_th033.json
8ace1f90c07b5161279ab4e6602ca8d72c08704a7512e49e20d72d7681377580  posthoc_stage3/curation_3_posthoc_refinement_th034.json
```
