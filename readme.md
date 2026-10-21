# TAVERN

**Temporal Anchoring for Version Consolidation**

*A framework for Abstractive Narrative Consolidation from multiple overlapping accounts.*

**Narrative Consolidation (NC)** is a Natural Language Processing task formulated in
this doctoral work: given several long narrative accounts of the same events,
produce one continuous account that is chronologically correct, carries every
detail any source contributes, and states shared material once. It is not
multi-document summarization — the objective is unification rather than
compression, chronology is constitutive rather than secondary, and the output may
be longer than any source. The definition also excludes extraction: no system
restricted to verbatim spans can carry every detail *and* avoid repeating what
the accounts share, so abstraction is what remains once the objectives are stated.

**TAVERN** is the framework that performs the task end to end, in six stages:

| \# | Stage | What it contributes | 
| ----- | ----- | ----- | 
| 1 | Pre-processing | documents, verses, the pericope layer, pronoun/entity chains | 
| 2 | Temporal annotation | ISO 24617-1:2012 (ISO-TimeML) events, times, links | 
| 3 | Anchoring and alignment | the induced timeline, anchor scaffold, event clusters | 
| 4 | Relational graph | the TAEG, typed inter- and intra-document relations | 
| 5 | Guided abstractive generation | one fused version per event, in induced order | 
| 6 | Evaluation | held-out chronology, reference consolidation, metrics | 

The framework's governing design principle is the separation of **temporal
planning** from **surface realisation**: the chronology of the output is settled
symbolically, before any text is generated.

Three works in this doctorate share the task and the sources:

1. **TAEG** (*Journal of the Brazilian Computer Society*, 2026) formulates Narrative Consolidation and solves version selection extractively.

2. **NSNC** (*IJCNN*, 2026) generalises the selection to neuro-symbolic abstractive fusion. Both take an external, ready-made chronology (Aschmann's harmony of the Gospels) as an input to alignment.

3. **TAVERN** (this repository & PhD thesis) removes that dependency: the temporal backbone is **induced** from an ISO-TimeML annotation of the sources, and the external harmony is held out and read only at Stage 6 evaluation.

Case study: the four canonical Gospels over the Passion Week (1,245 verses).

> **PhD thesis** — Roger Antonio Finger, UNISINOS (Applied Computing)
>
> *TAVERN: Temporal Anchoring for Version Consolidation — A Framework for Abstractive Narrative Consolidation from Multiple Overlapping Accounts*
>
> Advisor: Prof. Dr. Gabriel de Oliveira Ramos

## Why this repository changed

Earlier revisions of TAVERN itself — not TAEG or NSNC, which never claimed
otherwise — also took the Aschmann chronology as an input to alignment. That
made Kendall's τ — the ordering metric — **1.000 by construction** in every
configuration ever reported, because the system was never asked to exercise the
competence the metric measures. The chronology is now reachable only from Stage 6, enforced in code:

```
# tavern/config.py
def assert_no_chronology_import() -> None:
    """Raises if a chronology load originates in stages 1–5."""

```

τ becomes a measurement. It is **0.9274** on the primary run. Against a null system that uses
**no annotation at all** — N1, positional interleaving of the four documents
in raw verse order — τ is 0.8140. So 0.9274 closes **61.0%** of the distance
between "no annotation" and the curated chronology (1.000). Both numbers belong together; τ alone hides that the useful range is only 0.186 wide.

This repository tracks two key states:

1. **`ancoragem` (primary frozen baseline, commit-tagged `ancoragem-20260831`, digest `c903b9fd...1a4cd`):** The exact frozen run evaluated by theological experts in blind qualitative trials.

2. **`posthoc-refinement` (thesis Section 10.9, branch `posthoc-stage3-refinement`):** The structural refinement of Stage 3 motivated by expert evaluation diagnostics (Question G) and cluster-purity audits, resolving the baseline's cross-document grouping errors.

The previous four-stage pipeline is preserved under `legacy/` rather than deleted.

## Results

Measured by `run_experiments.py --all --backbone ollama --backbone-model gemma4:26b --ollama-repeat-penalty 1.1` on the digest-pinned corpus.

| Measure | canonical (before) | **ancoragem (frozen)** | **post-hoc refined (§10.9)** | 
| ----- | ----- | ----- | ----- | 
| Kendall's τ (plan / order) | 0.9155 | **0.9274** | **0.9395** | 
| Pairwise accuracy | 0.9577 | **0.9637** | **0.9698** | 
| Curated event coverage | 85.1% (143/168) | **88.7% (149/168)** | **92.3% (155/168)** | 
| Clusters (multi-witness) | 249 (120) | **289 (153)** | **291 (119)** | 
| **Cluster Purity** | 30.8% | **44.4% (68/153)** | **73.95% (88/119)** | 
| **Day-Mixing Rate** | 14.2% | **9.15% (14/153)** | **2.52% (3/119)** | 
| B-cubed Precision / Recall / F1 | 0.584 / 0.434 / 0.498 | 0.654 / 0.431 / **0.519** | **0.798** / 0.425 / **0.555** | 
| ROUGE-L, extractive (induced order + grouping) | 0.594 | **0.662** | **0.671** | 
| ROUGE-L, abstractive end-to-end (Ollama) | 0.497 | **0.622** | **0.635** | 
| Content coverage (overall / multi-source) | 91.2% / 89.7% | **98.8% / 98.3%** | **99.1% / 98.9%** | 
| Selection accuracy vs. floor | 0.2973 < 0.3446 (74) | **0.3600 > 0.3411 (75)** | **0.3733 > 0.3411 (75)** | 

The abstractive row is what the Stage 5 fusion work moved (R-L 0.566 → 0.604 with fixes, → 0.622 on gemma4:26b; content coverage 93.3% → 98.8%). τ, pairwise, coverage, clusters, purity and B-cubed reproduce to the digit.

## Post-Hoc Grouping Refinement (Thesis §10.9)

### Diagnostic Background

In the blind evaluation conducted with theological experts, TAVERN achieved 100% factual completeness and zero hallucination. However, qualitative feedback on Question G revealed that in \~40–50% of multi-source clusters, the system merged non-parallel scenes that occurred around the same liturgical day (e.g., merging John's Farewell Discourse with Synoptic Gethsemane/Denial episodes).

Auditing Stage 3 uncovered three architectural causes in the frozen baseline:

1. **Unconstrained profile column merges (`_mergeable`):** Adjacent columns in the profile alignment were merged into canonical episodes whenever verse contiguity permitted, even when the two columns shared *no source Gospel in common*, manufacturing cross-document identity out of mere narrative adjacency.

2. **Dormant lexical/dense semantics in pair scoring:** While dense embeddings were configured in `config.py`, `pipeline.py` passed `embeddings=None` to `cluster_units`. Pair scoring relied on exact lemma matches in `PredicateIDF` (causing verbs like *SIT* in Mark 12:36 to collide with Matthew 23:22).

3. **Spurious interpolation credit in anchor scoring:** `anchor_compatibility` treated linearly interpolated day-positions identically to observed narrative `<TIMEX3>` pins, allowing long unanchored monologues (John 14–16) to claim high temporal affinity with Synoptic action scenes.

### Three Non-Destructive Refinements

The post-hoc refinement (`scripts/posthoc_stage3_refinement.py`) introduces:

* **Contiguous-span book overlap guard:** `_mergeable` requires `set(cur_books) & set(nxt_books)` before joining adjacent columns.

* **Combined semantic representations:** `predicate_similarity` wires 96-dim SVD dense unit embeddings and content-word bigram TF-IDF vectors into `PredicateIDF` (40% dense, 25% bigram TF-IDF, 35% ISO predicate vector).

* **Interpolated-anchor discounting:** discounts anchor affinity by 0.60 when either unit's scaffold position is interpolated rather than pinned by an observed temporal expression.

### Impact on Evaluated Expert Cases

This refinement increases cluster purity from **44.4% to 73.95%** (reaching 82.5% of the 89.5% theoretical ceiling) and cuts day-mixing to **2.52%**, completely resolving all 4 contaminated cases identified in the theologian sample:

* **Case A01 (Mark 12:35–36 vs. Matthew 23:22):** Correctly matches Mark 12:35–36 with Matthew 22:41–46 and Luke 20:41–44 (the question about David's Son); isolates Matthew 23:22 (the Woes to the Scribes).

* **Case A02 (John 13:29–30 vs. Luke 22:29–30):** Separates Judas's departure into the night from Jesus's promise of thrones to the Apostles.

* **Case A04 (John 13:31 vs. Peter's Denial Prediction):** Excludes John 13:31 ("Now is the Son of Man glorified") and correctly aligns John 13:36–38 with Matthew 26:33–35, Mark 14:29–31, and Luke 22:31–34.

* **Case A07 (John 16:16 vs. Peter's Courtyard Denial):** Completely decouples the Farewell Discourse from the courtyard scene; aligns Matthew 26:69–70, Mark 14:66–68, and Luke 22:55–57 into a pure denial cluster.

To run the post-hoc benchmark:

```
python scripts/posthoc_stage3_refinement.py

```

## What the measurements show, and what they cannot

Four cautions belong next to the tables above. They are not disclaimers; each is a measured result.

1. **Kendall's τ has a floor here, and it is high.** A null model that reads no annotation at all — cutting each document into positional windows and interleaving them by position (N1) — reaches τ = 0.8140 on this corpus. The interpretable band is 0.186 wide, not 2.0. A τ of 0.9274 read at face value overstates the contribution by roughly a factor of three. `scripts/null_model_n1.py` reproduces it.

2. **The architecture protects its own metric.** Progressive profile alignment is monotone by construction, so the ordering graph is acyclic, the feedback arc set never fires, and τ cannot register a whole class of failure. When evaluated under an unconstrained agglomerative clustering of the same corpus, the feedback arc set removes 4,203 arcs and τ collapses to 0.7812 (*below the null model*). Run: `python run_experiments.py --legacy-agglomerative`.

3. **Reproducibility is not significance.** A ten-seed sweep of the version selection metric gives sd = 0.0042. The true sampling distribution is over the 75 events, where null standard deviation is 0.0539. On the honest denominator, z = 0.37 and P(random selector scores as high) = 0.31: the result is reproducible and indistinguishable from chance simultaneously. `scripts/seed_sweep_selection.py`.

4. **Faithfulness was a code defect; fusion was a capacity limit.** Stage 5 prompt fixes dropped invalid fusion episodes from 104 to 11, with 0 detail loss or hallucination. Changing the backbone to `gemma4:26b` cut excess repetition to 0.067 and reduced union fallback to 4 of 155 events (97.4% abstractive).

## Installation & Setup

```
git clone https://github.com/neemias8/TAVERN.git
cd TAVERN
pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

*Notes on dependencies:*

* `rouge-score==0.1.2`: if building against recent setuptools fails, install with `pip install --no-build-isolation rouge-score==0.1.2`.

* `pylcs`: optional C extension for fast LCS calculation. If absent, a bit-parallel fallback is used automatically.

* `torch`: required only for Stage 4 GNN. Without it, Stage 4 falls back to unpropagated aggregation.

## Running TAVERN

### Full Pipeline Run

```
ollama pull gemma4:26b
python run_all.py            # On Windows: .\run-all.bat

```

### Reproducing Experiments & Tables

```
# Stages 1–5 only (reads no external chronology or reference)
python main.py --backbone ollama --backbone-model gemma4:26b

# Full evaluation suite (~12 min on 2 cores)
python run_experiments.py --all --tag main --backbone extractive

# Post-Hoc Stage 3 Grouping Refinement Benchmark
python scripts/posthoc_stage3_refinement.py

```

### Outputs

Artefacts are written to `outputs/<tag>/`:

* `annotation/<book>.tml`: conformant stand-off ISO-TimeML annotations.

* `stage3/timeline.json`: induced event order, clusters, and detected conflicts.

* `consolidated.txt`: the final unified, non-redundant abstractive narrative.

* `consolidated_with_markers.txt`: consolidated text tagged with event IDs for audit.

* `results.json`: all quantitative evaluation metrics.

## Repository Structure

```
TAVERN/
├── main.py                          # CLI entry point for Stages 1–5
├── run_experiments.py               # Stage 6 quantitative evaluation suite
├── run_all.py                       # End-to-end execution and packaging
├── scripts/
│   ├── posthoc_stage3_refinement.py # Post-hoc Stage 3 refinement benchmark (§10.9)
│   ├── null_model_n1.py             # Positional interleaving baseline (τ = 0.8140)
│   ├── cluster_purity.py            # B-cubed and purity evaluation against gold
│   ├── check_text_quality.py        # Faithfulness and repetition auditing
│   └── seed_sweep_selection.py      # Statistical significance testing
├── tavern/
│   ├── config.py                    # Global config & assert_no_chronology_import()
│   ├── pipeline.py                  # Orchestration across stages 1–5
│   ├── stage1_preprocessing/        # Segmentation, pericopes, entity chains
│   ├── stage2_temporal_annotation/  # ISO-TimeML events, timexes, links
│   ├── stage3_anchoring_alignment/  # Scaffold, profile alignment, graph induction
│   ├── stage4_gnn/                  # Relational GNN for version selection
│   ├── stage5_generation/           # Constrained abstractive fusion
│   └── stage6_evaluation/           # Ground truth chronology, metrics
├── avaliacao_humana/                # Specialist evaluation instruments & frozen booklet
└── DATA_PROVENANCE.md               # Corpus hashes, verse maps, and caveats

```

## Citation & Publications

If you use TAVERN or the Narrative Consolidation task formulation, please cite:

```
@article{finger2026narrative,
  author  = {Finger, Roger Antonio and Cortes, Eduardo Gabriel and
             Rigo, Sandro José and Ramos, Gabriel de Oliveira},
  title   = {Narrative Consolidation: Formulating a New Task for
             Unifying Multi-Perspective Accounts},
  journal = {Journal of the Brazilian Computer Society},
  volume  = {32},
  number  = {1},
  pages   = {1--24},
  year    = {2026},
  doi     = {10.5753/jbcs.2026.7717},
  note    = {TAEG: https://github.com/neemias8/TAEG}
}

@inproceedings{finger2026neurosymbolic,
  author    = {Finger, Roger Antonio and Cortes, Eduardo Gabriel and
               Rigo, Sandro Jose and De Oliveira Ramos, Gabriel},
  title     = {Neurosymbolic Narrative Consolidation: Grounding Abstractive
               MDS and LLMs with Temporal Event Graphs},
  booktitle = {2026 International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--7},
  year      = {2026},
  doi       = {10.1109/IJCNN69854.2026.11729849},
  url       = {https://doi.org/10.1109/IJCNN69854.2026.11729849},
  note      = {NSNC: https://github.com/neemias8/Neuro-Symbolic-Narrative-Consolidation}
}

```

### Related Works & Context

* Cunha, A., Sena, C. & Barbosa, J. R. (2026). *Agentic Multi-Document Summarization with Anchoring and Consolidation of Parallel Events: An Experiment with the Four Gospels.* Anais do XX Workshop de Engenharia de Sistemas Agentes, Seus Ambientes e Aplicações (WESAAC/BRACIS 2026).

* Aschmann, P. *Chronology of the Four Gospels.*

* ISO 24617-1:2012. *Language Resource Management — Semantic Annotation Framework — Part 1: Time and Events.*

## Acknowledgements

This research was conducted within the Graduate Program in Applied Computing at UNISINOS, supported by EMBRAPII, AKCIT, CNPq (313845/2023-9), and Positivo Tecnologia S/A.

Reference consolidation and golden sample: Cunha (2025); Cunha & Sena (2026).

## License

Academic research project — UNISINOS. This licence covers the code and the stand-off annotation artefacts. It does **not** cover the New International Version text (© Biblica, Inc.) or Aschmann's harmony, present in `data/` strictly for cryptographic reproducibility — see [DATA_PROVENANCE.md](https://www.google.com/search?q=DATA_PROVENANCE.md).