# TAVERN

**Temporal Anchoring for Version Consolidation in Abstractive Multi-Document Summarization**

A framework for generating coherent, chronologically sound, and consolidated abstractive summaries from multiple long, overlapping narrative documents. Primary case study: the four canonical Gospels.

> PhD thesis proposal — Roger Antonio Finger, UNISINOS (2025)
> Advisor: Prof. Dr. Gabriel de Oliveira Ramos

---

## Architecture

TAVERN implements a 6-stage pipeline:

```
Gospel 1 ─┬
Gospel 2 ──┤    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐
Gospel 3 ──┤──> │ 1. Temporal   │──> │ 2. Cross-Doc    │──> │ 3. Graph     │
Gospel 4 ─┘    │    Annotation │    │    Alignment     │    │    + R-GAT   │
               │  (ISO-TimeML) │    │  (SBERT+Entity)  │    │  Processing  │
               └──────────────┘    └─────────────────┘    └──────┬───────┘
                                                                  │
               ┌──────────────┐    ┌─────────────────┐           │
               │ 5. Evaluation│<── │ 4. Temporally-  │<──────────┘
               │   (6 metrics)│    │    Guided Abs.   │
               └──────────────┘    │    Generation    │
                                   └─────────────────┘
```

### Stage 1: Temporal Annotation (ISO 24617-1 / ISO-TimeML)

Annotates each Gospel document with full ISO-TimeML compliance:
- **`<EVENT>`** — with `class` (OCCURRENCE, REPORTING, STATE, PERCEPTION, ASPECTUAL, I_ACTION, I_STATE), `tense`, `aspect`, `polarity`
- **`<TIMEX3>`** — with `type` (DATE, TIME, DURATION, SET) and normalised `value` (ISO 8601), including biblical temporal expressions
- **`<SIGNAL>`** — temporal connectives (before, after, during, while, ...)
- **`<TLINK>`** — intra-document temporal ordering (BEFORE, AFTER, IS_INCLUDED, SIMULTANEOUS, ...)
- **`<SLINK>`** — subordination links (EVIDENTIAL, MODAL, COUNTER_FACTIVE)

### Stage 2: Cross-Document Event Alignment

Multi-factor alignment scoring:
- **Semantic similarity** — Sentence-BERT embeddings (`all-MiniLM-L6-v2`) with hash-embedding fallback
- **Entity overlap** — Biblical person/location matching via spaCy NER + dictionary
- **Temporal proximity** — Chronology-guided (Aschmann table) + verse-distance heuristic
- **Conflict detection** — Identifies contradictory details (numbers, times, locations) between parallel accounts

### Stage 3: Graph Construction & GNN Processing

Builds a unified cross-document event graph `G = (V, E)` with heterogeneous edge types:
- `INTRA_BEFORE/AFTER` — intra-document temporal order
- `SAME_EVENT` — cross-document alignment
- `INTER_BEFORE/AFTER` — inter-document temporal adjacency
- `CONFLICT_DETAIL` — contradictory details

Processes the graph with a **Relational Graph Attention Network (R-GAT)**:
- Per-relation-type linear transformations
- Multi-head attention for neighbour aggregation
- Residual connections + LayerNorm
- Self-supervised training (reconstruction + neighbour consistency)

### Stage 4: Temporally-Guided Abstractive Generation

- **Temporal segmentation** — partitions the timeline into coherent narrative segments (Saturday, Palm Sunday, Monday, ..., Resurrection Sunday)
- **Graph-guided generation** — GNN-enriched embeddings inform the summarisation model
- **PRIMERA/LED** for multi-document consolidation when multiple accounts need merging
- **Fallback chain** — BART → extractive (MMR) → longest account

### Stage 5: Evaluation

Six evaluation metrics:
| Metric | Aspect | Reference |
|--------|--------|-----------|
| ROUGE-1/2/L | Content overlap | Lin (2004) |
| BERTScore | Semantic similarity | Zhang et al. (2020) |
| METEOR | Fluency | Banerjee & Lavie (2005) |
| Kendall's Tau | Temporal ordering | Thesis §4.6 |
| Adjacent Pair Accuracy | Temporal coherence | Thesis §4.6(b) |
| Redundancy Ratio | N-gram + sentence repetition | Thesis §4.6 |

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
# For better quality: python -m spacy download en_core_web_lg
```

### Intel GPU (Arc / Xe / Data Center) — Optional

If you have an Intel GPU and want to accelerate PRIMERA inference and GNN training:

```bash
# Install Intel Extension for PyTorch (IPEX)
pip install intel-extension-for-pytorch
# Or follow: https://intel.github.io/intel-extension-for-pytorch/
```

The pipeline automatically detects Intel XPU, NVIDIA CUDA, Apple MPS, or falls back to CPU.
No code changes needed — device selection is fully automatic.

## Usage

```bash
python main.py
```

Outputs are saved to `./outputs/`:

| File | Description |
|------|-------------|
| `summary.txt` | Final consolidated narrative |
| `consolidation_details.json` | Per-event consolidation log (input text, output text, method, sources) |
| `consolidation_details.csv` | Same data in CSV format |
| `metrics_summary.csv` | All evaluation metrics in one table |
| `metrics_summary.json` | All metrics as structured JSON |
| `rouge.json` | ROUGE-1/2/L scores |
| `bertscore.json` | BERTScore (P/R/F1) |
| `meteor.json` | METEOR score |
| `kendall_tau.json` | Kendall's Tau (temporal ordering) |
| `adjacent_pair_accuracy.json` | Adjacent pair accuracy |
| `redundancy.json` | N-gram + sentence redundancy |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TAVERN_USE_PRIMERA` | Enable/disable PRIMERA | `true` |
| `TAVERN_PRIMERA_MODEL` | HF model name | `allenai/PRIMERA` |
| `TAVERN_PRIMERA_LOCAL_DIR` | Local model path | — |
| `TAVERN_USE_BART` | Enable/disable BART fallback | `true` |
| `TAVERN_BART_MODEL` | BART model name | `sshleifer/distilbart-cnn-12-6` |

## Data

- `data/EnglishNIV*.xml` — Four canonical Gospels (NIV translation)
- `data/ChronologyOfTheFourGospels_PW.xml` — Aschmann chronology table
- `data/Golden_Sample.txt` — Reference summary for evaluation

## Project Structure

```
TAVERN/
├── main.py                              # Pipeline orchestrator
├── stage1_temporal_annotation/
│   └── annotator.py                     # ISO-TimeML annotation
├── stage2_event_alignment/
│   └── aligner.py                       # Cross-document alignment
├── stage3_gnn_modeling/
│   └── gnn_model.py                     # Graph construction + R-GAT
├── stage4_abstractive_generation/
│   └── generator.py                     # Temporal segmentation + generation
├── stage5_evaluation/
│   └── evaluator.py                     # 6-metric evaluation suite
├── utils/
│   └── helpers.py                       # Shared utilities
├── data/                                # Input corpus + chronology
├── outputs/                             # Generated summaries + metrics
└── requirements.txt
```

## References

- Pustejovsky, J. et al. (2010). ISO-TimeML: An International Standard for Temporal Annotation.
- ISO 24617-1:2012. Language resource management — Semantic annotation framework — Part 1: Time and events.
- Kipf, T. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks.
- Veličković, P. et al. (2018). Graph Attention Networks.
- Xiao, W. et al. (2022). PRIMERA: Pyramid-based Masked Sentence Pre-training for Multi-document Summarization.
- Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training.
- Aschmann, P. (2022). Chronology of the Four Gospels.

## License

Academic research project — UNISINOS.
