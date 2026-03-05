# TAVERN
**Temporal Analysis of Verse-Event Relations in Narratives**

A five-stage NLP pipeline for extracting, aligning, encoding, and narrating
temporal events from the four Gospel texts.

---

## Architecture

```
Gospel XML files
      │
      ▼
┌─────────────────────────────────────────┐
│  Stage 1 – ISO-TimeML Annotation        │  annotator.py
│  EVENT · TIMEX3 · SIGNAL · TLINK · SLINK│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 2 – Cross-Document Alignment     │  aligner.py
│  SBERT 45% · Entity 30% · Proximity 25% │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 3 – Relational GAT               │  gnn_model.py
│  Per-relation transforms · Multi-head   │
│  attention · Residual + LayerNorm       │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 4 – Narrative Generation         │  generator.py
│  Day segmentation · Topo-sort · LLM/    │
│  template fallback                      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Stage 5 – Evaluation                   │  evaluator.py
└─────────────────────────────────────────┘
```

---

## Stages

### Stage 1 — ISO-TimeML Temporal Annotation
Annotates each Gospel verse according to **ISO 24617-1:2012**:

| Element  | Attributes |
|----------|------------|
| `EVENT`  | `class` (7 types), `tense`, `aspect`, `polarity` |
| `TIMEX3` | `type` (DATE/TIME/DURATION/SET), ISO 8601 `value` |
| `SIGNAL` | temporal connective text |
| `TLINK`  | Allen's 16 interval relations |
| `SLINK`  | EVIDENTIAL / MODAL / COUNTER_FACTIVE |

### Stage 2 — Cross-Document Event Alignment
Multi-factor alignment score:

```
score = 0.45 × semantic_similarity (SBERT)
      + 0.30 × entity_overlap
      + 0.25 × temporal_proximity
```

Includes conflict detection for incompatible temporal relations
(e.g. one gospel says A BEFORE B, another says A AFTER B).

### Stage 3 — Relational GAT
A two-layer **Relational Graph Attention Network** over the temporal
knowledge graph:
- 20 relation types (16 TLINKs + SLINK types + ALIGNED + SEQUENTIAL)
- Per-relation linear transform → multi-head attention → residual + LayerNorm
- PyTorch backend when available, pure-NumPy fallback otherwise

### Stage 4 — Narrative Generation
1. Temporal segmentation by day (TIMEX3 anchors + day-boundary markers)
2. Graph-guided event ordering (topological sort on TLINKs)
3. LLM-based paragraph generation (GPT-4o-mini) with template fallback

### Stage 5 — Evaluation

| Metric | Description |
|--------|-------------|
| Temporal Ordering Accuracy | % of TLINKs matching gold |
| Temporal Precision/Recall/F1 | Set-based TLINK evaluation |
| Adjacent Pair Accuracy | Accuracy on consecutive event pairs |
| Entity Coverage | % of gold entities in narrative |
| Narrative Coherence Score | Avg. consecutive-sentence cosine similarity |
| Redundancy Ratio | Fraction of near-duplicate sentences |

---

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

```bash
python main.py --docs data/ --chronology data/chronology.csv --output output/
```

## Data Format

Gospel XML files should follow this structure:

```xml
<book>
  <chapter number="1">
    <verse number="1">In the beginning was the Word...</verse>
    ...
  </chapter>
</book>
```

## Project Structure

```
TAVERN/
├── main.py
├── requirements.txt
├── README.md
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── stage1_temporal_annotation/
│   ├── __init__.py
│   └── annotator.py
├── stage2_event_alignment/
│   ├── __init__.py
│   └── aligner.py
├── stage3_gnn_modeling/
│   ├── __init__.py
│   └── gnn_model.py
├── stage4_abstractive_generation/
│   ├── __init__.py
│   └── generator.py
└── stage5_evaluation/
    ├── __init__.py
    └── evaluator.py
```
