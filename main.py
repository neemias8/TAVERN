"""
TAVERN — Temporal Analysis of Verse-Event Relations in Narratives
=================================================================
End-to-end pipeline runner.

Usage
-----
    python main.py [--docs <path>] [--chronology <csv>] [--output <dir>]

Arguments
---------
--docs          Directory containing Gospel XML files (default: data/)
--chronology    Path to the Aschmann harmony CSV (default: data/chronology.csv)
--output        Output directory for results (default: output/)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Stage imports
# ---------------------------------------------------------------------------
from stage1_temporal_annotation.annotator import run_annotation
from stage2_event_alignment.aligner import align_events
from stage3_gnn_modeling.gnn_model import build_temporal_graph, encode_graph
from stage4_abstractive_generation.generator import generate_narrative
from stage5_evaluation.evaluator import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chronology(csv_path: str):
    """Load the Aschmann harmony CSV as a list of dicts."""
    try:
        import csv
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"  Warning: could not load chronology ({exc})")
        return []


def _discover_docs(docs_dir: str) -> dict:
    """Return {doc_id: file_path} for all *.xml files under docs_dir."""
    p = Path(docs_dir)
    docs = {}
    if p.is_dir():
        for xml_file in sorted(p.glob('*.xml')):
            doc_id = xml_file.stem.lower()
            docs[doc_id] = str(xml_file)
    return docs


def _print_section(title: str, width: int = 60) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print('=' * width)


def _print_metrics(metrics: dict) -> None:
    """Pretty-print the evaluation metrics dict."""
    col_w = 38
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<{col_w}} {val:.4f}")
        else:
            print(f"  {key:<{col_w}} {val}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
        docs_dir: str = 'data',
        chronology_path: str = 'data/chronology.csv',
        output_dir: str = 'output',
        save_outputs: bool = True,
) -> dict:
    """Execute the full TAVERN pipeline and return the evaluation metrics."""

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1 — Temporal Annotation
    # ------------------------------------------------------------------
    _print_section('Stage 1 — ISO-TimeML Temporal Annotation')
    docs = _discover_docs(docs_dir)
    if not docs:
        print(f'  No XML files found in "{docs_dir}". '
              'Please add Gospel XML files and re-run.')
        return {}

    print(f'  Documents: {", ".join(docs.keys())}')
    annotated_docs = run_annotation(docs)

    if save_outputs:
        _save_json(annotated_docs, output_dir, 'stage1_annotations.json')

    total_events = sum(
        sum(1 for a in anns if a['type'] == 'EVENT')
        for anns in annotated_docs.values()
    )
    total_tlinks = sum(
        sum(1 for a in anns if a['type'] == 'TLINK')
        for anns in annotated_docs.values()
    )
    print(f'  Total: {total_events} EVENTs, {total_tlinks} TLINKs extracted')

    # ------------------------------------------------------------------
    # Stage 2 — Cross-Document Alignment
    # ------------------------------------------------------------------
    _print_section('Stage 2 — Cross-Document Event Alignment')
    chronology = _load_chronology(chronology_path)
    if chronology:
        print(f'  Loaded chronology with {len(chronology)} entries')
    else:
        print('  No chronology loaded — using semantic alignment only')

    alignments = align_events(annotated_docs, chronology=chronology)
    if save_outputs:
        _save_json(alignments, output_dir, 'stage2_alignments.json')

    print(f'  Alignment groups: {len(alignments)}')

    # ------------------------------------------------------------------
    # Stage 3 — GNN Encoding
    # ------------------------------------------------------------------
    _print_section('Stage 3 — Relational GAT Graph Encoding')
    nodes, edges = build_temporal_graph(annotated_docs, alignments)
    print(f'  Graph: {len(nodes)} nodes, {len(edges)} edges')

    embeddings = encode_graph(nodes, edges)
    print(f'  Encoded {len(embeddings)} node embeddings')

    if save_outputs:
        _save_json(
            {k: v.tolist() for k, v in embeddings.items()},
            output_dir, 'stage3_embeddings.json'
        )

    # ------------------------------------------------------------------
    # Stage 4 — Narrative Generation
    # ------------------------------------------------------------------
    _print_section('Stage 4 — Temporal Narrative Generation')
    narrative = generate_narrative(annotated_docs, alignments, embeddings)
    print(f'  Narrative: {len(narrative.split())} words, '
          f'{narrative.count(chr(10) + chr(10)) + 1} paragraph(s)')
    print()
    # Print a short preview
    preview = narrative[:500] + ('...' if len(narrative) > 500 else '')
    for line in preview.splitlines():
        print(f'  {line}')

    if save_outputs:
        narrative_path = os.path.join(output_dir, 'stage4_narrative.txt')
        with open(narrative_path, 'w', encoding='utf-8') as f:
            f.write(narrative)
        print(f'  Saved narrative to {narrative_path}')

    # ------------------------------------------------------------------
    # Stage 5 — Evaluation
    # ------------------------------------------------------------------
    _print_section('Stage 5 — Evaluation')
    metrics = evaluate(
        annotated_docs=annotated_docs,
        alignments=alignments,
        generated_narrative=narrative,
    )
    _print_metrics(metrics)

    if save_outputs:
        _save_json(metrics, output_dir, 'stage5_metrics.json')

    elapsed = time.time() - t_start
    print(f'\n  Pipeline completed in {elapsed:.1f}s')

    return metrics


def _save_json(data, output_dir: str, filename: str) -> None:
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    print(f'  Saved {filename}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='TAVERN — Temporal Analysis of Verse-Event Relations in Narratives'
    )
    parser.add_argument('--docs', default='data',
                        help='Directory with Gospel XML files')
    parser.add_argument('--chronology', default='data/chronology.csv',
                        help='Path to Aschmann harmony CSV')
    parser.add_argument('--output', default='output',
                        help='Output directory')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save intermediate outputs')
    args = parser.parse_args()

    run_pipeline(
        docs_dir=args.docs,
        chronology_path=args.chronology,
        output_dir=args.output,
        save_outputs=not args.no_save,
    )


if __name__ == '__main__':
    main()
