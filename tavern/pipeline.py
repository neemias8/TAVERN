"""
End-to-end pipeline driver.

Stages 1 to 5 run here; Stage 6 is invoked separately by `run_experiments.py`,
so the guard of `config.assert_no_chronology_import` continues to hold for
everything this module touches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import OUTPUT_DIR, TavernConfig, verify_corpus
from .stage1_preprocessing.coref import EntityChains, resolve_all
from .stage1_preprocessing.corpus import Corpus
from .stage1_preprocessing.pericopes import PericopeLayer, load_pericopes
from .stage1_preprocessing.segmentation import Segmenter, SegmentedDocument
from .stage2_temporal_annotation import annotate_corpus
from .stage2_temporal_annotation.model import AnnotationStructure
from .stage2_temporal_annotation.validator import ValidationReport
from .stage3_anchoring_alignment import Stage3Result
from .stage3_anchoring_alignment import run as run_stage3
from .stage4_gnn import GNNResult, aggregate_without_propagation, mean_over_seeds
from .stage5_generation import (Consolidation, SelectionStrategy,
                                build_fuser, consolidate)


@dataclass
class PipelineResult:
    cfg: TavernConfig
    corpus: Corpus
    pericopes: PericopeLayer
    segments: Dict[str, SegmentedDocument]
    chains: Dict[str, EntityChains]
    structs: Dict[str, AnnotationStructure]
    reports: Dict[str, ValidationReport]
    stage3: Stage3Result
    gnn: Optional[GNNResult] = None
    consolidation: Optional[Consolidation] = None
    backbone_note: Optional[str] = None

    @property
    def units(self):
        return self.stage3.graph.node_units

    @property
    def ordering_conflicts(self) -> List[Tuple[str, str]]:
        return self.stage3.induced.conflicted_clusters(self.stage3.clustering)

    def nrt_chain_count(self) -> int:
        return sum(len(tl.nrt_chain) for tl in self.stage3.timelines.values())


_CACHE: Dict[str, object] = {}


def _apply_granularity(cfg: TavernConfig) -> None:
    """Push the configuration's granularity knobs into the Stage 3 modules, so
    ablations do not require editing module constants."""
    from .stage3_anchoring_alignment import event_coref, local_timeline
    local_timeline.MAX_UNIT_VERSES = cfg.max_unit_verses
    local_timeline.ENTITY_TURNOVER = cfg.entity_turnover
    event_coref.MAX_EPISODE_VERSES = cfg.max_episode_verses
    event_coref.ANCHOR_BAND = cfg.anchor_band
    event_coref.GAP_COST = cfg.gap_cost
    event_coref.MATCH_THRESHOLD = cfg.match_threshold
    event_coref.GATED_SCORE = cfg.gated_score
    event_coref.NO_ANCHOR_CREDIT = cfg.no_anchor_credit


def prepare(cfg: TavernConfig, verify: bool = True):
    """Stages 1 and 2. Cached, because the ablation grid re-uses them."""
    key = f"{cfg.projection_mode}:{cfg.use_veridicality}:" \
          f"{cfg.cascade_levels}:{cfg.use_closure}"
    if key in _CACHE:
        return _CACHE[key]
    if verify:
        verify_corpus()
    corpus = Corpus()
    pericopes = load_pericopes(corpus)
    segments = Segmenter(cfg.spacy_model).segment_corpus(corpus)
    chains = resolve_all(segments)
    structs, reports = annotate_corpus(segments, pericopes, cfg, write=False)
    out = (corpus, pericopes, segments, chains, structs, reports)
    _CACHE[key] = out
    return out


def run(cfg: TavernConfig, with_gnn: bool = True, write: bool = True,
        verify: bool = True) -> PipelineResult:
    _apply_granularity(cfg)
    corpus, pericopes, segments, chains, structs, reports = prepare(cfg, verify)

    stage3 = run_stage3(structs, corpus, pericopes, chains, cfg, write=write)

    gnn = None
    scores: Dict[str, float] = {}
    if with_gnn:
        if cfg.use_graph_propagation:
            gnn = mean_over_seeds(stage3.graph, cfg)
        else:
            gnn = aggregate_without_propagation(stage3.graph)
        scores = gnn.node_scores

    kw = {}
    if cfg.backbone_model:
        kw["model_name" if cfg.backbone != "ollama" else "model"] = \
            cfg.backbone_model
    if cfg.backbone == "ollama" and "ollama_repeat_penalty" in cfg.extra:
        # A/B control for the backend-specific decoding fix; see
        # OLLAMA_REPEAT_PENALTY in stage5_generation/backbones.py. Not a
        # tuning knob for the other backbones' fixed decoding controls.
        kw["repeat_penalty"] = cfg.extra["ollama_repeat_penalty"]
    # Shared across every tag and config, not cfg.run_dir(): CachedFuser's
    # key already digests the backbone, model, repeat_penalty, the conflict
    # flag and the exact texts (backbones.py, CachedFuser._key), so a fusion
    # is only ever reused for an identical call, regardless of which run
    # produced it first. The ablation table reruns the pipeline under 5-9
    # different configs whose clusters mostly overlap the main run's (e.g.
    # "- closure" touches TLINK derivation, not clustering, so its ~249
    # clusters are close to byte-identical to "full"'s); a per-tag cache
    # path made every one of those a near-total cache miss, turning a
    # 20-90 min generation into hours. One shared file lets the second and
    # later configs in a sequence hit the cache for whatever they share with
    # the ones already run.
    fuser, note = build_fuser(cfg.backbone,
                              cache_path=OUTPUT_DIR / "fusion_cache.jsonl",
                              **kw)

    cons = consolidate(
        stage3.induced, stage3.clustering, stage3.graph.node_units,
        fuser=fuser,
        strategy=SelectionStrategy("graph_score" if scores else "longest",
                                   scores=scores),
        conflicts=stage3.induced.conflicted_clusters(stage3.clustering))

    res = PipelineResult(cfg, corpus, pericopes, segments, chains, structs,
                         reports, stage3, gnn, cons, note)
    if write:
        out = cfg.run_dir()
        cons.write(out / "consolidated.txt")
        cons.write(out / "consolidated_with_markers.txt", with_markers=True)
        import json as _json
        (out / "curation.json").write_text(
            _json.dumps({"backbone": cons.backbone,
                         "fallback_note": note,
                         "cache": {"hits": getattr(fuser, "hits", 0),
                                   "misses": getattr(fuser, "misses", 0)},
                         "events": cons.records}, indent=1),
            encoding="utf-8")
        from .stage2_temporal_annotation.serializer import (
            serialise, serialise_token_layer, write_json_projection)
        ann = out / "annotation"
        for book, s in structs.items():
            serialise(s, segments[book], ann / f"{book}.tml")
            serialise_token_layer(segments[book], ann / f"{book}.tokens.xml")
            write_json_projection(s, ann / f"{book}.json")
    return res
