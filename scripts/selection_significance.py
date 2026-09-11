#!/usr/bin/env python3
"""
Is the induced configuration's selection accuracy distinguishable from a
uniform-random selector, on the sampling distribution that actually matters?

The seed sweep (Addendum 17, `seed_sweep_selection.py`) measures dispersion
across GNN training seeds: sd = 0.0042 for the adopted configuration. That is
reproducibility, not significance -- it says repeated runs of the SAME
selector agree with each other, not that the selector beats chance. The
question of chance is about the OTHER axis: 75 independent per-event coin
flips, each with its own success probability 1/n_versions(event). This script
computes that null distribution exactly (a Poisson-binomial: independent but
non-identical Bernoulli trials, via convolution -- not a normal
approximation, which understates the tail here) and reports where the
observed mean sits in it.

    python scripts/selection_significance.py
"""
from __future__ import annotations

import statistics
import sys
from math import erf, sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern import pipeline
from tavern.config import TavernConfig
from tavern.stage3_anchoring_alignment import run as run_stage3
from tavern.stage6_evaluation import chronology as chrono_mod, selection_eval, timeline_eval

OBSERVED_MEAN = 0.3613  # mean selection accuracy, 'full' config, N=10 seeds


def poisson_binomial_pmf(ps):
    dp = [1.0]
    for p in ps:
        ndp = [0.0] * (len(dp) + 1)
        for k, mass in enumerate(dp):
            if mass:
                ndp[k] += mass * (1 - p)
                ndp[k + 1] += mass * p
        dp = ndp
    return dp


def main() -> int:
    cfg = TavernConfig(tag="selection_significance", backbone="extractive")
    pipeline._apply_granularity(cfg)
    corpus, pericopes, segments, chains, structs, reports = pipeline.prepare(
        cfg, verify=True)
    stage3 = run_stage3(structs, corpus, pericopes, chains, cfg, write=False)
    units = stage3.graph.node_units
    ch = chrono_mod.load(corpus)
    matching, _q = timeline_eval.match_clusters_to_events(ch, stage3.clustering, units)
    subset = selection_eval.contested_intersection(ch, stage3.clustering, matching)
    n_versions = {e.event_id: e.n_versions for e in ch.events}

    n = len(subset)
    ps = [1.0 / n_versions[eid] for eid in subset]
    floor = statistics.fmean(ps)

    dp = poisson_binomial_pmf(ps)
    mean_count = sum(k * m for k, m in enumerate(dp))
    var_count = sum((k - mean_count) ** 2 * m for k, m in enumerate(dp))
    sd_acc = (var_count ** 0.5) / n

    z = (OBSERVED_MEAN - floor) / sd_acc
    threshold = OBSERVED_MEAN * n
    p_exact = sum(m for k, m in enumerate(dp) if k >= threshold - 1e-9)
    p_normal = 0.5 * (1 - erf(z / sqrt(2)))

    print(f"subset n={n}")
    print(f"null mean (floor)      = {floor:.4f}")
    print(f"null sd (exact)        = {sd_acc:.4f}")
    print(f"observed mean (N=10 seeds, 'full' config) = {OBSERVED_MEAN:.4f}")
    print(f"z                      = {z:.4f}")
    print(f"P(random >= observed), exact Poisson-binomial  = {p_exact:.4f}")
    print(f"P(random >= observed), normal approximation    = {p_normal:.4f}")
    print("(the normal approximation understates the tail here -- report the exact figure)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
