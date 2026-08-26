#!/usr/bin/env python3
"""
Build the expert-evaluation booklet, and score it when the ratings come back.

    # build (after run_all.py has produced both configurations)
    python scripts/make_human_eval.py build \
        --abstractive outputs/ollama/curation.json \
        --extractive  outputs/extractive/curation.json \
        --out human_eval

    # score (after the three raters return their sheets)
    python scripts/make_human_eval.py score --dir human_eval

This implements the protocol pre-registered in Section 9.3.4 of the thesis. Two
properties of that protocol are load-bearing and are enforced here rather than
left to the operator:

  * The three conditions are presented unlabelled, in an order permuted per item
    and per rater. The unblinding key is written to a separate file that the
    raters never receive.
  * The 30 adjacent pairs deliberately over-sample suspected errors, so the raw
    proportion correct is a pessimistic estimate. `score` reports the raw figure
    AND the stratum-weighted estimate, and only the latter is comparable with the
    pairwise figure against the chronology.

This script reads the chronology, so it belongs to Stage 6 and must never be
imported from stages 1-5. It is a standalone script for exactly that reason.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SEED = 20260826          # fixed so the sample is reproducible
N_EVENTS = 32
N_CONTROLS = 4        # single-account events: all conditions identical, by construction
N_PAIRS = 30
N_RATERS = 3
CONDITIONS = ("abstractive", "extractive", "longest")


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------
def _load(path: Path) -> List[dict]:
    """Read a curation.json. Accepts the pipeline's own shape and a bare list."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d
    for field in ("events", "records"):
        if isinstance(d.get(field), list):
            return d[field]
    if all(isinstance(v, dict) for v in d.values()):
        return list(d.values())
    raise ValueError(f"{path}: cannot find the event list")


def _accounts(rec: dict) -> List[Tuple[str, str, str, List[str]]]:
    """The contributing accounts as (gospel, reference, text, verse keys).

    Verse keys are carried through so the published ratings can be keyed on
    book:chapter:verse without reproducing the copyrighted text.
    """
    out: List[Tuple[str, str, str, List[str]]] = []
    for s in rec.get("sources") or []:
        if isinstance(s, dict):
            text = (s.get("text") or "").strip()
            if text:
                out.append((s.get("gospel", "?"), s.get("ref", ""), text,
                            list(s.get("verses") or [])))
    if out:
        return out
    # fall back to the flattened CSV form: "[Matthew] ... || [Mark] ..."
    for chunk in (rec.get("source_text") or "").split("||"):
        chunk = chunk.strip()
        if chunk.startswith("["):
            book, _, text = chunk[1:].partition("]")
            if text.strip():
                out.append((book.strip(), "", text.strip(), []))
    return out


def _longest(rec: dict) -> str:
    """Timeline+Longest's selection policy: the longest available account."""
    acc = _accounts(rec)
    return max((a[2] for a in acc), key=len) if acc else ""


def _n_accounts(rec: dict) -> int:
    return max(1, len(_accounts(rec)))


def _is_conflict(rec: dict) -> bool:
    v = rec.get("conflicted", rec.get("conflict", False))
    return v is True or str(v).strip().lower() in ("yes", "true", "1")


def _marker(rec: dict) -> str:
    return str(rec.get("marker") or rec.get("cluster") or rec.get("position"))


def _selected(rec: dict) -> str:
    """The account the extractive configuration would emit for this event."""
    want = (rec.get("selected_source") or "").lower()
    acc = _accounts(rec)
    for gospel, _ref, text, _vv in acc:
        if gospel.lower() == want:
            return text
    return acc[0][2] if acc else ""


def _distinct(variants: Dict[str, str]) -> bool:
    """True when the conditions produce pairwise different text."""
    vals = [v.strip() for v in variants.values()]
    return len(set(vals)) == len(vals) and all(vals)


def stratify_events(recs: List[dict], variants_of, rng: random.Random
                    ) -> Tuple[List[dict], List[dict], dict]:
    """Select the Part A sample.

    Returns (comparison items, noise-floor controls, diagnostics).

    Only events on which the conditions actually differ can inform a comparison
    between them. TAVERN's default selection is length-based, so the extractive
    condition frequently coincides with Timeline+Longest; asking a rater to
    prefer one of two identical passages is asking them to guess, and it
    contaminates both the preference and the agreement figures. The comparison
    sample is therefore drawn from the events where the three conditions are
    pairwise distinct --- the discriminative subset, on the same principle by
    which selection accuracy is reported over contested events rather than over
    all of them. A handful of single-account events, where every condition is
    necessarily identical, is retained deliberately as a noise floor: any
    difference a rater records there is measurement noise.
    """
    eligible, identical, single = [], [], []
    for r in recs:
        if _n_accounts(r) == 1:
            single.append(r)
        elif _distinct(variants_of(r)):
            eligible.append(r)
        else:
            identical.append(r)

    by_n: Dict[int, List[dict]] = defaultdict(list)
    for r in eligible:
        by_n[min(4, _n_accounts(r))].append(r)

    forced = [r for r in eligible if _is_conflict(r)]
    rng.shuffle(forced)
    chosen = forced[:8]
    seen = {id(r) for r in chosen}

    if by_n:
        per = max(1, (N_EVENTS - len(chosen)) // len(by_n))
        for n in sorted(by_n):
            pool = [r for r in by_n[n] if id(r) not in seen]
            rng.shuffle(pool)
            for r in pool[:per]:
                chosen.append(r)
                seen.add(id(r))

    pool = [r for r in eligible if id(r) not in seen]
    rng.shuffle(pool)
    while len(chosen) < N_EVENTS and pool:
        chosen.append(pool.pop())

    chosen = chosen[:N_EVENTS]
    chosen.sort(key=lambda r: r.get("position", 0))

    rng.shuffle(single)
    controls = sorted(single[:N_CONTROLS], key=lambda r: r.get("position", 0))

    diag = {
        "events_total": len(recs),
        "single_account": len(single),
        "multi_account_conditions_identical": len(identical),
        "multi_account_conditions_distinct": len(eligible),
        "comparison_items": len(chosen),
        "noise_floor_controls": len(controls),
    }
    return chosen, controls, diag


def stratify_pairs(recs: List[dict], suspect: Sequence[int],
                   rng: random.Random) -> List[Tuple[dict, dict, str]]:
    """15 pairs inside the monotone subsequence, 15 suspected transpositions."""
    ordered = sorted(recs, key=lambda r: r.get("position", 0))
    suspect_set = set(suspect)

    # index over adjacent pairs, so membership never needs a hashable record
    sus_i, clean_i = [], []
    for i in range(len(ordered) - 1):
        a, b = ordered[i], ordered[i + 1]
        if a.get("position") in suspect_set or b.get("position") in suspect_set:
            sus_i.append(i)
        else:
            clean_i.append(i)
    rng.shuffle(sus_i)
    rng.shuffle(clean_i)

    half = N_PAIRS // 2
    picked = [(i, "suspect") for i in sus_i[:half]]
    picked += [(i, "monotone") for i in clean_i[:N_PAIRS - len(picked)]]
    # if one stratum is short, top up from the other so the booklet is full
    if len(picked) < N_PAIRS:
        used = {i for i, _ in picked}
        for i in sus_i + clean_i:
            if i not in used:
                picked.append((i, "suspect" if i in set(sus_i) else "monotone"))
                if len(picked) == N_PAIRS:
                    break

    out = [(ordered[i], ordered[i + 1], st) for i, st in picked]
    rng.shuffle(out)
    return out


def build(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    abst = _load(Path(args.abstractive))
    extr = {_marker(r): r for r in _load(Path(args.extractive))} \
        if args.extractive else {}

    suspect: List[int] = []
    if args.suspect and Path(args.suspect).exists():
        suspect = json.loads(Path(args.suspect).read_text())

    def variants_of(rec: dict) -> Dict[str, str]:
        return {
            "abstractive": (rec.get("consolidated") or "").strip(),
            "extractive": ((extr.get(_marker(rec), {}) or {}).get("consolidated")
                           or _selected(rec)).strip(),
            "longest": _longest(rec).strip(),
        }

    events, controls, diag = stratify_events(abst, variants_of, rng)
    pairs = stratify_pairs(abst, suspect, rng)
    if diag["multi_account_conditions_distinct"] < N_EVENTS:
        print("WARNING: only %d events have three distinct conditions; the "
              "comparison sample is smaller than planned."
              % diag["multi_account_conditions_distinct"], file=sys.stderr)
    items = [(r, "comparison") for r in events] + [(r, "control") for r in controls]

    key: Dict[str, dict] = {}
    for rater in range(1, N_RATERS + 1):
        lines = [f"# Expert Evaluation --- Rater {rater}", "",
                 "Read the briefing before starting. The three versions of each",
                 "episode are in a different order in every item and for every",
                 "rater; there is no pattern to find.", "",
                 "## Part A --- Episodes", ""]
        for i, (rec, kind) in enumerate(items, 1):
            variants = variants_of(rec)
            order = list(CONDITIONS)
            rng.shuffle(order)
            key[f"R{rater}:A{i:02d}"] = {
                "marker": _marker(rec), "position": rec.get("position"),
                "conflict": _is_conflict(rec), "kind": kind,
                "verses": [v for a in _accounts(rec) for v in a[3]],
                "n_accounts": _n_accounts(rec),
                "order": order,
            }
            lines += [f"### A{i:02d}", "", "**Source accounts (the only evidence):**",
                      ""]
            for gospel, ref, text, _vv in _accounts(rec):
                label = f"{gospel} ({ref})" if ref else gospel
                lines.append(f"- *{label}*: {text}")
            lines.append("")
            for slot, cond in zip("XYZ", order):
                lines += [f"**Version {slot}.** {variants[cond]}", "",
                          f"- A1 unsupported content (Yes/No + words): ",
                          f"- A2 completeness (3/2/1): ",
                          f"- A3 richness (3/2/1/NA): ", ""]
            lines += ["- A4 preferred version (X/Y/Z): ", "", "---", ""]

        lines += ["## Part B --- Ordering", "",
                  "For each pair the system placed the FIRST before the SECOND.",
                  "Answer Correct / Incorrect / Indeterminate.", ""]
        for j, (a, b, stratum) in enumerate(pairs, 1):
            key[f"R{rater}:B{j:02d}"] = {
                "first": _marker(a), "second": _marker(b), "stratum": stratum}
            lines += [f"### B{j:02d}", "",
                      f"**First.** {a.get('consolidated','')}", "",
                      f"**Second.** {b.get('consolidated','')}", "",
                      "- B1 (Correct / Incorrect / Indeterminate): ", "", "---", ""]

        (out / f"booklet_rater{rater}.md").write_text(
            "\n".join(lines), encoding="utf-8")

    (out / "KEY_DO_NOT_DISTRIBUTE.json").write_text(
        json.dumps({"seed": SEED, "conditions": list(CONDITIONS), "items": key},
                   indent=1), encoding="utf-8")

    strata = Counter(k["stratum"] for k in key.values() if "stratum" in k)
    nacc = Counter(k["n_accounts"] for k in key.values() if "n_accounts" in k)
    manifest = {
        "seed": SEED, "n_pairs": len(pairs), "raters": N_RATERS,
        "eligibility": diag,
        "events_by_account_count": {str(k): v // N_RATERS
                                    for k, v in sorted(nacc.items())},
        "pairs_by_stratum": {k: v // N_RATERS for k, v in strata.items()},
        "conflict_events": sum(1 for r in events if _is_conflict(r)),
        "notes": [
            "Condition 'longest' applies Timeline+Longest's selection policy to "
            "the induced clusters. State this exactly in the thesis: it is the "
            "policy, not the published system with its given timeline.",
            "The comparison sample is restricted to events where the three "
            "conditions produce different text. Report it as such: it is the "
            "discriminative subset, not an estimate over all events.",
            "The control items are single-account events where all three "
            "conditions are identical by construction. Any rating difference "
            "there is measurement noise and bounds the rest.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                       encoding="utf-8")

    print(json.dumps(manifest, indent=1))
    print(f"\n{N_RATERS} booklets + key written to {out}/")
    print("Send booklet_rater*.md to the raters. Do NOT send the key.")
    if not suspect:
        print("\nWARNING: no --suspect list given, so Part B is not stratified "
              "and the weighted estimate cannot be computed. Pass the transposed "
              "positions from the error taxonomy.")
    return 0


# --------------------------------------------------------------------------
# agreement statistics
# --------------------------------------------------------------------------
def _counts(ratings: List[List[str]], cats: Sequence[str]):
    return [[row.count(c) for c in cats] for row in ratings]


def fleiss_kappa(ratings: List[List[str]], cats: Sequence[str]) -> Optional[float]:
    tab = _counts(ratings, cats)
    tab = [row for row in tab if sum(row) > 1]
    if not tab:
        return None
    N = len(tab)
    m = sum(tab[0])
    p_a = sum((sum(x * x for x in row) - m) / (m * (m - 1)) for row in tab) / N
    p_j = [sum(row[j] for row in tab) / (N * m) for j in range(len(cats))]
    p_e = sum(p * p for p in p_j)
    return None if abs(1 - p_e) < 1e-12 else (p_a - p_e) / (1 - p_e)


def gwet_ac1(ratings: List[List[str]], cats: Sequence[str]) -> Optional[float]:
    """Gwet's AC1. Robust where Fleiss' kappa collapses on skewed marginals."""
    tab = _counts(ratings, cats)
    tab = [row for row in tab if sum(row) > 1]
    if not tab:
        return None
    N, q = len(tab), len(cats)
    m = sum(tab[0])
    p_a = sum((sum(x * x for x in row) - m) / (m * (m - 1)) for row in tab) / N
    pi = [sum(row[j] for row in tab) / (N * m) for j in range(q)]
    p_e = sum(p * (1 - p) for p in pi) / (q - 1) if q > 1 else 0.0
    return None if abs(1 - p_e) < 1e-12 else (p_a - p_e) / (1 - p_e)


def krippendorff_alpha_ordinal(ratings: List[List[str]],
                               cats: Sequence[str]) -> Optional[float]:
    """Krippendorff's alpha with the ordinal difference function.

    Coincidence-matrix formulation; `cats` must be given in rank order.
    """
    idx = {c: i for i, c in enumerate(cats)}
    q = len(cats)
    o = [[0.0] * q for _ in range(q)]
    for row in ratings:
        vals = [idx[v] for v in row if v in idx]
        m = len(vals)
        if m < 2:
            continue
        cnt = Counter(vals)
        for c in cnt:
            for k in cnt:
                pairs = cnt[c] * cnt[k] - (cnt[c] if c == k else 0)
                o[c][k] += pairs / (m - 1)
    n_c = [sum(o[c]) for c in range(q)]
    n = sum(n_c)
    if n < 2:
        return None

    def delta2(c: int, k: int) -> float:
        lo, hi = min(c, k), max(c, k)
        s = sum(n_c[lo:hi + 1]) - (n_c[lo] + n_c[hi]) / 2.0
        return s * s

    d_o = sum(o[c][k] * delta2(c, k) for c in range(q) for k in range(q))
    d_e = sum(n_c[c] * n_c[k] * delta2(c, k)
              for c in range(q) for k in range(q)) / (n - 1)
    return None if abs(d_e) < 1e-12 else 1.0 - d_o / d_e


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def score(args) -> int:
    d = Path(args.dir)
    key = json.loads((d / "KEY_DO_NOT_DISTRIBUTE.json").read_text())["items"]
    sheets = sorted(d.glob("ratings_rater*.csv"))
    if not sheets:
        print(f"No ratings_rater*.csv in {d}/. Expected columns: "
              "item,slot,A1,A2,A3,A4,B1")
        return 1

    # item -> criterion -> [rating per rater], keyed by condition for part A
    a: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    b: Dict[str, List[str]] = defaultdict(list)
    ctrl: Dict[str, List[str]] = defaultdict(list)
    for s in sheets:
        rater = s.stem.replace("ratings_rater", "")
        with s.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                item = (row.get("item") or "").strip()
                if not item:
                    continue
                ref = key.get(f"R{rater}:{item}")
                if not ref:
                    continue
                if item.startswith("A"):
                    if ref.get("kind") == "control":
                        slot = (row.get("slot") or "").strip().upper()
                        for crit in ("A1", "A2", "A3"):
                            v = (row.get(crit) or "").strip()
                            if v:
                                ctrl[f"{item}|{crit}"].append(v)
                        continue
                    slot = (row.get("slot") or "").strip().upper()
                    if slot not in ("X", "Y", "Z"):
                        continue
                    cond = ref["order"]["XYZ".index(slot)]
                    for crit in ("A1", "A2", "A3"):
                        v = (row.get(crit) or "").strip()
                        if v:
                            a[f"{cond}|{crit}"][item].append(v)
                    pref = (row.get("A4") or "").strip().upper()
                    if pref in ("X", "Y", "Z"):
                        a["preference|A4"][item].append(
                            ref["order"]["XYZ".index(pref)])
                elif item.startswith("B"):
                    v = (row.get("B1") or "").strip().capitalize()
                    if v:
                        b[item].append(v)

    print("=" * 62)
    print("AGREEMENT  (gate: 0.60; below it the criterion is not usable)")
    print("=" * 62)
    scales = {"A1": ["No", "Yes"], "A2": ["1", "2", "3"], "A3": ["1", "2", "3"]}
    for cond in CONDITIONS:
        for crit, cats in scales.items():
            rows = [v for v in a[f"{cond}|{crit}"].values() if len(v) > 1]
            if not rows:
                continue
            if crit == "A1":
                k1 = fleiss_kappa(rows, cats)
                ac = gwet_ac1(rows, cats)
                gate = "ok " if (ac or 0) >= 0.60 else "LOW"
                print(f"  {cond:12s} {crit}  kappa={_f(k1)}  AC1={_f(ac)}  {gate}")
            else:
                al = krippendorff_alpha_ordinal(rows, cats)
                gate = "ok " if (al or 0) >= 0.60 else "LOW"
                print(f"  {cond:12s} {crit}  alpha_o={_f(al)}          {gate}")

    brows = [v for v in b.values() if len(v) > 1]
    if brows:
        cats = ["Incorrect", "Indeterminate", "Correct"]
        print(f"  {'ordering':12s} B1  kappa={_f(fleiss_kappa(brows, cats))}"
              f"  AC1={_f(gwet_ac1(brows, cats))}")

    print()
    print("=" * 62)
    print("PART A  --- majority verdict over raters")
    print("=" * 62)
    for cond in CONDITIONS:
        hall = _majority(a[f"{cond}|A1"], "Yes")
        comp = _mean_ordinal(a[f"{cond}|A2"])
        rich = _mean_ordinal(a[f"{cond}|A3"])
        n = len(a[f"{cond}|A1"]) or 1
        lo, hi = wilson(hall, n)
        print(f"  {cond:12s} hallucination {hall}/{n} = {hall/n:.3f} "
              f"[{lo:.3f},{hi:.3f}]   completeness {comp}   richness {rich}")
    pref = Counter()
    for votes in a["preference|A4"].values():
        pref[Counter(votes).most_common(1)[0][0]] += 1
    tot = sum(pref.values()) or 1
    print("  preference:  " + "  ".join(
        f"{c}={pref[c]}/{tot} ({pref[c]/tot:.2f})" for c in CONDITIONS))

    print()
    print("=" * 62)
    print("PART B  --- ordering, raw and stratum-weighted")
    print("=" * 62)
    per_stratum: Dict[str, List[str]] = defaultdict(list)
    for item, votes in b.items():
        ref = next((key[k] for k in key if k.endswith(f":{item}")), None)
        if not ref or not votes:
            continue
        per_stratum[ref.get("stratum", "?")].append(
            Counter(votes).most_common(1)[0][0])

    raw_ok = raw_n = 0
    for st in sorted(per_stratum):
        v = per_stratum[st]
        ok = v.count("Correct")
        ind = v.count("Indeterminate")
        raw_ok += ok
        raw_n += len(v)
        print(f"  {st:12s} n={len(v):3d}  correct={ok/len(v):.3f}  "
              f"indeterminate={ind/len(v):.3f}")
    if raw_n:
        lo, hi = wilson(raw_ok, raw_n)
        print(f"  {'RAW':12s} n={raw_n:3d}  correct={raw_ok/raw_n:.3f} "
              f"[{lo:.3f},{hi:.3f}]   <- pessimistic: adversarial sample")

    if args.weights and per_stratum:
        w = json.loads(Path(args.weights).read_text())
        num = den = 0.0
        for st, v in per_stratum.items():
            if st in w and v:
                num += w[st] * v.count("Correct") / len(v)
                den += w[st]
        if den:
            print(f"  {'WEIGHTED':12s}       correct={num/den:.3f}   "
                  "<- comparable with the 0.9581 against the chronology")
    else:
        print("  WEIGHTED     not computed: pass --weights with the population "
              "share of each stratum")
    return 0


def _f(x: Optional[float]) -> str:
    return "  n/a " if x is None else f"{x:6.3f}"


def _majority(items: Dict[str, List[str]], target: str) -> int:
    n = 0
    for votes in items.values():
        if votes and Counter(votes).most_common(1)[0][0] == target:
            n += 1
    return n


def _mean_ordinal(items: Dict[str, List[str]]) -> str:
    vals = []
    for votes in items.values():
        nums = [float(v) for v in votes if v.replace(".", "").isdigit()]
        if nums:
            vals.append(sum(nums) / len(nums))
    return f"{sum(vals)/len(vals):.2f}" if vals else " n/a"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--abstractive", required=True)
    b.add_argument("--extractive", default="")
    b.add_argument("--suspect", default="",
                   help="JSON list of induced positions flagged as transposed")
    b.add_argument("--out", default="human_eval")
    b.set_defaults(fn=build)

    s = sub.add_parser("score")
    s.add_argument("--dir", default="human_eval")
    s.add_argument("--weights", default="",
                   help="JSON {stratum: population share} for the weighted estimate")
    s.set_defaults(fn=score)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
