#!/usr/bin/env python3
"""
Addendum 16, Task 2 -- recall by EXPRESSION, not by verse, over the 80
anchorable TIMEX3s (thesis Table tab:ann-density).

The verse-level criterion of Addendum 15 (25/80 = 31.2%) overstates
HeidelTime's actual coverage: a verse counts as "recovered" if HeidelTime
fires anywhere in it, even on unrelated text. Mark 14:1 counts as recovered
because HeidelTime finds "Now" and "two days" while missing "the Passover
and the Feast of Unleavened Bread" -- the actual anchorable expression --
entirely. This measures whether HeidelTime's OWN detected text actually
overlaps the anchorable expression itself.

Method used (the "shortcut" Addendum 16 sanctions if the token-id ->
character-offset path costs too much time): the token layer
(outputs/ancoragem/annotation/<book>.tokens.xml) carries each token's LEMMA,
not its surface offset, so exact character-span overlap is not available
without re-running Stage 1 segmentation. Instead: the anchorable TIMEX3's
own @target token range is resolved to its lemma sequence, and a verse
counts as recovered-by-expression if HeidelTime's detected surface text
(lower-cased, tokenised, suffix-stripped with the same stemmer redundancy.py
uses) shares at least one content word with that lemma sequence. This is a
lexical-overlap approximation, not a verified character-span overlap --
reported as exactly that, not as the stricter criterion.

    python scripts/heideltime_expression_recall.py
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tavern.stage6_evaluation.redundancy import _STOP, _TOKEN, _stem

ANNOTATION_DIR = ROOT / "outputs" / "ancoragem" / "annotation"
HEIDELTIME_JSON = ROOT / "outputs" / "heideltime" / "agreement.json"
BOOK_ORDER = ["matthew", "mark", "luke", "john"]

ISO_NS = "http://www.iso.org/ns/semaf-time"
TVN_NS = "http://tavern.unisinos.br/ns/isotimeml-ext"


def _content_words(text: str) -> set:
    return {_stem(m.group(0)) for m in _TOKEN.finditer(text)
           if len(m.group(0)) > 2 and m.group(0).lower() not in _STOP}


def _expand_target(target: str) -> List[str]:
    """'#tk_mk11_1_0_2' -> ['tk_mk11_1_0_2']; '#range(a,b)' -> [a..b]."""
    target = target.lstrip("#")
    if target.startswith("range("):
        a, b = target[len("range("):-1].split(",")
        prefix = a.rsplit("_", 1)[0]
        lo = int(a.rsplit("_", 1)[1])
        hi = int(b.rsplit("_", 1)[1])
        return [f"{prefix}_{i}" for i in range(lo, hi + 1)]
    return [target]


def load_lemmas(book: str) -> Dict[str, str]:
    root = ET.parse(ANNOTATION_DIR / f"{book}.tokens.xml").getroot()
    out = {}
    for tk in root.iter("tk"):
        xid = tk.get("{http://www.w3.org/XML/1998/namespace}id")
        out[xid] = tk.get("lemma", "")
    return out


def load_anchorable(book: str) -> List[Tuple[Tuple[int, int], str, str]]:
    """[(verse_key, timex_id, pred), ...] for anchorable TIMEX3, with their
    resolved lemma text stashed by caller via load_lemmas."""
    root = ET.parse(ANNOTATION_DIR / f"{book}.tml").getroot()
    out = []
    for tx in root.iter(f"{{{ISO_NS}}}TIMEX3"):
        if tx.get(f"{{{TVN_NS}}}anchorable") != "true":
            continue
        verse_attr = tx.get(f"{{{TVN_NS}}}verse")
        if not verse_attr:
            continue
        _b, c, v = verse_attr.split(":")
        target = tx.get("target")
        out.append(((int(c), int(v)), tx.get("{http://www.w3.org/XML/1998/namespace}id"),
                   tx.get("pred"), target))
    return out


def main() -> int:
    d = json.loads(HEIDELTIME_JSON.read_text(encoding="utf-8"))
    heidel_by_book_verse: Dict[str, Dict[Tuple[int, int], List[dict]]] = defaultdict(
        lambda: defaultdict(list))
    for det in d["heideltime_detections"]:
        heidel_by_book_verse[det["book"]][(det["chapter"], det["verse"])].append(det)

    total_anchorable = 0
    total_verse_fires = 0
    total_expression_recovered = 0
    rows = []

    for book in BOOK_ORDER:
        lemmas = load_lemmas(book)
        anchors = load_anchorable(book)
        for verse_key, timex_id, pred, target in anchors:
            total_anchorable += 1
            token_ids = _expand_target(target)
            anchor_words = {_stem(lemmas[t]) for t in token_ids
                            if t in lemmas and len(lemmas[t]) > 2}
            dets = heidel_by_book_verse[book].get(verse_key, [])
            if not dets:
                rows.append((book, verse_key, pred, "no_heideltime", None))
                continue
            total_verse_fires += 1
            recovered = False
            matched_text = None
            for det in dets:
                hwords = _content_words(det["heideltime_text"])
                if hwords & anchor_words:
                    recovered = True
                    matched_text = det["heideltime_text"]
                    break
            if recovered:
                total_expression_recovered += 1
                rows.append((book, verse_key, pred, "recovered", matched_text))
            else:
                rows.append((book, verse_key, pred, "verse_only",
                            [dt["heideltime_text"] for dt in dets]))

    # per-verse aggregation (a verse can carry >1 anchorable expression,
    # e.g. Mark 14:1 has both PASSOVER and UNLEAVENED_BREAD): recovered if
    # ANY of its anchorable expressions is lexically matched -- this is the
    # "over 80" figure Addendum 16 Task 2 asks for, alongside the 112-
    # expression figure (matches Addendum 9's 42/112, 46/112 denominators).
    anchorable_verses = {(b, vk) for b, vk, _p, _s, _e in rows}
    recovered_verses = {(b, vk) for b, vk, _p, s, _e in rows if s == "recovered"}

    print(f"anchorable expressions: {total_anchorable}  "
         f"(anchorable verses: {len(anchorable_verses)})")
    print(f"verses where HeidelTime fires at all: {total_verse_fires}")
    print(f"recovered by expression, per EXPRESSION: "
         f"{total_expression_recovered}/{total_anchorable} "
         f"({total_expression_recovered/total_anchorable:.1%})")
    print(f"recovered by expression, per VERSE (>=1 of its expressions "
         f"matched): {len(recovered_verses)}/{len(anchorable_verses)} "
         f"({len(recovered_verses)/len(anchorable_verses):.1%})")
    print()
    print("--- verse-only fires (HeidelTime detected something in the "
         "verse, but not the anchorable expression itself) ---")
    for book, (c, v), pred, status, extra in rows:
        if status == "verse_only":
            print(f"  {book} {c}:{v}  anchor={pred}  heideltime found: {extra}")
    print()
    print("--- recovered by expression ---")
    for book, (c, v), pred, status, extra in rows:
        if status == "recovered":
            print(f"  {book} {c}:{v}  anchor={pred}  heideltime text: {extra!r}")

    out_path = ROOT / "outputs" / "heideltime" / "expression_recall.json"
    out_path.write_text(json.dumps({
        "method": "lexical_overlap_lemma_stem",
        "note": "approximation: token layer has no character offsets, so "
               "this checks content-word overlap between HeidelTime's "
               "detected text and the anchorable TIMEX3's own token span's "
               "lemmas, not verified character-span overlap.",
        "anchorable_total": total_anchorable,
        "anchorable_verses": len(anchorable_verses),
        "verse_fires": total_verse_fires,
        "expression_recovered": total_expression_recovered,
        "expression_recall": round(total_expression_recovered / total_anchorable, 4),
        "recovered_verses": len(recovered_verses),
        "expression_recall_per_verse": round(
            len(recovered_verses) / len(anchorable_verses), 4),
        "rows": [{"book": b, "chapter": c, "verse": v, "pred": p,
                 "status": s, "detail": e}
                for b, (c, v), p, s, e in rows],
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
