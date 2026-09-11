#!/usr/bin/env python3
"""
Addendum 15 -- HeidelTime as a second, independent temporal annotator.

Not a gold standard: an independent measurement instrument for thesis
Section 9.3.6's limitation ("no intrinsic annotation metric exists"). Where
TAVERN and HeidelTime disagree, that is disagreement, reported in both
directions -- never a correction applied back to either.

Deliberately standalone: imports NOTHING from `tavern.stage1`...`stage6`,
touches no `TavernConfig`, is never called from `run_experiments.py`.
Reads the raw corpus XML directly (its own small parser, not
`Corpus`/`ReferenceParser`) and the already-serialised `outputs/ancoragem/
annotation/*.tml` (plain `xml.etree`, not the annotation model classes) --
so nothing HeidelTime returns can reach Layer A, `@value`, the coreference
score, or any pipeline run, even by accident of a shared import.

    pip install py_heideltime emoji   # emoji is undeclared but required:
                                        # py_heideltime's import breaks without it
    python tools/heideltime_agreement.py

Needs a JRE on PATH (`java -version`); no manual TreeTagger or Java install
beyond that -- the wheel bundles `de.unihd.dbs.heideltime.standalone.jar`
and a full TreeTagger (English parameters, Linux and Windows binaries).

Runs HeidelTime per CHAPTER, not per verse: each call starts a JVM, so
per-verse is ~4.6s/call (~5h for 1,245 verses) against ~6.6s for a whole
47-verse chapter (~3 min for the corpus). Each call chdir's into its own
temp directory first: `py_heideltime` writes `./config.props` at the start
of a call and deletes it at the end, so two calls sharing a working
directory race and the loser gets zero detections silently, no error --
run sequentially (this script does) or isolate each worker's CWD if you
parallelise it later.

A fourth pitfall, not in Addendum 15's list, found running this on Windows
with anaconda under ``C:/ProgramData/anaconda3/...``:
``py_heideltime.config._write_config_props()`` writes the TreeTagger path
into ``config.props`` with plain Windows backslashes
(``str(Path.absolute())``), and a Java ``.properties`` file treats an
unescaped backslash as the start of an escape sequence -- an unrecognised
one is simply dropped, so a path with backslash separators is read back by
HeidelTime with the separators removed, TreeTagger's own path resolution
breaks, and
`heideltime()` returns silently empty output -- no exception, `stderr` is
captured and discarded by `py_heideltime.utils.execute_command`, so nothing
surfaces unless you rerun the subprocess yourself with `stderr` printed.
This is very likely why an earlier attempt at HeidelTime on Windows failed
outright. Worked around below by monkeypatching `_write_config_props` to
write the tagger path with forward slashes instead (Java's `File` accepts
`/` on Windows natively) -- nothing in the installed package is modified,
only this process's imported copy of the function.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANNOTATION_DIR = ROOT / "outputs" / "ancoragem" / "annotation"
OUT_DIR = ROOT / "outputs" / "heideltime"

GOSPEL_FILES = {
    "matthew": "EnglishNIVMatthew40_PW.xml",
    "mark": "EnglishNIVMark41_PW.xml",
    "luke": "EnglishNIVLuke42_PW.xml",
    "john": "EnglishNIVJohn43_PW.xml",
}
GOSPEL_SCOPE = {
    "matthew": (21, 28), "mark": (11, 16), "luke": (19, 24), "john": (12, 20),
}
BOOK_ORDER = ["matthew", "mark", "luke", "john"]

XML_NS = "http://www.w3.org/XML/1998/namespace"
TVN_NS = "http://tavern.unisinos.br/ns/isotimeml-ext"
ISO_NS = "http://www.iso.org/ns/semaf-time"
DCT = "2020-01-01"   # arbitrary and fixed; the corpus has no document creation
                     # time (thesis Section 6.2.5) -- any @value that depends
                     # on this DCT is not read here for exactly that reason


# ---------------------------------------------------------------------------
# 1. Raw corpus text, own minimal parser -- no tavern import at all.
def load_verses(book: str) -> List[Tuple[int, int, str]]:
    path = DATA_DIR / GOSPEL_FILES[book]
    root = ET.parse(path).getroot()
    book_node = root.find(".//book")
    lo, hi = GOSPEL_SCOPE[book]
    out: List[Tuple[int, int, str]] = []
    for chapter in book_node.findall("chapter"):
        cnum = int(chapter.get("number"))
        if not (lo <= cnum <= hi):
            continue
        for verse in chapter.findall("verse"):
            vnum = int(verse.get("number"))
            text = (verse.text or "").strip()
            if text:
                out.append((cnum, vnum, text))
    return out


def chapter_texts(book: str) -> Dict[int, Tuple[str, List[Tuple[int, int, int]]]]:
    """chapter -> (joined_text, [(verse_num, start_offset, end_offset), ...])."""
    by_chapter: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for c, v, t in load_verses(book):
        by_chapter[c].append((v, t))
    out = {}
    for c, vs in by_chapter.items():
        parts, spans, pos = [], [], 0
        for v, t in vs:
            parts.append(t)
            spans.append((v, pos, pos + len(t)))
            pos += len(t) + 1          # single space joiner between verses
        out[c] = (" ".join(parts), spans)
    return out


# ---------------------------------------------------------------------------
# 2. HeidelTime, per chapter, CWD-isolated, span-validated.
def _patch_windows_config_props_bug() -> None:
    """See the module docstring: on Windows, py_heideltime's own
    _write_config_props() writes the TreeTagger path with backslashes,
    which the Java .properties reader mangles, silently breaking
    TreeTagger. Replace it with a version that writes forward slashes.
    Only this process's imported copy of the function is touched."""
    import platform
    if platform.system() != "Windows":
        return
    import py_heideltime.config as cfg_mod
    import py_heideltime.py_heideltime as impl_mod

    def _fixed() -> None:
        lib = Path(cfg_mod.__file__).parent
        tagger_path = lib / "Heideltime" / "TreeTaggerWindows"
        template = (lib / "resources" / "config_props_template").open().read()
        content = template.replace("{path}", tagger_path.absolute().as_posix())
        Path("config.props").open("w").write(content)

    cfg_mod._write_config_props = _fixed
    impl_mod._write_config_props = _fixed


def run_heideltime(text: str):
    _patch_windows_config_props_bug()
    from py_heideltime import heideltime
    workdir = tempfile.mkdtemp(prefix="heideltime_")
    prev = os.getcwd()
    try:
        os.chdir(workdir)
        return heideltime(text, language="english",
                          document_type="narrative", dct=DCT)
    finally:
        os.chdir(prev)


def map_to_verses(timexes, chapter_text: str,
                  verse_spans: List[Tuple[int, int, int]]) -> List[dict]:
    out = []
    dropped = 0
    for tx in (timexes or []):
        span = tx.get("span")
        if not span or len(span) != 2:
            dropped += 1
            continue
        s, e = span
        if chapter_text[s:e] != tx.get("text", ""):
            dropped += 1            # span does not validate; discard, don't guess
            continue
        verse = next((v for v, vs, ve in verse_spans if vs <= s < ve), None)
        if verse is None:
            dropped += 1
            continue
        out.append({"verse": verse, "text": tx.get("text"),
                    "type": tx.get("type"), "value": tx.get("value"),
                    "span": span})
    return out, dropped


# ---------------------------------------------------------------------------
# 3. TAVERN's own TIMEX3 (and EVENT eligibility, for the veridicality check),
#    read straight from the serialised .tml -- xml.etree only, no model import.
def load_tavern(book: str):
    path = ANNOTATION_DIR / f"{book}.tml"
    root = ET.parse(path).getroot()

    timexes_by_verse: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    anchorable_verses: set = set()
    no_verse = 0
    for tx in root.iter(f"{{{ISO_NS}}}TIMEX3"):
        verse_attr = tx.get(f"{{{TVN_NS}}}verse")
        if not verse_attr:
            no_verse += 1            # the empty anchor forest of Section 6.2.5:
            continue                 # no textual realisation, no external
                                      # annotator could find these by construction
        _b, c, v = verse_attr.split(":")
        c, v = int(c), int(v)
        timexes_by_verse[(c, v)].append({
            "type": tx.get("type"), "value": tx.get("value"),
            "pred": tx.get("pred"),
        })
        if tx.get(f"{{{TVN_NS}}}anchorable") == "true":
            anchorable_verses.add((c, v))

    eligible_by_verse: Dict[Tuple[int, int], List[bool]] = defaultdict(list)
    for ev in root.iter(f"{{{ISO_NS}}}EVENT"):
        verse_attr = ev.get(f"{{{TVN_NS}}}verse")
        if not verse_attr:
            continue
        _b, c, v = verse_attr.split(":")
        eligible = ev.get(f"{{{TVN_NS}}}eligible") == "true"
        eligible_by_verse[(int(c), int(v))].append(eligible)

    return timexes_by_verse, anchorable_verses, eligible_by_verse, no_verse


def verse_is_subordinated(eligible_by_verse, c: int, v: int) -> Optional[bool]:
    flags = eligible_by_verse.get((c, v))
    if not flags:
        return None
    return sum(flags) < len(flags) / 2.0   # majority non-eligible


def _latex_table(per_book: dict, total: dict) -> str:
    rows = []
    for book in BOOK_ORDER:
        b = per_book[book]
        rows.append(
            f"{book.capitalize():8s} & {b['only_tavern']:3d} & "
            f"{b['only_heideltime']:3d} & {b['both']:3d} & "
            f"{b['anchorable_found_by_heideltime']:2d}/{b['anchorable_verses']:2d} \\\\")
    total_row = (
        f"\\textbf{{Total}} & \\textbf{{{total['only_tavern']}}} & "
        f"\\textbf{{{total['only_heideltime']}}} & \\textbf{{{total['both']}}} & "
        f"\\textbf{{{total['anchorable_found_by_heideltime']}/"
        f"{total['anchorable_verses']}}} \\\\")
    body = "\n".join(rows)
    return (
        "% Addendum 15 -- TAVERN vs HeidelTime, an independent second\n"
        "% annotator, not a gold standard. Generated by\n"
        "% tools/heideltime_agreement.py; do not hand-edit, regenerate.\n"
        "\\begin{table}[h]\n\\centering\n"
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Gospel & only TAVERN & only HeidelTime & both & anchorable found \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\midrule\n"
        f"{total_row}\n"
        "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{TAVERN vs.\\ HeidelTime per-verse TIMEX3 agreement. "
        f"Type agreement where both fire: "
        f"{total['type_agreement_rate']:.1%} ({total['type_match']}/"
        f"{total['type_compared']}). "
        f"``anchorable found'' is of the "
        f"{total['anchorable_verses']} anchorable verses (Table "
        "tab:ann-density), how many HeidelTime also detects "
        f"({total['anchorable_recall']:.1%}).}}\n"
        "\\label{tab:heideltime-agreement}\n\\end{table}\n"
    )


# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_book = {}
    total_only_tavern = total_only_heidel = total_both = total_neither = 0
    total_type_match = total_type_compared = 0
    anchorable_total = anchorable_found = 0
    missed_by_tavern: List[dict] = []
    heideltime_detections: List[dict] = []
    dropped_total = 0
    no_verse_total = 0

    for book in BOOK_ORDER:
        print(f"=== {book} ===", flush=True)
        tavern_tx, anchorable, eligible_by_verse, no_verse = load_tavern(book)
        no_verse_total += no_verse
        anchorable_total += len(anchorable)

        chapters = chapter_texts(book)
        heidel_tx: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
        for c, (text, spans) in sorted(chapters.items()):
            raw = run_heideltime(text)
            mapped, dropped = map_to_verses(raw, text, spans)
            dropped_total += dropped
            for m in mapped:
                heidel_tx[(c, m["verse"])].append(m)
            print(f"  chapter {c}: {len(mapped)} validated TIMEX3 "
                 f"({dropped} dropped)", flush=True)

        verse_text_of = {(c, v): t for c, v, t in load_verses(book)}
        all_verses = set(verse_text_of)
        only_tavern = only_heidel = both = neither = 0
        type_match = type_compared = 0
        for key in all_verses:
            has_t = key in tavern_tx
            has_h = key in heidel_tx
            c, v = key
            sub = verse_is_subordinated(eligible_by_verse, c, v)

            if has_h:
                for e in heidel_tx[key]:
                    entry = {"book": book, "chapter": c, "verse": v,
                            "verse_text": verse_text_of[key],
                            "heideltime_text": e["text"], "type": e["type"],
                            "value": e["value"], "also_tavern": has_t,
                            "subordinated": sub}
                    heideltime_detections.append(entry)
                    if not has_t:
                        missed_by_tavern.append(entry)

            if has_t and has_h:
                both += 1
                t_types = {e["type"] for e in tavern_tx[key]}
                h_types = {e["type"] for e in heidel_tx[key] if e["type"]}
                if h_types:
                    type_compared += 1
                    if t_types & h_types:
                        type_match += 1
            elif has_t:
                only_tavern += 1
            elif has_h:
                only_heidel += 1
            else:
                neither += 1

        found = sum(1 for key in anchorable if key in heidel_tx)
        anchorable_found += found

        per_book[book] = {
            "only_tavern": only_tavern, "only_heideltime": only_heidel,
            "both": both, "neither": neither,
            "type_match": type_match, "type_compared": type_compared,
            "anchorable_verses": len(anchorable),
            "anchorable_found_by_heideltime": found,
        }
        total_only_tavern += only_tavern
        total_only_heidel += only_heidel
        total_both += both
        total_neither += neither
        total_type_match += type_match
        total_type_compared += type_compared

    sub_counts = Counter(
        "subordinated" if d["subordinated"] else
        ("narrative" if d["subordinated"] is False else "unknown")
        for d in heideltime_detections)

    summary = {
        "dct_used": DCT,
        "empty_anchor_forest_excluded": no_verse_total,
        "dropped_unvalidated_spans": dropped_total,
        "per_book": per_book,
        "total": {
            "only_tavern": total_only_tavern,
            "only_heideltime": total_only_heidel,
            "both": total_both,
            "neither": total_neither,
            "type_match": total_type_match,
            "type_compared": total_type_compared,
            "type_agreement_rate": round(total_type_match / total_type_compared, 4)
                if total_type_compared else None,
            "anchorable_verses": anchorable_total,
            "anchorable_found_by_heideltime": anchorable_found,
            "anchorable_recall": round(anchorable_found / anchorable_total, 4)
                if anchorable_total else None,
        },
        "heideltime_detections_subordination": dict(sub_counts),
        "missed_by_tavern": missed_by_tavern,
        "heideltime_detections": heideltime_detections,
    }

    out_path = OUT_DIR / "agreement.json"
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    tex_path = OUT_DIR / "agreement_table.tex"
    tex_path.write_text(_latex_table(per_book, summary["total"]), encoding="utf-8")

    print("\n=== TOTAL ===")
    print(f"only TAVERN: {total_only_tavern}  only HeidelTime: {total_only_heidel}  "
         f"both: {total_both}  neither: {total_neither}")
    print(f"type agreement (where both fire): {summary['total']['type_agreement_rate']} "
         f"({total_type_match}/{total_type_compared})")
    print(f"anchorable verses: {anchorable_total}, found by HeidelTime: "
         f"{anchorable_found} ({summary['total']['anchorable_recall']})")
    print(f"empty anchor forest excluded (no tvn:verse): {no_verse_total}")
    print(f"dropped (span did not validate): {dropped_total}")
    print(f"HeidelTime detections by subordination: {dict(sub_counts)}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
