#!/usr/bin/env python3
"""
Run everything and package the results in one file.

    python run_all.py                      # abstractive via Ollama + extractive
    python run_all.py --model gemma3:4b
    python run_all.py --skip-extractive    # if you already have that run
    python run_all.py --backbone union     # no model needed, for a dry run

Two configurations are measured, because the thesis needs both and they are not
the same kind of system:

  extractive   one account per event, verbatim. Comparable with the published
               degradation curve, whose rows are extractive.
  abstractive  per-event fusion, which is what the framework is for.

The generation run is 248 model calls and is cached to disk as it goes, so an
interrupted run resumes where it stopped: re-run the same command.

At the end you get `tavern_results_<timestamp>.zip`. Send that file back and the
thesis tables can be filled from it.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OLLAMA_TAGS = "http://localhost:11434/api/tags"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def rule(title: str) -> None:
    say()
    say("=" * 76)
    say(title)
    say("=" * 76)


# ---------------------------------------------------------------------------
def check_python() -> bool:
    ok = True
    for mod, hint in (("spacy", "pip install spacy"),
                      ("networkx", "pip install networkx"),
                      ("scipy", "pip install scipy"),
                      ("rouge_score", "pip install rouge-score --no-build-isolation"),
                      ("lxml", "pip install lxml"),
                      ("nltk", "pip install nltk")):
        try:
            __import__(mod)
        except ImportError:
            say(f"  MISSING  {mod:12s} -> {hint}")
            ok = False
    try:
        import pylcs                      # noqa: F401
        say("  ok       pylcs (compiled LCS)")
    except ImportError:
        say("  absent   pylcs -> optional; ROUGE-L uses the pure-Python "
            "bit-parallel fallback")
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except Exception:
        say("  MISSING  en_core_web_sm -> python -m spacy download en_core_web_sm")
        ok = False
    try:
        import torch
        say(f"  ok       torch {torch.__version__}")
    except ImportError:
        say("  absent   torch -> Stage 4 falls back to unpropagated features "
            "(pip install torch)")
    return ok


def ollama_models() -> list:
    try:
        with urllib.request.urlopen(OLLAMA_TAGS, timeout=5) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def check_ollama(model: str) -> bool:
    models = ollama_models()
    if not models:
        say("  Ollama is not answering on localhost:11434.")
        say("    1. install from https://ollama.com/download")
        say("    2. leave it running (it starts a background service)")
        say(f"    3. ollama pull {model}")
        return False
    say(f"  ok       Ollama is up, {len(models)} model(s) installed")
    if model in models or any(m.split(":")[0] == model.split(":")[0]
                              for m in models):
        say(f"  ok       {model} present")
        return True
    say(f"  MISSING  {model} -> ollama pull {model}")
    say(f"           installed: {', '.join(models)}")
    return False


# ---------------------------------------------------------------------------
def run(cmd: list, label: str) -> bool:
    say()
    say(f"--- {label}")
    say(f"    $ {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        say(f"    FAILED after {dt/60:.1f} min (exit {proc.returncode})")
        return False
    say(f"    done in {dt/60:.1f} min")
    return True


def package(tags: list, model: str, backbone: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M")
    out = ROOT / f"tavern_results_{stamp}.zip"
    manifest = {
        "created": stamp,
        "backbone": backbone,
        "ollama_model": model,
        "tags": tags,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
    }
    try:
        import torch
        manifest["torch"] = torch.__version__
        manifest["cuda"] = torch.cuda.is_available()
        manifest["xpu"] = bool(getattr(torch, "xpu", None)
                               and torch.xpu.is_available())
    except ImportError:
        manifest["torch"] = None

    keep = ("results.json", "curation.json", "consolidated.txt",
            "consolidated_with_markers.txt", "stage3/timeline.json")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        for tag in tags:
            base = OUTPUTS / tag
            if not base.exists():
                continue
            for rel in keep:
                p = base / rel
                if p.exists():
                    z.write(p, f"{tag}/{rel}")
            ann = base / "annotation"
            if ann.exists():
                for p in sorted(ann.glob("*.tml")):
                    z.write(p, f"{tag}/annotation/{p.name}")
        cur = ROOT / "consolidations"
        if cur.exists():
            for p in sorted(cur.rglob("*")):
                if p.is_file() and p.suffix in (".md", ".csv", ".txt"):
                    z.write(p, f"consolidations/{p.relative_to(cur)}")
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="ollama",
                    choices=("ollama", "instruct", "bart", "pegasus",
                             "primera", "union"))
    ap.add_argument("--model", default="gemma3:4b",
                    help="Ollama model, or the HF checkpoint for the others")
    ap.add_argument("--skip-extractive", action="store_true")
    ap.add_argument("--skip-abstractive", action="store_true")
    args = ap.parse_args()

    rule("TAVERN — full run")
    say("Two configurations will be measured, then packaged into one zip.")

    rule("1. Environment")
    if not check_python():
        say()
        say("Install the missing pieces above and run again.")
        return 1

    needs_ollama = args.backbone == "ollama" and not args.skip_abstractive
    if needs_ollama and not check_ollama(args.model):
        say()
        say("Fix the Ollama setup above and run again, or use")
        say("  python run_all.py --backbone union    (no model needed)")
        return 1

    rule("2. Corpus")
    sys.path.insert(0, str(ROOT))
    from tavern.config import verify_corpus
    for name, (status, _e, _a) in verify_corpus(strict=False).items():
        say(f"  {status:9s} {name}")
    try:
        verify_corpus()
    except RuntimeError as exc:
        say(f"\n  {exc}")
        return 1

    py = sys.executable
    tags = []

    if not args.skip_extractive:
        rule("3. Extractive configuration  (~12 min)")
        say("This is the row comparable with the published degradation curve.")
        if not run([py, "run_experiments.py", "--all", "--tag", "extractive",
                    "--backbone", "extractive"], "extractive"):
            return 1
        tags.append("extractive")

    if not args.skip_abstractive:
        rule(f"4. Abstractive configuration  ('{args.backbone}')")
        say("248 per-event fusions. Cached as it goes: if this is interrupted,")
        say("run the same command again and it resumes.")
        say("Expect 20-90 min with a GPU, longer on CPU only.")
        cmd = [py, "run_experiments.py", "--all", "--tag", args.backbone,
               "--backbone", args.backbone]
        if args.model:
            cmd += ["--backbone-model", args.model]
        if not run(cmd, f"abstractive ({args.backbone})"):
            say()
            say("Partial fusions are cached in")
            say(f"  outputs/{args.backbone}/fusion_cache.jsonl")
            say("so re-running resumes rather than starting over.")
            return 1
        tags.append(args.backbone)

        rule("5. Curation sheets")
        run([py, "scripts/make_curation.py",
             f"outputs/{args.backbone}/curation.json",
             f"consolidations/{args.backbone}"], "curation")

    rule("6. Packaging")
    zip_path = package(tags, args.model if needs_ollama else "",
                       args.backbone)
    size = zip_path.stat().st_size / 1e6
    say(f"  {zip_path.name}  ({size:.1f} MB)")
    say()
    say("Send that file back and the thesis tables can be filled from it.")
    say("It contains, per configuration: results.json with every measured")
    say("figure, the curation records, the consolidated narrative, and the")
    say("conformant .tml annotation.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nInterrupted. Re-run the same command to resume.")
        sys.exit(130)
