#!/usr/bin/env python3
"""
A/B de prompts de fusão sobre os eventos que mais repetem.

    python scripts/fusion_prompt_ab.py --n 8
    python scripts/fusion_prompt_ab.py --n 8 --model gemma3:4b --variantes v3,simples

Por que existe: o prompt em produção (`build_prompt`, v3) manda o modelo
tomar o relato mais longo como espinha, **manter a redação dele** e inserir o
que os outros acrescentam. Isso é uma instrução de acréscimo. Ela resolveu a
perda de detalhe e a invenção -- ambas foram a zero -- e não resolveu a
redundância, porque nunca manda colapsar nada: medido sobre os 155 eventos
multi-fonte, a repetição excedente mediana da fusão (0.115) é
indistinguível da baseline determinística `union` (0.125).

O instrumento é `fidelity.excess_repetition`, que conta quantas vezes a fusão
repete um termo de conteúdo além do que qualquer relato isolado o repete. Ele
existe porque `internal_redundancy` compara 5-gramas e só enxerga repetição
literal: em E060 a mesma profecia chega três vezes, uma por Evangelista, com
as mesmas palavras de conteúdo e quase nenhum 5-grama em comum, e o evento
passou com 0.24 contra um limiar de 0.25.

Não é um seletor automático de prompt. Roda a grade, imprime as quatro
medidas lado a lado e deixa a escolha para quem lê -- com n = 8 episódios
qualquer critério automático estaria ajustando contra a amostra.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tavern.stage5_generation import fidelity as F          # noqa: E402
from tavern.stage5_generation.backbones import (            # noqa: E402
    OLLAMA_STOP, UnionFuser, _budget, _clean, build_prompt)

#: v3 é o prompt em produção; as demais são as candidatas. Cada uma muda UMA
#: coisa em relação à anterior, para que a diferença medida seja atribuível.
VARIANTES: Dict[str, str] = {

    # ---- candidata 1: sem método, uma instrução dominante ----------------
    "simples":
        "Below are {n} accounts of one event, written by different people.\n"
        "\n"
        "Write the event as ONE paragraph of continuous narrative.\n"
        "\n"
        "Say each thing once. Where the accounts report the same thing, "
        "write it a single time, in one set of words. Where only one account "
        "has a detail, keep that detail. Add nothing that no account has. "
        "Do not name or number the accounts.\n"
        "{conflict}"
        "\n"
        "Write only the paragraph.\n",

    # ---- candidata 2: um narrador reconta, uma vez -----------------------
    "reconto":
        "{n} people below describe the same event.\n"
        "\n"
        "You have read all {n}. Now tell that event once, as a single "
        "narrator would, in one paragraph of continuous prose.\n"
        "\n"
        "- One telling, not {n} tellings joined together.\n"
        "- When two of them report the same action or the same speech, it "
        "happens once in your telling, not once per person.\n"
        "- A detail only one of them gives still belongs in your telling.\n"
        "- Nothing may appear that none of them gives: no names, places, "
        "numbers or motives of your own.\n"
        "- Use their wording where you can, but do not repeat a sentence "
        "just because two of them wrote it.\n"
        "{conflict}"
        "\n"
        "Write only the paragraph.\n",

    # ---- candidata 3: método, mas o passo 1 é colapsar -------------------
    "colapso":
        "Below are {n} accounts of the same event.\n"
        "\n"
        "Method:\n"
        "1. Find what the accounts have in common and write it ONCE, in one "
        "set of words. This is the backbone of your paragraph.\n"
        "2. Go through each account again and add only what it alone "
        "contributes, folding each addition into the sentence it belongs to "
        "rather than appending a new one.\n"
        "3. Read your paragraph back. If any fact, question or line of "
        "speech appears twice, delete the second one.\n"
        "\n"
        "Rules:\n"
        "- Every detail from every account appears, exactly once.\n"
        "- Invent nothing: no names, places, numbers, motives or events.\n"
        "- Do not name or number the accounts; do not comment.\n"
        "{conflict}"
        "\n"
        "Write only the paragraph.\n",
}

CONFLITO = {
    "simples":
        "The accounts disagree about the order or circumstances: keep both "
        "readings, joined in one sentence with \"while\" or \"although\".\n",
    "reconto":
        "- They disagree about the order or circumstances. Keep both "
        "readings in your telling, joined with \"while\" or \"although\"; do "
        "not choose one and do not drop either.\n",
    "colapso":
        "- The accounts disagree about the order or circumstances of the "
        "event. Keep both readings, joined in one sentence with \"while\" or "
        "\"although\".\n",
}


def prompt_de(variante: str, n: int, conflito: bool) -> str:
    if variante == "v3":
        return build_prompt(n, conflito)
    return VARIANTES[variante].format(
        n=n, conflict=CONFLITO[variante] if conflito else "")


def gerar(modelo: str, prompt: str, textos: Sequence[str],
          endpoint: str, pensar: bool = False) -> str:
    corpo = "\n\n".join(f"Account {i + 1}: {t.strip()}"
                        for i, t in enumerate(textos))
    payload = json.dumps({
        "model": modelo, "prompt": prompt + "\n" + corpo, "stream": False,
        # gemma4 e os demais modelos de raciocínio gastam o num_predict
        # inteiro no campo `thinking` e devolvem `response` vazio, sem erro
        # algum. Desligar o raciocínio é o que os torna utilizáveis aqui;
        # a chave é ignorada pelos modelos que não pensam.
        "think": pensar,
        "options": {"num_predict": _budget(textos), "temperature": 0.0,
                    "seed": 0, "repeat_penalty": 1.1, "stop": OLLAMA_STOP},
    }).encode()
    req = urllib.request.Request(endpoint, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return _clean(json.loads(r.read()).get("response", ""))


def medidas(texto: str, fontes: Sequence[str]) -> dict:
    v = F.check(texto, fontes)
    return {"excesso": F.excess_repetition(texto, fontes),
            "rep5": v.redundancy, "recall": v.recall, "novo": v.novel,
            "ok": v.ok, "palavras": len(texto.split())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curation", type=Path,
                    default=Path("outputs/ancoragem/curation.json"))
    ap.add_argument("--n", type=int, default=8,
                    help="quantos episódios, os de maior repetição excedente")
    ap.add_argument("--model", default="gemma3:4b")
    ap.add_argument("--variantes", default="v3,simples,reconto,colapso")
    ap.add_argument("--endpoint",
                    default="http://localhost:11434/api/generate")
    ap.add_argument("--pensar", action="store_true",
                    help="deixa o modelo de raciocínio pensar (padrão: não; "
                         "com raciocínio ligado ele consome num_predict "
                         "inteiro e devolve texto vazio)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    eventos = json.loads(args.curation.read_text(encoding="utf-8"))["events"]
    multi = [e for e in eventos if len(e["sources"]) > 1]
    for e in multi:
        e["_exc"] = F.excess_repetition(
            e["consolidated"], [s["text"] for s in e["sources"]])
    alvo = sorted(multi, key=lambda e: -e["_exc"])[:args.n]
    variantes = [v.strip() for v in args.variantes.split(",") if v.strip()]
    uf = UnionFuser()

    print(f"modelo {args.model} | {len(alvo)} episódios | "
          f"variantes {', '.join(variantes)}\n")
    linhas: List[dict] = []
    for e in alvo:
        S = [s["text"] for s in e["sources"]]
        conf = bool(e.get("conflicted"))
        base = medidas(e["consolidated"], S)
        uni = medidas(uf.fuse(S), S)
        print(f"{e['marker']}  {len(S)} fontes"
              f"{'  CONFLITO' if conf else ''}"
              f"   (em produção: excesso {base['excesso']:.3f}, "
              f"{base['palavras']} palavras; union {uni['excesso']:.3f})")
        linhas.append({"marker": e["marker"], "variante": "produção",
                       "modelo": "gemma3:4b", **base})
        linhas.append({"marker": e["marker"], "variante": "union",
                       "modelo": "-", **uni})
        for v in variantes:
            t0 = time.time()
            try:
                texto = gerar(args.model, prompt_de(v, len(S), conf), S,
                              args.endpoint, args.pensar)
            except Exception as exc:                     # noqa: BLE001
                print(f"    {v:9} ERRO {type(exc).__name__}: {exc}")
                continue
            if not texto.strip():
                print(f"    {v:9} VAZIO (o modelo não devolveu texto em "
                      f"'response'; modelos de raciocínio precisam de "
                      f"/api/chat)")
                continue
            m = medidas(texto, S)
            linhas.append({"marker": e["marker"], "variante": v,
                           "modelo": args.model, "texto": texto, **m})
            print(f"    {v:9} excesso {m['excesso']:.3f}  rep5 {m['rep5']:.2f}"
                  f"  recall {m['recall']:.2f}  novo {m['novo']:.2f}"
                  f"  {m['palavras']:4d} palavras  {time.time() - t0:.0f}s"
                  f"  {'ok' if m['ok'] else 'REPROVA'}")
        print()

    print("=" * 78)
    print(f"{'variante':12}{'excesso':>9}{'rep5':>8}{'recall':>8}{'novo':>7}"
          f"{'palavras':>10}{'reprovas':>10}")
    import statistics as st
    for v in ["produção", "union"] + variantes:
        g = [l for l in linhas if l["variante"] == v]
        if not g:
            continue
        print(f"{v:12}{st.median(l['excesso'] for l in g):>9.3f}"
              f"{st.median(l['rep5'] for l in g):>8.2f}"
              f"{st.median(l['recall'] for l in g):>8.2f}"
              f"{st.median(l['novo'] for l in g):>7.2f}"
              f"{st.median(l['palavras'] for l in g):>10.0f}"
              f"{sum(1 for l in g if not l['ok']):>10d}")
    print("\nmedianas sobre os episódios MAIS repetitivos da rodada, não "
          "sobre a rodada inteira: é o recorte onde a diferença aparece, e "
          "não é uma estimativa do efeito médio.")

    if args.out:
        args.out.write_text(json.dumps(linhas, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
