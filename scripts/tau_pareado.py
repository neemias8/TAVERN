#!/usr/bin/env python3
"""
Compara duas execuções sobre a MESMA população de eventos casados.

    python scripts/tau_pareado.py \\
        --a outputs/ancoragem/curation.json \\
        --b .../outputs/tese_principal_20260915/curation.json

τ de duas execuções só é comparável quando ambos medem o mesmo conjunto.
0,9274 sobre 149 eventos casados e 0,9340 sobre 154 não são o mesmo número
medido duas vezes: são duas amostras diferentes da mesma população de 168.
Um τ maior sobre um conjunto maior pode significar ordem melhor, ou apenas
que os cinco eventos a mais caíram em trechos fáceis.

Este script casa e ordena cada execução com o instrumento do próprio
repositório (`timeline_eval.match_clusters_to_events`, Jaccard exato sobre
endereços livro:capítulo:versículo, atribuição gulosa e um-para-um),
reimplementado aqui sobre `curation.json` porque a execução B vem de outro
checkout e seus objetos Python não são importáveis daqui. A reimplementação
é validada reproduzindo o τ publicado de cada execução antes de qualquer
comparação pareada; se não reproduzir, o script para.

Reporta também quantos clusters têm partição idêntica entre as duas, medida
sobre conjuntos de versículos — a única chave comum, já que os
identificadores de unidade não atravessam checkouts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.stats import kendalltau                      # noqa: E402

from tavern.stage1_preprocessing.corpus import Corpus    # noqa: E402
from tavern.stage6_evaluation import chronology as chrono_mod   # noqa: E402

Chave = Tuple[str, int, int]


def carregar(p: Path) -> Tuple[Dict[str, Set[Chave]], Dict[str, int]]:
    """cluster -> conjunto de versículos, e cluster -> posição induzida."""
    d = json.loads(p.read_text(encoding="utf-8"))
    chaves: Dict[str, Set[Chave]] = {}
    pos: Dict[str, int] = {}
    for e in d["events"]:
        cid = e["cluster"]
        ks: Set[Chave] = set()
        for s in e["sources"]:
            for v in s.get("verses", []):
                livro, cap, ver = v.split(":")
                ks.add((livro, int(cap), int(ver)))
        chaves[cid] = ks
        pos[cid] = int(e["position"])
    return chaves, pos


def casar(ch, chaves: Dict[str, Set[Chave]], min_overlap: float = 0.10
          ) -> Dict[int, str]:
    """`timeline_eval.match_clusters_to_events`, sobre os conjuntos acima."""
    scored: List[Tuple[float, int, str]] = []
    for ev in ch.events:
        ekeys = set(ev.all_keys)
        if not ekeys:
            continue
        for cid, ckeys in chaves.items():
            inter = ekeys & ckeys
            if not inter:
                continue
            jac = len(inter) / len(ekeys | ckeys)
            rec = len(inter) / len(ekeys)
            scored.append((0.5 * jac + 0.5 * rec, ev.event_id, cid))
    scored.sort(reverse=True)
    usados_c: Set[str] = set()
    usados_e: Set[int] = set()
    m: Dict[int, str] = {}
    for s, eid, cid in scored:
        if eid in usados_e or cid in usados_c or s < min_overlap:
            continue
        m[eid] = cid
        usados_e.add(eid)
        usados_c.add(cid)
    return m


def tau_sobre(eids, ranks, matching, pos):
    pares = sorted((ranks[e], pos[matching[e]]) for e in eids)
    if len(pares) < 3:
        return None, None
    ref = [p[0] for p in pares]
    hyp = [p[1] for p in pares]
    t, _ = kendalltau(ref, hyp)
    conc = disc = 0
    for i in range(len(pares)):
        for j in range(i + 1, len(pares)):
            a = hyp[i] - hyp[j]
            if a == 0:
                continue
            conc += a < 0
            disc += a > 0
    return float(t), (conc / (conc + disc) if conc + disc else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path,
                    default=Path("outputs/ancoragem/curation.json"))
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--tau-a", type=float, default=0.9274,
                    help="τ publicado de A, para validar a reimplementação")
    ap.add_argument("--tau-b", type=float, default=0.934)
    ap.add_argument("--tol", type=float, default=0.001)
    args = ap.parse_args()

    ch = chrono_mod.load(Corpus())
    ranks = ch.rank()
    total = len([e for e in ch.events if e.all_keys])

    ka, pa = carregar(args.a)
    kb, pb = carregar(args.b)
    ma, mb = casar(ch, ka), casar(ch, kb)

    ta, pwa = tau_sobre(ma, ranks, ma, pa)
    tb, pwb = tau_sobre(mb, ranks, mb, pb)
    print(f"A  {args.a}")
    print(f"   {len(ka)} clusters | casados {len(ma)}/{total} "
          f"({len(ma)/total:.4f}) | tau {ta:.4f} | pairwise {pwa:.4f}")
    print(f"B  {args.b}")
    print(f"   {len(kb)} clusters | casados {len(mb)}/{total} "
          f"({len(mb)/total:.4f}) | tau {tb:.4f} | pairwise {pwb:.4f}")

    ok = (abs(ta - args.tau_a) <= args.tol and abs(tb - args.tau_b) <= args.tol)
    print(f"\nvalidação da reimplementação: A esperado {args.tau_a} obtido "
          f"{ta:.4f}; B esperado {args.tau_b} obtido {tb:.4f} -> "
          f"{'confere' if ok else 'NÃO CONFERE'}")
    if not ok:
        print("Parando: sem reproduzir os τ publicados, o pareado não vale.")
        return 1

    inter = sorted(set(ma) & set(mb))
    tia, pwia = tau_sobre(inter, ranks, ma, pa)
    tib, pwib = tau_sobre(inter, ranks, mb, pb)
    print(f"\n=== PAREADO, sobre os {len(inter)} eventos casados por AMBAS ===")
    print(f"   tau A | interseção  {tia:.4f}   (pairwise {pwia:.4f})")
    print(f"   tau B | interseção  {tib:.4f}   (pairwise {pwib:.4f})")
    print(f"   diferença pareada   {tib - tia:+.4f}   "
          f"(não pareada: {tb - ta:+.4f})")
    so_a = sorted(set(ma) - set(mb))
    so_b = sorted(set(mb) - set(ma))
    print(f"   casados só por A: {len(so_a)} {so_a[:12]}")
    print(f"   casados só por B: {len(so_b)} {so_b[:12]}")

    # ---- partição: quantos clusters de A existem idênticos em B -----------
    conj_b = {frozenset(v) for v in kb.values() if v}
    conj_a = {frozenset(v) for v in ka.values() if v}
    iguais = len(conj_a & conj_b)
    print(f"\n=== PARTIÇÃO (conjuntos de versículos) ===")
    print(f"   clusters em A: {len(conj_a)} | em B: {len(conj_b)}")
    print(f"   idênticos nas duas: {iguais}")
    print(f"   só em A: {len(conj_a - conj_b)} | só em B: {len(conj_b - conj_a)}")
    multi_a = {c for c in conj_a if len({k[0] for k in c}) > 1}
    multi_b = {c for c in conj_b if len({k[0] for k in c}) > 1}
    print(f"   multi-livro: A {len(multi_a)}, B {len(multi_b)}, "
          f"idênticos {len(multi_a & multi_b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
