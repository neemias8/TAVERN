#!/usr/bin/env python3
"""
A cadeia até Kendall's τ é determinística dada a mesma entrada?

    python scripts/tau_determinism.py --n 3
    python scripts/tau_determinism.py --n 3 --sem-gnn

Executa o pipeline N vezes no MESMO processo não — cada repetição refaz
stages 1 a 4 do zero, porque `pipeline.run` é o que a tese reporta, e o
objetivo é medir o que um leitor obteria reexecutando o comando.

Registra por execução: τ, τ pairwise, cobertura (numerador/denominador),
número de clusters, arcos removidos no torneio, conflitos, e o dígito do
agrupamento (um SHA-1 da partição em si, que muda se QUALQUER unidade trocar
de cluster, mesmo quando as métricas agregadas não se movem).

Por que o dígito do agrupamento importa: τ e cobertura são estatísticas
resumo e podem coincidir por compensação. A partição não compensa.

O que este script NÃO mede: se duas execuções de CÓDIGOS diferentes dão
valores diferentes. Isso não é indeterminismo, é outra configuração, e a
comparação correta é um diff de código -- não uma faixa de variação.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

#: executado em subprocesso para que cada repetição seja um processo novo:
#: caches em memória, ordem de iteração de dicionários e estado de RNG global
#: não atravessam o limite do processo, que é exatamente o que se quer testar.
FILHO = r'''
import hashlib, json, sys
sys.path.insert(0, %r)
from tavern import pipeline
from tavern.config import TavernConfig
from tavern.stage6_evaluation import chronology as chrono_mod, timeline_eval

cfg = TavernConfig(tag="determinismo", backbone="extractive")
res = pipeline.run(cfg, with_gnn=%s, write=False, verify=False)
ch = chrono_mod.load(res.corpus)
ev = timeline_eval.evaluate(ch, res.stage3.clustering, res.stage3.induced,
                            res.units)
cl = res.stage3.clustering
# dígito da PARTIÇÃO: para cada cluster, as unidades ordenadas; clusters
# ordenados entre si. Independe da ordem de criação e dos identificadores.
blocos = sorted("|".join(sorted(c.members)) for c in cl.clusters)
dig = hashlib.sha1("\n".join(blocos).encode()).hexdigest()
ordem = hashlib.sha1("\n".join(res.stage3.induced.order).encode()).hexdigest()
print("__RESULTADO__" + json.dumps({
    "tau": ev.tau, "pairwise": ev.pairwise_accuracy,
    "coverage": ev.coverage,
    "coberto": ev.matched_events,
    "total": ev.total_events,
    "clusters": len(cl.clusters),
    "unidades": len(res.units),
    "conflitos": len(res.stage3.induced.conflicted_clusters(cl)),
    "digest_particao": dig,
    "digest_ordem": ordem,
}))
'''


def uma(com_gnn: bool) -> dict:
    r = subprocess.run([sys.executable, "-c", FILHO % (str(RAIZ), com_gnn)],
                       capture_output=True, text=True, cwd=str(RAIZ))
    for linha in r.stdout.splitlines():
        if linha.startswith("__RESULTADO__"):
            return json.loads(linha[len("__RESULTADO__"):])
    print(r.stdout[-2000:])
    print(r.stderr[-2000:])
    raise SystemExit("execução falhou")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--sem-gnn", action="store_true",
                    help="isola stages 1-3; o GNN é Stage 4 e alimenta apenas "
                         "a seleção, nunca o agrupamento ou a ordem")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    com_gnn = not args.sem_gnn
    print(f"{args.n} execuções, processo novo a cada uma, "
          f"GNN {'ligado' if com_gnn else 'desligado'}\n")
    linhas = []
    for i in range(1, args.n + 1):
        d = uma(com_gnn)
        linhas.append(d)
        print(f"  #{i}  tau={d['tau']:.6f}  pairwise={d['pairwise']}  "
              f"cobertura={d['coverage']:.6f} ({d['coberto']}/{d['total']})  "
              f"clusters={d['clusters']}  conflitos={d['conflitos']}")
        print(f"      partição {d['digest_particao'][:16]}  "
              f"ordem {d['digest_ordem'][:16]}")

    print()
    chaves = ["tau", "pairwise", "coverage", "coberto", "clusters",
              "conflitos", "digest_particao", "digest_ordem"]
    estavel = True
    for k in chaves:
        vals = {json.dumps(l[k]) for l in linhas}
        if len(vals) == 1:
            print(f"  {k:18} IDÊNTICO em {args.n} execuções")
        else:
            estavel = False
            v = [l[k] for l in linhas]
            print(f"  {k:18} VARIA: {v}")
    print()
    if estavel:
        print("VEREDICTO: determinístico ATÉ o τ, com envelope a jusante.")
        print("  Medido: stages 1-3 -> agrupamento -> ordem induzida -> τ.")
        print("  O Stage 4 NÃO entra aqui porque não realimenta o Stage 3,")
        print("  mas tem ruído de execução (§9.6): move a seleção extrativa")
        print("  em ~3 dos 289 eventos entre execuções, e por ela o texto")
        print("  gerado. Não escreva 'o pipeline é determinístico'.")
    else:
        print("VEREDICTO: há variação entre execuções — reporte a faixa, "
              "não um valor pontual.")
    if args.out:
        args.out.write_text(json.dumps(linhas, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
