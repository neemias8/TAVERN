#!/usr/bin/env python3
"""
Congela a amostra da avaliação humana por especialistas (Capítulo 10, RQ3).

    python avaliacao_humana/amostragem.py \\
        --curation outputs/ancoragem/curation.json \\
        --out avaliacao_humana/amostra_congelada.json

Desenho: amostragem estratificada uniforme sobre os grupos MULTI-FONTE, com
pesos w_h = N_h / n_h, de modo que a soma ponderada reproduz a população.
Estratos são o número de Evangelhos que relatam o episódio: 2, 3 ou 4.

Três decisões são deliberadas e ficam registradas no arquivo de saída, porque
cada uma muda o que a estimativa significa:

1. **Os pesos são derivados da rodada, não fixados no código.** A
   especificação de origem trazia N = (79, 50, 10) = 139 grupos; a rodada
   `ancoragem` tem (78, 67, 10) = 155. Números fixos teriam produzido uma
   estimativa populacional silenciosamente errada no estrato de 3 fontes.
   O arquivo grava a população medida e o dígito do artefato que a produziu.

2. **A amostra inclui os eventos que o backbone não conseguiu fundir.** 30
   dos 155 grupos multi-fonte saíram do fallback determinístico
   (`UnionFuser`) depois que o guarda de fidelidade rejeitou duas tentativas
   do modelo. Sortear apenas os eventos que o modelo produziu daria uma
   estimativa do TAVERN sem os seus próprios fracassos. O caminho de fusão
   fica gravado por item (`fusion`) e a análise reporta a estimativa com e
   sem eles.

3. **Só entram grupos em que os dois sistemas produzem texto diferente.** Um
   item em que TAVERN e Longest coincidem não discrimina nada e gastaria uma
   das dez vagas.

O cegamento é contrabalançado dentro de cada estrato: metade dos itens com
TAVERN como Sistema X, metade como Sistema Y. A chave sai em arquivo
separado, que não acompanha os cadernos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Dict, List

#: fixa a amostra; mudar isto sorteia outros episódios
SEMENTE = 20260918

#: itens por estrato (número de fontes -> quantos sortear)
PLANO = {2: 4, 3: 4, 4: 2}

#: episódio de nivelamento, fora da contagem estatística. Mateus 22:46 e
#: Marcos 12:34: duas fontes curtas e claramente complementares, que é o que
#: um item de calibração precisa ser.
PILOTO_REFS = ("Matthew 22:46", "Mark 12:34")

LIVROS = {"matthew": "Mateus", "mark": "Marcos",
          "luke": "Lucas", "john": "João"}


def digest(eventos: List[dict]) -> str:
    """O dígito por evento de scripts/verify_for_thesis.py.

    Amarra a amostra ao artefato exato: o Stage 4 não é reprodutível entre
    execuções, então duas rodadas da mesma configuração dão textos
    ligeiramente diferentes. Se este dígito não bater com o de
    `consolidations/`, os cadernos não são do artefato depositado.
    """
    linhas = sorted(f"{e['marker']}\t{' '.join(e['consolidated'].split())}"
                    for e in eventos)
    return hashlib.sha256("\n".join(linhas).encode()).hexdigest()


def longest(ev: dict) -> str:
    """A baseline extrativa: o relato individual mais longo, sem fusão."""
    return max((s["text"] for s in ev["sources"]), key=len).strip()


def referencia(s: dict) -> str:
    ref = s.get("ref", "")
    livro = LIVROS.get(s["gospel"], s["gospel"])
    # `ref` já vem como "Matthew 21:1-3"; troca só o nome do livro
    for en, pt in LIVROS.items():
        ref = ref.replace(en.capitalize(), pt)
    return ref or livro


def item(ev: dict, codigo: str, peso: float, papel_tavern: str) -> dict:
    tav = ev["consolidated"].strip()
    lon = longest(ev)
    return {
        "codigo": codigo,
        "marker": ev["marker"],
        "posicao": ev.get("position"),
        "n_fontes": len(ev["sources"]),
        "estrato": len(ev["sources"]),
        "peso": peso,
        "conflito": bool(ev.get("conflicted")),
        "fusion": ev.get("fusion", "desconhecido"),
        "fontes": [{"evangelho": LIVROS.get(s["gospel"], s["gospel"]),
                    "referencia": referencia(s),
                    "texto": s["text"].strip()} for s in ev["sources"]],
        "sistemas": {"X": tav if papel_tavern == "X" else lon,
                     "Y": lon if papel_tavern == "X" else tav},
        "_cegamento": {"X": "tavern" if papel_tavern == "X" else "longest",
                       "Y": "longest" if papel_tavern == "X" else "tavern"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curation", type=Path,
                    default=Path("outputs/ancoragem/curation.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("avaliacao_humana/amostra_congelada.json"))
    ap.add_argument("--chave", type=Path,
                    default=Path("avaliacao_humana/chave_cegamento.json"))
    args = ap.parse_args()

    dados = json.loads(args.curation.read_text(encoding="utf-8"))
    eventos = dados["events"]
    dig = digest(eventos)

    multi = [e for e in eventos if len(e["sources"]) > 1]
    populacao: Dict[int, int] = {}
    for e in multi:
        populacao[len(e["sources"])] = populacao.get(len(e["sources"]), 0) + 1

    piloto_ev = next(
        (e for e in multi
         if {s.get("ref", "") for s in e["sources"]} >= set(PILOTO_REFS)),
        None)
    if piloto_ev is None:
        raise SystemExit(f"episódio piloto {PILOTO_REFS} não encontrado; "
                         f"escolha outro em --curation")

    # elegíveis: os dois sistemas têm de diferir, e o piloto sai do sorteio
    def elegivel(e: dict) -> bool:
        return (e["marker"] != piloto_ev["marker"]
                and " ".join(e["consolidated"].split())
                != " ".join(longest(e).split()))

    rng = random.Random(SEMENTE)
    itens, pesos, diag = [], {}, {}
    n_item = 0
    for estrato in sorted(PLANO):
        pool = sorted((e for e in multi
                       if len(e["sources"]) == estrato and elegivel(e)),
                      key=lambda e: e["marker"])
        n = PLANO[estrato]
        diag[str(estrato)] = {"populacao": populacao.get(estrato, 0),
                              "elegiveis": len(pool), "sorteados": n}
        if len(pool) < n:
            raise SystemExit(f"estrato {estrato}: {len(pool)} elegíveis para "
                             f"{n} vagas")
        # o peso é da POPULAÇÃO do estrato, não do conjunto elegível: a
        # estimativa é sobre todos os grupos daquele tamanho
        w = populacao[estrato] / n
        pesos[str(estrato)] = round(w, 4)
        escolhidos = rng.sample(pool, n)
        # contrabalanceamento dentro do estrato: metade X, metade Y
        papeis = ["X", "Y"] * (n // 2) + (["X"] if n % 2 else [])
        rng.shuffle(papeis)
        for ev, papel in zip(escolhidos, papeis):
            n_item += 1
            itens.append(item(ev, f"A{n_item:02d}", w, papel))

    rng.shuffle(itens)
    for i, it in enumerate(itens, 1):          # renumera após embaralhar
        it["codigo"] = f"A{i:02d}"

    piloto = item(piloto_ev, "PA01", 0.0, "X")
    piloto["peso"] = 0.0
    piloto["observacao"] = ("item de nivelamento; não entra em nenhuma "
                            "estimativa")

    soma = sum(it["peso"] for it in itens)
    saida = {
        "gerado_em": date.today().isoformat(),
        "semente": SEMENTE,
        "artefato": {
            "curation": str(args.curation).replace("\\", "/"),
            "tag": "ancoragem",
            "backbone": dados.get("backbone"),
            "digest_por_evento": dig,
            "eventos_totais": len(eventos),
        },
        "populacao": {"multi_fonte": len(multi), "por_estrato": {
            str(k): v for k, v in sorted(populacao.items())}},
        "plano_amostral": {str(k): v for k, v in sorted(PLANO.items())},
        "pesos": pesos,
        "soma_ponderada": round(soma, 4),
        "confere_populacao": abs(soma - len(multi)) < 1e-6,
        "diagnostico_por_estrato": diag,
        "caminho_de_fusao_na_amostra": {
            p: sum(1 for it in itens if it["fusion"] == p)
            for p in sorted({it["fusion"] for it in itens})},
        "piloto": {k: v for k, v in piloto.items() if k != "_cegamento"},
        "itens": [{k: v for k, v in it.items() if k != "_cegamento"}
                  for it in itens],
    }
    chave = {
        "semente": SEMENTE,
        "digest_por_evento": dig,
        "aviso": "CONFIDENCIAL — não acompanha os cadernos.",
        "piloto": {"codigo": "PA01", **piloto["_cegamento"]},
        "itens": {it["codigo"]: {**it["_cegamento"], "marker": it["marker"],
                                 "estrato": it["estrato"], "peso": it["peso"],
                                 "fusion": it["fusion"]}
                  for it in itens},
        "balanco": {
            "tavern_como_X": sum(1 for it in itens
                                 if it["_cegamento"]["X"] == "tavern"),
            "tavern_como_Y": sum(1 for it in itens
                                 if it["_cegamento"]["Y"] == "tavern")},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(saida, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    args.chave.write_text(json.dumps(chave, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    print(f"artefato          {dig[:12]}  ({len(eventos)} eventos)")
    print(f"população         {len(multi)} grupos multi-fonte "
          f"{saida['populacao']['por_estrato']}")
    print(f"pesos             {pesos}")
    print(f"soma ponderada    {soma:.2f}  confere={saida['confere_populacao']}")
    print(f"caminho de fusão  {saida['caminho_de_fusao_na_amostra']}")
    print(f"cegamento         TAVERN como X em "
          f"{chave['balanco']['tavern_como_X']} itens, como Y em "
          f"{chave['balanco']['tavern_como_Y']}")
    print(f"piloto            {piloto['marker']} "
          f"({', '.join(f['referencia'] for f in piloto['fontes'])})")
    print(f"\n{args.out}\n{args.chave}  (CONFIDENCIAL)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
