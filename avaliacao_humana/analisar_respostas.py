#!/usr/bin/env python3
"""
Analisa as respostas dos especialistas e gera as tabelas do Capítulo 10.

    python avaliacao_humana/analisar_respostas.py \\
        --respostas avaliacao_humana/respostas_avaliadores.json \\
        --out avaliacao_humana/resultados

Produz, em `--out`:

    resultados.md            tudo, para leitura
    tab_medias.tex           médias ponderadas por critério e sistema
    tab_concordancia.tex     Krippendorff ordinal e Fleiss por critério
    tab_testes.tex           Wilcoxon pareado TAVERN x Longest
    resultados.json          os mesmos números, para reuso

Três decisões de método, todas visíveis na saída:

* **A concordância é medida por critério, não em bloco.** A1 e A2 são
  construtos diferentes e não há razão para um α único; um valor agregado
  esconderia justamente a possibilidade interessante, que é os avaliadores
  concordarem sobre fidelidade e divergirem sobre fluência.

* **A estimativa populacional usa w_h = N_h/n_h da amostra congelada.** Os
  pesos não estão escritos aqui: vêm do arquivo, que os derivou da rodada.
  A saída confere que Σw = N antes de usar.

* **Tudo é reportado duas vezes, com e sem os itens que o backbone não
  fundiu.** Quando o guarda de fidelidade rejeita duas tentativas do modelo,
  o fallback determinístico assume o evento; esses itens são saída do
  sistema e pertencem à estimativa, mas uma afirmação sobre o que a *LLM*
  produz precisa do recorte sem eles.

Krippendorff (α ordinal) e Fleiss (κ) são implementados aqui, sem
dependência externa: são dez linhas cada um e uma dependência a menos é uma
reprodução a mais.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

CRITERIOS = ["A1", "A2", "A3", "A4"]
NOMES = {"A1": "Completude de detalhes",
         "A2": "Fidelidade factual (não-alucinação)",
         "A3": "Fluência, coesão e legibilidade",
         "A4": "Eliminação de redundância"}
SISTEMAS = ("tavern", "longest")
ROTULO = {"tavern": "TAVERN", "longest": "Longest (extrativo)"}


# ------------------------------------------------------------ concordância
def _delta2_ordinal(valores: Sequence[int], n: Dict[int, float]):
    """Matriz de diferenças ordinais de Krippendorff.

    δ²(c,k) = ( Σ_{g=c..k} n_g − (n_c + n_k)/2 )² , com n_g as frequências
    marginais. É o que distingue o α ordinal do nominal: errar por um ponto
    numa escala de 4 não é o mesmo que errar por três.
    """
    d = {}
    for c in valores:
        for k in valores:
            lo, hi = (c, k) if c <= k else (k, c)
            s = sum(n[g] for g in valores if lo <= g <= hi)
            d[(c, k)] = (s - (n[c] + n[k]) / 2.0) ** 2
    return d


def krippendorff_ordinal(unidades: Sequence[Sequence[Optional[int]]]
                         ) -> Optional[float]:
    """α ordinal sobre unidades × avaliadores, tolerando faltantes."""
    coinc: Dict[Tuple[int, int], float] = defaultdict(float)
    for u in unidades:
        vals = [v for v in u if v is not None]
        m = len(vals)
        if m < 2:
            continue
        for i, c in enumerate(vals):
            for j, k in enumerate(vals):
                if i != j:
                    coinc[(c, k)] += 1.0 / (m - 1)
    if not coinc:
        return None
    valores = sorted({c for c, _ in coinc} | {k for _, k in coinc})
    n = {g: sum(v for (c, k), v in coinc.items() if c == g) for g in valores}
    n_tot = sum(n.values())
    if n_tot <= 1:
        return None
    d2 = _delta2_ordinal(valores, n)
    Do = sum(coinc[(c, k)] * d2[(c, k)] for c in valores for k in valores
             if (c, k) in coinc) / n_tot
    De = sum(n[c] * n[k] * d2[(c, k)] for c in valores for k in valores) \
        / (n_tot * (n_tot - 1))
    if De == 0:
        return 1.0 if Do == 0 else None
    return 1.0 - Do / De


def fleiss_kappa(unidades: Sequence[Sequence[Optional[int]]],
                 categorias: Sequence[int] = (1, 2, 3, 4)) -> Optional[float]:
    """κ de Fleiss, exigindo o mesmo número de avaliadores por unidade."""
    linhas = [[v for v in u if v is not None] for u in unidades]
    linhas = [l for l in linhas if len(l) >= 2]
    if not linhas:
        return None
    m = len(linhas[0])
    if any(len(l) != m for l in linhas):
        return None                       # κ de Fleiss não tolera faltantes
    N = len(linhas)
    cont = np.array([[Counter(l)[c] for c in categorias] for l in linhas],
                    dtype=float)
    P_i = (cont ** 2).sum(axis=1) - m
    P_i = P_i / (m * (m - 1))
    p_j = cont.sum(axis=0) / (N * m)
    P_bar = P_i.mean()
    P_e = float((p_j ** 2).sum())
    if abs(1 - P_e) < 1e-12:
        return None
    return (P_bar - P_e) / (1 - P_e)


# ------------------------------------------------------------- estimativas
def media_ponderada(valores: Sequence[float], pesos: Sequence[float]
                    ) -> Optional[float]:
    if not valores:
        return None
    v = np.asarray(valores, dtype=float)
    w = np.asarray(pesos, dtype=float)
    return float((v * w).sum() / w.sum())


def carregar(respostas: Path, amostra: Path, chave: Path):
    R = json.loads(respostas.read_text(encoding="utf-8"))
    A = json.loads(amostra.read_text(encoding="utf-8"))
    K = json.loads(chave.read_text(encoding="utf-8"))
    if K["digest_por_evento"] != A["artefato"]["digest_por_evento"]:
        raise SystemExit("amostra e chave vêm de artefatos diferentes")
    d = (R.get("digest") or "").strip()
    if d and not A["artefato"]["digest_por_evento"].startswith(d):
        print(f"AVISO: o dígito digitado ({d}) não confere com o da amostra "
              f"({A['artefato']['digest_por_evento'][:16]}) — as respostas "
              f"podem ser de outro conjunto de cadernos")
    if not A["confere_populacao"]:
        raise SystemExit("amostra congelada inconsistente: Σw ≠ N")
    return R, A, K


def nota(R: dict, av: str, item: str, sistema: str, crit: str,
         K: dict) -> Optional[int]:
    """A nota que `av` deu ao SISTEMA (tavern/longest), desfazendo o cegamento."""
    slot = "X" if K["itens"][item]["X"] == sistema else "Y"
    v = R["avaliadores"].get(av, {}).get("itens", {}).get(item, {}) \
         .get(slot, {}).get(crit)
    return int(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    base = Path("avaliacao_humana")
    ap.add_argument("--respostas", type=Path,
                    default=base / "respostas_avaliadores.json")
    ap.add_argument("--amostra", type=Path,
                    default=base / "amostra_congelada.json")
    ap.add_argument("--chave", type=Path, default=base / "chave_cegamento.json")
    ap.add_argument("--out", type=Path, default=base / "resultados")
    args = ap.parse_args()

    R, A, K = carregar(args.respostas, args.amostra, args.chave)
    args.out.mkdir(parents=True, exist_ok=True)
    avs = sorted(R["avaliadores"])
    itens = [it["codigo"] for it in A["itens"]]
    peso = {it["codigo"]: it["peso"] for it in A["itens"]}
    fusao = {it["codigo"]: it["fusion"] for it in A["itens"]}
    estrato = {it["codigo"]: it["estrato"] for it in A["itens"]}
    so_modelo = [c for c in itens if fusao[c] != "union"]

    md: List[str] = []
    out: Dict[str, object] = {"artefato": A["artefato"], "pesos": A["pesos"]}
    md.append("# Avaliação por especialistas — resultados\n")
    md.append(f"Artefato `{A['artefato']['digest_por_evento'][:16]}`, "
              f"{A['artefato']['eventos_totais']} eventos, backbone "
              f"`{A['artefato']['backbone']}`. Amostra estratificada de "
              f"{len(itens)} itens sobre {A['populacao']['multi_fonte']} "
              f"grupos multi-fonte; pesos {A['pesos']}, Σw = "
              f"{A['soma_ponderada']:.2f}. Avaliadores: {', '.join(avs)}.\n")
    md.append(f"Caminho de fusão na amostra: "
              f"{A['caminho_de_fusao_na_amostra']}. "
              f"{len(itens) - len(so_modelo)} item(ns) vieram do fallback "
              f"determinístico e são reportados à parte.\n")

    # ---------------------------------------------------- 1. concordância
    md.append("\n## 1. Concordância entre avaliadores\n")
    md.append("| Critério | Sistema | Krippendorff α (ordinal) | Fleiss κ |")
    md.append("|---|---|---|---|")
    conc = {}
    linhas_tex = []
    for crit in CRITERIOS:
        for sis in SISTEMAS:
            unidades = [[nota(R, av, c, sis, crit, K) for av in avs]
                        for c in itens]
            a = krippendorff_ordinal(unidades)
            f = fleiss_kappa(unidades)
            conc[f"{crit}:{sis}"] = {"alpha": a, "kappa": f}
            fa = "n/d" if a is None else f"{a:.3f}"
            ff = "n/d" if f is None else f"{f:.3f}"
            md.append(f"| {crit} — {NOMES[crit]} | {ROTULO[sis]} | {fa} | {ff} |")
            linhas_tex.append(f"{crit} & {ROTULO[sis]} & {fa} & {ff} \\\\")
    md.append("\nα ordinal penaliza um erro de três pontos mais do que três "
              "erros de um ponto; κ de Fleiss trata as quatro categorias como "
              "nominais e é por construção mais baixo numa escala ordenada. "
              "A divergência entre os dois é esperada e não indica erro.\n")
    out["concordancia"] = conc

    # ------------------------------------------- 2. médias ponderadas
    md.append("\n## 2. Médias ponderadas por critério\n")
    md.append("| Critério | TAVERN | Longest | Δ | TAVERN (só modelo) | "
              "Longest (só modelo) |")
    md.append("|---|---|---|---|---|---|")
    medias, tex_medias = {}, []
    for crit in CRITERIOS:
        linha = {}
        for subset, rot in ((itens, "todos"), (so_modelo, "so_modelo")):
            for sis in SISTEMAS:
                vals, ws = [], []
                for c in subset:
                    ns = [nota(R, av, c, sis, crit, K) for av in avs]
                    ns = [n for n in ns if n is not None]
                    if ns:
                        vals.append(float(np.mean(ns)))
                        ws.append(peso[c])
                linha[f"{sis}_{rot}"] = media_ponderada(vals, ws)
        medias[crit] = linha
        f = lambda v: "n/d" if v is None else f"{v:.2f}"
        delta = (None if linha["tavern_todos"] is None
                 or linha["longest_todos"] is None
                 else linha["tavern_todos"] - linha["longest_todos"])
        md.append(f"| {crit} — {NOMES[crit]} | {f(linha['tavern_todos'])} | "
                  f"{f(linha['longest_todos'])} | "
                  f"{'n/d' if delta is None else f'{delta:+.2f}'} | "
                  f"{f(linha['tavern_so_modelo'])} | "
                  f"{f(linha['longest_so_modelo'])} |")
        tex_medias.append(
            f"{crit} & {f(linha['tavern_todos'])} & "
            f"{f(linha['longest_todos'])} & "
            f"{'n/d' if delta is None else f'{delta:+.2f}'} \\\\")
    out["medias"] = medias

    # --------------------------------------------------- 3. preferência
    md.append("\n## 3. Preferência global (A5)\n")
    pref = Counter()
    pref_pond = defaultdict(float)
    for av in avs:
        for c in itens:
            v = R["avaliadores"][av]["itens"].get(c, {}).get("A5")
            if not v:
                continue
            alvo = ("indiferente" if v == "indiferente"
                    else K["itens"][c][v])
            pref[alvo] += 1
            pref_pond[alvo] += peso[c]
    tot = sum(pref.values()) or 1
    tot_w = sum(pref_pond.values()) or 1.0
    md.append("| Preferido | Julgamentos | Proporção bruta | "
              "Proporção ponderada |")
    md.append("|---|---|---|---|")
    for k in ("tavern", "longest", "indiferente"):
        md.append(f"| {ROTULO.get(k, 'Indiferente')} | {pref[k]} | "
                  f"{pref[k]/tot:.1%} | {pref_pond[k]/tot_w:.1%} |")
    out["preferencia"] = {"bruta": dict(pref),
                          "ponderada": {k: v / tot_w
                                        for k, v in pref_pond.items()}}

    # ------------------------------------------------------- 4. Wilcoxon
    md.append("\n## 4. Wilcoxon pareado (TAVERN × Longest)\n")
    md.append("| Critério | n pares | W | p (bicaudal) | Mediana Δ |")
    md.append("|---|---|---|---|---|")
    testes, tex_testes = {}, []
    for crit in CRITERIOS:
        a, b = [], []
        for c in itens:
            na = [nota(R, av, c, "tavern", crit, K) for av in avs]
            nb = [nota(R, av, c, "longest", crit, K) for av in avs]
            na = [x for x in na if x is not None]
            nb = [x for x in nb if x is not None]
            if na and nb:
                a.append(float(np.mean(na)))
                b.append(float(np.mean(nb)))
        if len(a) < 5 or all(x == y for x, y in zip(a, b)):
            md.append(f"| {crit} | {len(a)} | n/d | n/d | n/d |")
            testes[crit] = None
            continue
        st = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        dif = float(np.median(np.array(a) - np.array(b)))
        testes[crit] = {"n": len(a), "W": float(st.statistic),
                        "p": float(st.pvalue), "mediana_delta": dif}
        md.append(f"| {crit} | {len(a)} | {st.statistic:.1f} | "
                  f"{st.pvalue:.4f} | {dif:+.2f} |")
        tex_testes.append(f"{crit} & {len(a)} & {st.statistic:.1f} & "
                          f"{st.pvalue:.4f} & {dif:+.2f} \\\\")
    md.append(f"\nCom n = {len(itens)} pares, o menor p bicaudal alcançável "
              f"pelo teste é da ordem de 0,002; a potência é baixa por "
              f"desenho, e um p não significativo aqui **não** é evidência de "
              f"equivalência. O tamanho da amostra foi escolhido para caber "
              f"em 1h30 de especialista, não para detectar um efeito pequeno.\n")
    out["wilcoxon"] = testes

    # ------------------------------------------------- 5. viés de posição
    md.append("\n## 5. Controle de viés de posição\n")
    bias = {}
    for slot in ("X", "Y"):
        vals = []
        for c in itens:
            if K["itens"][c][slot] != "tavern":
                continue
            for av in avs:
                for crit in CRITERIOS:
                    v = nota(R, av, c, "tavern", crit, K)
                    if v is not None:
                        vals.append(v)
        bias[slot] = float(np.mean(vals)) if vals else None
    fx, fy = bias["X"], bias["Y"]
    md.append(f"Média das notas do TAVERN quando foi apresentado como "
              f"Sistema X: **{'n/d' if fx is None else f'{fx:.2f}'}**; "
              f"como Sistema Y: **{'n/d' if fy is None else f'{fy:.2f}'}**.")
    if fx is not None and fy is not None:
        md.append(f"Diferença {fx - fy:+.2f}. É isso que o "
                  f"contrabalanceamento 5/5 permite estimar; uma diferença "
                  f"grande aqui contamina todas as comparações acima.\n")
    out["vies_posicao"] = bias

    # ---------------------------------------------- 6. por estrato
    md.append("\n## 6. Por estrato (número de fontes)\n")
    md.append("| Fontes | Itens | TAVERN (média simples) | Longest |")
    md.append("|---|---|---|---|")
    for h in sorted({estrato[c] for c in itens}):
        sub = [c for c in itens if estrato[c] == h]
        linha = []
        for sis in SISTEMAS:
            vals = [nota(R, av, c, sis, crit, K)
                    for c in sub for av in avs for crit in CRITERIOS]
            vals = [v for v in vals if v is not None]
            linha.append(f"{np.mean(vals):.2f}" if vals else "n/d")
        md.append(f"| {h} | {len(sub)} | {linha[0]} | {linha[1]} |")

    # ---------------------------------------------------------- saída
    (args.out / "resultados.md").write_text("\n".join(md), encoding="utf-8")
    (args.out / "resultados.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    def tabela(nome, cab, corpo, legenda, rotulo):
        (args.out / nome).write_text(
            "\\begin{table}[htbp]\n\\centering\n\\small\n"
            f"\\begin{{tabular}}{{{'l' * len(cab)}}}\n\\hline\n"
            + " & ".join(f"\\textbf{{{c}}}" for c in cab) + " \\\\\n\\hline\n"
            + "\n".join(corpo) + "\n\\hline\n\\end{tabular}\n"
            f"\\caption{{{legenda}}}\n\\label{{{rotulo}}}\n\\end{{table}}\n",
            encoding="utf-8")

    tabela("tab_medias.tex", ["Critério", "TAVERN", "Longest", "$\\Delta$"],
           tex_medias,
           "Médias ponderadas ($w_h = N_h/n_h$) das notas dos especialistas, "
           "escala forçada de 4 pontos.", "tab:aval-medias")
    tabela("tab_concordancia.tex",
           ["Critério", "Sistema", "$\\alpha$ ordinal", "$\\kappa$ Fleiss"],
           linhas_tex, "Concordância entre os três especialistas.",
           "tab:aval-concordancia")
    tabela("tab_testes.tex",
           ["Critério", "$n$", "$W$", "$p$", "Mediana $\\Delta$"],
           tex_testes or ["\\multicolumn{5}{c}{sem pares válidos} \\\\"],
           "Wilcoxon pareado entre TAVERN e a baseline extrativa.",
           "tab:aval-testes")

    print("\n".join(md))
    print(f"\n-> {args.out}/resultados.md, resultados.json, "
          f"tab_medias.tex, tab_concordancia.tex, tab_testes.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
