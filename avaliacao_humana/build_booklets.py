#!/usr/bin/env python3
"""
Compila os cadernos de avaliação humana a partir da amostra congelada.

    python avaliacao_humana/amostragem.py          # congela a amostra
    python avaliacao_humana/build_booklets.py      # gera os cadernos

Saída em `avaliacao_humana/saida/`:

    Caderno_E01.pdf / .tex / .html      26 páginas, A4, frente e verso
    Caderno_E02.pdf / .tex / .html
    Caderno_E03.pdf / .tex / .html
    Guia_do_pesquisador.pdf / .tex      CONFIDENCIAL: chave do cegamento

O PDF sai por `pdflatex` quando há um motor LaTeX instalado. Quando não há,
o `.html` é o caminho completo: layout A4 paginado em CSS, mesma estrutura de
26 páginas, e o PDF sai pelo "Imprimir -> Salvar como PDF" do navegador com
margens em Padrão e cabeçalhos/rodapés desligados. Ver
LEIA_ANTES_DE_IMPRIMIR.md.

Estrutura de 26 páginas (13 folhas físicas em frente e verso):

     1        rosto, instruções, registro de tempo
     2-3      item piloto PA01 (fontes | sistemas + gabarito comentado)
     4        transição e lembrete das regras
     5-24     10 itens, 2 páginas cada (ímpar: fontes | par: sistemas + notas)
    25        três perguntas qualitativas finais
    26        folha pautada para anotações

Os três cadernos trazem os mesmos itens, na mesma ordem, com a mesma
atribuição X/Y. Isso é deliberado: a concordância entre avaliadores
(Krippendorff, Fleiss) mede discordância de julgamento, e permutar a
apresentação por avaliador acrescentaria uma fonte de variação que não se
consegue separar depois com n = 10. O contrabalanceamento de posição está nos
itens — TAVERN é o Sistema X em 5 deles e o Sistema Y nos outros 5 —, o que
permite estimar o viés de posição comparando os dois grupos.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

AVALIADORES = ("E01", "E02", "E03")

CRITERIOS = [
    ("A1", "Completude de detalhes",
     "O texto incluiu os detalhes específicos relatados por cada uma das fontes?",
     ["Insuficiente — omitiu detalhes importantes de alguma fonte",
      "Parcial — manteve só o núcleo, descartando particularidades relevantes",
      "Substancial — cobriu a quase totalidade dos detalhes de todas as fontes",
      "Completa — preservou todos os detalhes informativos das fontes"]),
    ("A2", "Fidelidade factual (não-alucinação)",
     "O texto é estritamente fiel aos relatos, sem inventar fatos, diálogos, "
     "nomes ou conexões causais inexistentes?",
     ["Invenção severa — acrescentou fato, nome ou fala que nenhuma fonte sustenta",
      "Distorções pontuais — pequenas imprecisões ou inferências causais não explícitas",
      "Quase perfeita — fiel aos textos, com liberdade apenas conectiva",
      "Totalmente fiel — restrito ao que as fontes relatam"]),
    ("A3", "Fluência, coesão e legibilidade",
     "O texto flui como narrativa natural, legível, em prosa contínua?",
     ["Ruim — fragmentado, truncado, com costuras artificiais evidentes",
      "Aceitável — compreensível, mas com transições ásperas",
      "Boa — leitura coesa, raras frases mecânicas",
      "Excelente — prosa fluida e natural"]),
    ("A4", "Eliminação de redundância",
     "Quando duas ou mais fontes narram o mesmo fato ou diálogo com palavras "
     "diferentes, o texto fundiu as informações em vez de repetir o mesmo fato?",
     ["Excessiva — repete frases, perguntas ou ações idênticas das fontes",
      "Moderada — repetições perceptíveis que poderiam ter sido unificadas",
      "Baixa — boa síntese, pouquíssima repetição",
      "Nenhuma — fundiu os relatos sem repetições desnecessárias"]),
]

INSTRUCOES = [
    "Você vai ler, em cada item, as passagens bíblicas originais de um mesmo "
    "episódio, relatadas por dois, três ou quatro Evangelhos. Em seguida verá "
    "dois textos, <b>Sistema X</b> e <b>Sistema Y</b>, que tentaram reunir "
    "aqueles relatos numa narrativa única.",
    "As fontes da página anterior são a <b>única</b> evidência. Um detalhe que "
    "você sabe de outra passagem, mas que não está nas fontes impressas, conta "
    "como invenção em A2.",
    "Os dois sistemas são apresentados em ordem trocada de item para item. "
    "<b>Não há padrão a descobrir</b>, e nem sempre o mesmo sistema é o X.",
    "A escala tem quatro pontos e <b>não tem ponto neutro</b>: escolha o lado. "
    "Marque um valor por critério, mesmo quando a decisão for apertada.",
    "Avalie cada sistema por si, contra as fontes — não um contra o outro. A "
    "comparação direta é só a última pergunta, A5.",
    "Não volte para corrigir itens anteriores depois de virar a página.",
]

QUALITATIVAS = [
    "Uma narrativa consolidada como estas seria útil no seu trabalho "
    "exegético ou de ensino? Em que situação concretamente?",
    "Qual foi o defeito mais frequente que você notou nos textos avaliados?",
    "O que precisaria mudar para que você usasse uma ferramenta destas "
    "com confiança?",
]


# --------------------------------------------------------------------- LaTeX
_TEX = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex(s: str) -> str:
    out = "".join(_TEX.get(c, c) for c in s)
    # aspas curvas do corpus -> aspas tipográficas do LaTeX
    out = (out.replace("\u201c", "``").replace("\u201d", "''")
              .replace("\u2018", "`").replace("\u2019", "'")
              .replace("\u2014", "---").replace("\u2013", "--"))
    return out


def tex_b(s: str) -> str:
    """Converte o <b>...</b> das instruções para \\textbf."""
    return re.sub(r"&lt;b&gt;(.*?)&lt;/b&gt;", r"\\textbf{\1}",
                  tex(s.replace("<b>", "&lt;b&gt;")
                       .replace("</b>", "&lt;/b&gt;")))


def _tex_escala(codigo: str) -> str:
    return (r"\textbf{%s} \hfill $\square$\,1 \quad $\square$\,2 \quad "
            r"$\square$\,3 \quad $\square$\,4" % codigo)


def bloco_notas_tex(prefixo: str) -> str:
    linhas = [r"\begin{center}\rule{\textwidth}{0.4pt}\end{center}",
              r"\vspace{-2mm}"]
    for s in ("X", "Y"):
        linhas.append(r"\textbf{\large Notas — Sistema %s}\\[1mm]" % s)
        linhas.append(r"\begin{tabular}{@{}p{0.47\textwidth}@{}p{0.47\textwidth}@{}}")
        cel = [_tex_escala(f"{c[0]}") for c in CRITERIOS]
        linhas.append(f"{cel[0]} & {cel[1]} \\\\[2mm]")
        linhas.append(f"{cel[2]} & {cel[3]} \\\\[3mm]")
        linhas.append(r"\end{tabular}")
        linhas.append(r"\vspace{2mm}")
    linhas += [
        r"\textbf{\large A5 — Preferência global}\\[1mm]",
        r"$\square$ Prefiro o Sistema X \quad $\square$ Prefiro o Sistema Y "
        r"\quad $\square$ Indiferente / equivalentes \\[2mm]",
        r"Justificativa sucinta (opcional):\\[1mm]",
        r"\rule{\textwidth}{0.4pt}\\[3mm]",
        r"\rule{\textwidth}{0.4pt}",
    ]
    return "\n".join(linhas)


def pagina_fontes_tex(it: dict) -> str:
    L = [r"\section*{%s \quad \normalsize\mdseries Fontes bíblicas originais}"
         % tex(it["codigo"]),
         r"\vspace{-3mm}",
         r"\noindent\emph{%d relatos do mesmo episódio. São a única evidência "
         r"para o julgamento da página seguinte.}\\[3mm]" % it["n_fontes"]]
    for f in it["fontes"]:
        L.append(r"\noindent\textbf{%s}\\[1mm]" % tex(f["referencia"]))
        L.append(r"\begin{quote}\setlength{\parindent}{0pt}%s\end{quote}"
                 % tex(f["texto"]))
    return "\n".join(L)


def pagina_sistemas_tex(it: dict, gabarito: str = "") -> str:
    L = [r"\section*{%s \quad \normalsize\mdseries Textos consolidados}"
         % tex(it["codigo"]), r"\vspace{-3mm}"]
    for s in ("X", "Y"):
        L.append(r"\noindent\textbf{\large Sistema %s}\\[1mm]" % s)
        L.append(r"\begin{quote}\setlength{\parindent}{0pt}%s\end{quote}"
                 % tex(it["sistemas"][s]))
    if gabarito:
        L.append(r"\vspace{2mm}\noindent\fbox{\parbox{0.97\textwidth}{"
                 r"\small\textbf{Gabarito comentado (apenas neste item de "
                 r"nivelamento):} %s}}\\[2mm]" % tex(gabarito))
    L.append(bloco_notas_tex(it["codigo"]))
    return "\n".join(L)


def gabarito_piloto(it: dict) -> str:
    return (
        "Compare cada texto com as duas fontes acima. Um deles reproduz um "
        "único relato na íntegra e ignora o outro: em A1 isso é Insuficiente "
        "ou Parcial, mas em A2 costuma ser Totalmente fiel, porque não "
        "inventou nada — os critérios são independentes e frequentemente "
        "divergem. O outro reúne os dois relatos: verifique se todo detalhe "
        "das duas fontes sobreviveu (A1), se nada foi acrescentado (A2), se a "
        "emenda entre eles lê bem (A3) e se o que as duas fontes dizem em "
        "comum aparece uma vez só (A4). Não há resposta certa registrada aqui: "
        "o item existe para calibrar o uso da escala, e suas notas nele não "
        "entram em nenhuma estatística.")


def caderno_tex(av: str, amostra: dict) -> str:
    P: List[str] = []

    # 1 -------------------------------------------------------- rosto
    P.append("\n".join([
        r"\thispagestyle{empty}",
        r"\begin{center}",
        r"{\Large\bfseries Avaliação por especialistas}\\[2mm]",
        r"{\large Consolidação de narrativas nos Evangelhos}\\[6mm]",
        r"{\large Caderno \textbf{%s}}\\[2mm]" % av,
        r"\end{center}",
        r"\noindent Nome: \rule{0.72\textwidth}{0.4pt}\\[3mm]",
        r"\noindent Data: \rule{0.3\textwidth}{0.4pt} \hfill "
        r"Início: \rule{0.15\textwidth}{0.4pt} \hfill "
        r"Término: \rule{0.15\textwidth}{0.4pt}\\[2mm]",
        r"\noindent Pausas (início/fim): \rule{0.62\textwidth}{0.4pt}\\[5mm]",
        r"\noindent\textbf{\large Como preencher}\\[1mm]",
        r"\begin{enumerate}\setlength{\itemsep}{1.2mm}",
        *[r"\item %s" % tex_b(i) for i in INSTRUCOES],
        r"\end{enumerate}",
        r"\vspace{2mm}\noindent\textbf{\large A escala}\\[1mm]",
        r"\small",
        *[(r"\noindent\textbf{%s — %s.} \emph{%s}\\[0.5mm]"
           r"\textbf{1} %s \textbf{2} %s \textbf{3} %s \textbf{4} %s\\[2mm]"
           % (tex(c[0]), tex(c[1]), tex(c[2]), *[tex(x) for x in c[3]]))
          for c in CRITERIOS],
        r"\noindent\textbf{A5 — Preferência global.} Qual dos dois você "
        r"entregaria a um leitor? Há campo para justificar.",
        r"\normalsize",
        r"\vfill",
        r"\noindent\footnotesize Tempo previsto: 1h15 a 1h30. Pare quando "
        r"precisar e registre a pausa acima.",
    ]))

    # 2-3 ------------------------------------------------------ piloto
    pil = amostra["piloto"]
    P.append(r"\section*{Item de nivelamento}" + "\n"
             + r"\noindent\emph{Este item não entra em nenhuma estatística. "
               r"Serve para você calibrar o uso da escala, e traz um gabarito "
               r"comentado na página seguinte.}\\[4mm]" + "\n"
             + pagina_fontes_tex(pil))
    P.append(pagina_sistemas_tex(pil, gabarito_piloto(pil)))

    # 4 ------------------------------------------------- transição
    P.append("\n".join([
        r"\section*{A partir daqui vale}",
        r"\noindent Os dez itens seguintes são a coleta oficial. Antes de "
        r"começar, três lembretes:\\[3mm]",
        r"\begin{itemize}\setlength{\itemsep}{2.5mm}",
        r"\item \textbf{As fontes impressas são a única evidência.} O que "
        r"você sabe de outras passagens não conta a favor nem contra.",
        r"\item \textbf{Os critérios são independentes.} Um texto pode ser "
        r"completo e infiel, ou fiel e truncado. Não deixe uma nota puxar "
        r"a outra.",
        r"\item \textbf{A escala não tem meio-termo.} Entre 2 e 3, escolha.",
        r"\end{itemize}",
        r"\vspace{6mm}",
        r"\noindent Cada item ocupa duas páginas: as fontes numa, os dois "
        r"textos e as notas na outra. Você pode voltar à página das fontes "
        r"quantas vezes quiser enquanto responde.",
        r"\vfill",
        r"\noindent\footnotesize Se em algum item os dois textos lhe "
        r"parecerem idênticos, marque as mesmas notas e assinale "
        r"``Indiferente'' em A5 --- não é pegadinha, e acontece.",
    ]))

    # 5-24 ------------------------------------------------- 10 itens
    for it in amostra["itens"]:
        P.append(pagina_fontes_tex(it))
        P.append(pagina_sistemas_tex(it))

    # 25 -------------------------------------------- qualitativas
    P.append("\n".join([
        r"\section*{Três perguntas finais}",
        r"\noindent\emph{Sobre o conjunto, não sobre um item.}\\[4mm]",
        *sum([[r"\noindent\textbf{Q%d.} %s\\[2mm]" % (i, tex(q)),
               r"\rule{\textwidth}{0.4pt}\\[4mm]",
               r"\rule{\textwidth}{0.4pt}\\[4mm]",
               r"\rule{\textwidth}{0.4pt}\\[7mm]"]
              for i, q in enumerate(QUALITATIVAS, 1)], []),
        r"\vfill",
        r"\noindent Obrigado pelo tempo e pelo cuidado. Seu julgamento é a "
        r"única evidência que esta parte do trabalho tem: as métricas "
        r"automáticas comparam o texto com uma referência que é, ela "
        r"própria, uma seleção entre as fontes.",
    ]))

    # 26 -------------------------------------------------- pautada
    P.append(r"\section*{Anotações}" + "\n"
             + "\n".join([r"\rule{\textwidth}{0.4pt}\\[6.2mm]"] * 28))

    corpo = "\n\\clearpage\n".join(P)
    return r"""\documentclass[11pt,a4paper,twoside]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[portuguese]{babel}
\usepackage[top=18mm,bottom=18mm,left=18mm,right=18mm]{geometry}
\usepackage{amssymb}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\pagestyle{plain}
\begin{document}
%s
\end{document}
""" % corpo


# ---------------------------------------------------------------------- HTML
CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; }
body { margin:0; font-family: Georgia,'Times New Roman',serif; font-size:10.5pt;
       line-height:1.42; color:#111; background:#e9e9e9; }
.pagina { width:210mm; min-height:297mm; padding:18mm; background:#fff;
          margin:0 auto 6mm; position:relative; break-after:page;
          page-break-after:always; }
.pagina:last-child { break-after:auto; page-break-after:auto; }
.folio { position:absolute; bottom:9mm; left:0; right:0; text-align:center;
         font-size:8.5pt; color:#666; }
h1 { font-size:17pt; margin:0 0 2mm; text-align:center; }
h2 { font-size:12.5pt; margin:0 0 4mm; border-bottom:1px solid #999;
     padding-bottom:1.5mm; }
h2 small { font-weight:normal; color:#555; }
h3 { font-size:11pt; margin:4mm 0 1.5mm; }
.sub { text-align:center; font-size:12pt; margin:0 0 8mm; color:#333; }
.linha { border-bottom:1px solid #444; display:inline-block; }
.passagem { margin:0 0 3.5mm; padding-left:5mm; border-left:2px solid #ccc; }
.passagem b { display:block; margin-bottom:1mm; }
.sistema { margin:0 0 4mm; padding:3mm 4mm; border:1px solid #bbb; }
.sistema b { display:block; margin-bottom:1.5mm; font-size:11pt; }
.gabarito { border:1px solid #888; background:#f6f6f6; padding:3mm;
            font-size:9.5pt; margin:3mm 0; }
.notas { border-top:2px solid #444; margin-top:4mm; padding-top:2.5mm; }
.grade { display:grid; grid-template-columns:1fr 1fr; gap:1.5mm 6mm;
         margin:1.5mm 0 3mm; }
.crit { font-size:10pt; }
.cx { letter-spacing:1.5px; }
.escala { font-size:9pt; line-height:1.34; }
.escala li { margin-bottom:1.1mm; }
.capa ol li { margin-bottom:1.0mm; }
.capa h2 { margin:3mm 0 2mm; }
.pauta { border-bottom:1px solid #999; height:6.4mm; }
.rodape { position:absolute; bottom:15mm; left:18mm; right:18mm;
          font-size:9pt; color:#444; }
ol,ul { margin:0 0 3mm; padding-left:5mm; }
li { margin-bottom:1.3mm; }
@media print { body { background:#fff; } .pagina { margin:0; box-shadow:none; } }
"""

CX = '<span class="cx">&#9744;&nbsp;1&nbsp;&nbsp;&#9744;&nbsp;2&nbsp;&nbsp;' \
     '&#9744;&nbsp;3&nbsp;&nbsp;&#9744;&nbsp;4</span>'


def h(s: str) -> str:
    return html.escape(s, quote=False)


def notas_html(it: dict) -> str:
    L = ['<div class="notas">']
    for s in ("X", "Y"):
        L.append(f"<h3>Notas &mdash; Sistema {s}</h3><div class='grade'>")
        for c in CRITERIOS:
            L.append(f"<div class='crit'><b>{c[0]}</b> {CX}</div>")
        L.append("</div>")
    L.append("<h3>A5 &mdash; Preferência global</h3>"
             "<div class='crit'>&#9744; Prefiro o Sistema X &nbsp;&nbsp; "
             "&#9744; Prefiro o Sistema Y &nbsp;&nbsp; "
             "&#9744; Indiferente / equivalentes</div>"
             "<div style='margin-top:2mm;font-size:9.5pt'>"
             "Justificativa sucinta (opcional):</div>"
             "<div class='pauta'></div><div class='pauta'></div></div>")
    return "\n".join(L)


def caderno_html(av: str, amostra: dict) -> str:
    P: List[str] = []

    esc = "".join(
        f"<li><b>{h(c[0])} &mdash; {h(c[1])}.</b> <i>{h(c[2])}</i><br>"
        f"<b>1</b> {h(c[3][0])} &nbsp; <b>2</b> {h(c[3][1])} &nbsp; "
        f"<b>3</b> {h(c[3][2])} &nbsp; <b>4</b> {h(c[3][3])}</li>"
        for c in CRITERIOS)
    P.append(
        f"<h1>Avaliação por especialistas</h1>"
        f"<div class='sub'>Consolidação de narrativas nos Evangelhos<br>"
        f"Caderno <b>{av}</b></div>"
        f"<p>Nome: <span class='linha' style='width:70%'>&nbsp;</span></p>"
        f"<p>Data: <span class='linha' style='width:28%'>&nbsp;</span> "
        f"&nbsp; Início: <span class='linha' style='width:16%'>&nbsp;</span> "
        f"&nbsp; Término: <span class='linha' style='width:16%'>&nbsp;</span></p>"
        f"<p>Pausas (início/fim): "
        f"<span class='linha' style='width:60%'>&nbsp;</span></p>"
        f"<div class='capa'><h2>Como preencher</h2><ol>"
        + "".join(f"<li>{i}</li>" for i in INSTRUCOES) +
        f"</ol><h2>A escala</h2><ul class='escala'>{esc}</ul>"
        f"<p><b>A5 &mdash; Preferência global.</b> Qual dos dois você "
        f"entregaria a um leitor? Há campo para justificar.</p></div>"
        f"<div class='rodape'>Tempo previsto: 1h15 a 1h30. Pare quando "
        f"precisar e registre a pausa acima.</div>")

    def fontes_html(it: dict, titulo: str = "") -> str:
        L = [f"<h2>{h(it['codigo'])} <small>&mdash; fontes bíblicas "
             f"originais</small></h2>"]
        if titulo:
            L.append(f"<p><i>{titulo}</i></p>")
        L.append(f"<p><i>{it['n_fontes']} relatos do mesmo episódio. São a "
                 f"única evidência para o julgamento da página seguinte.</i></p>")
        for f in it["fontes"]:
            L.append(f"<div class='passagem'><b>{h(f['referencia'])}</b>"
                     f"{h(f['texto'])}</div>")
        return "\n".join(L)

    def sistemas_html(it: dict, gab: str = "") -> str:
        L = [f"<h2>{h(it['codigo'])} <small>&mdash; textos "
             f"consolidados</small></h2>"]
        for s in ("X", "Y"):
            L.append(f"<div class='sistema'><b>Sistema {s}</b>"
                     f"{h(it['sistemas'][s])}</div>")
        if gab:
            L.append(f"<div class='gabarito'><b>Gabarito comentado (apenas "
                     f"neste item de nivelamento):</b> {h(gab)}</div>")
        L.append(notas_html(it))
        return "\n".join(L)

    pil = amostra["piloto"]
    P.append(fontes_html(pil, "Item de nivelamento: não entra em nenhuma "
                              "estatística."))
    P.append(sistemas_html(pil, gabarito_piloto(pil)))
    P.append(
        "<h2>A partir daqui vale</h2>"
        "<p>Os dez itens seguintes são a coleta oficial. Antes de começar, "
        "três lembretes:</p><ul>"
        "<li><b>As fontes impressas são a única evidência.</b> O que você "
        "sabe de outras passagens não conta a favor nem contra.</li>"
        "<li><b>Os critérios são independentes.</b> Um texto pode ser "
        "completo e infiel, ou fiel e truncado. Não deixe uma nota puxar a "
        "outra.</li>"
        "<li><b>A escala não tem meio-termo.</b> Entre 2 e 3, escolha.</li>"
        "</ul><p>Cada item ocupa duas páginas: as fontes numa, os dois textos "
        "e as notas na outra. Você pode voltar à página das fontes quantas "
        "vezes quiser enquanto responde.</p>"
        "<div class='rodape'>Se em algum item os dois textos lhe parecerem "
        "idênticos, marque as mesmas notas e assinale &ldquo;Indiferente&rdquo; "
        "em A5 &mdash; não é pegadinha, e acontece.</div>")

    for it in amostra["itens"]:
        P.append(fontes_html(it))
        P.append(sistemas_html(it))

    P.append("<h2>Três perguntas finais</h2><p><i>Sobre o conjunto, não sobre "
             "um item.</i></p>"
             + "".join(f"<p><b>Q{i}.</b> {h(q)}</p><div class='pauta'></div>"
                       f"<div class='pauta'></div><div class='pauta'></div>"
                       for i, q in enumerate(QUALITATIVAS, 1))
             + "<div class='rodape'>Obrigado pelo tempo e pelo cuidado. Seu "
               "julgamento é a única evidência que esta parte do trabalho "
               "tem: as métricas automáticas comparam o texto com uma "
               "referência que é, ela própria, uma seleção entre as fontes."
               "</div>")
    P.append("<h2>Anotações</h2>" + "<div class='pauta'></div>" * 28)

    corpo = "\n".join(
        f"<div class='pagina'>{p}<div class='folio'>{i}/26</div></div>"
        for i, p in enumerate(P, 1))
    return (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'>"
            f"<title>Caderno {av}</title><style>{CSS}</style></head>"
            f"<body>{corpo}</body></html>")


# ------------------------------------------------------------------- guia
def guia_tex(amostra: dict, chave: dict) -> str:
    linhas = [
        r"\section*{Guia do pesquisador — CONFIDENCIAL}",
        r"\noindent Não imprimir junto com os cadernos, não mostrar aos "
        r"avaliadores.\\[3mm]",
        r"\noindent Artefato: \texttt{%s} (%d eventos, tag \texttt{ancoragem}, "
        r"backbone \texttt{%s}).\\" % (tex(amostra["artefato"]
                                           ["digest_por_evento"][:16]),
                                       amostra["artefato"]["eventos_totais"],
                                       tex(str(amostra["artefato"]["backbone"]))),
        r"Semente: %d. População: %d grupos multi-fonte %s.\\" % (
            amostra["semente"], amostra["populacao"]["multi_fonte"],
            tex(str(amostra["populacao"]["por_estrato"]))),
        r"Pesos $w_h = N_h/n_h$: %s. Soma ponderada: %.2f.\\[4mm]" % (
            tex(str(amostra["pesos"])), amostra["soma_ponderada"]),
        r"\noindent\textbf{Chave do cegamento}\\[2mm]",
        r"\begin{tabular}{@{}llllll@{}}",
        r"\textbf{Item} & \textbf{Sistema X} & \textbf{Sistema Y} & "
        r"\textbf{Estrato} & \textbf{Peso} & \textbf{Fusão}\\\hline",
    ]
    p = chave["piloto"]
    linhas.append(r"PA01 & %s & %s & --- & 0 & (piloto)\\" %
                  (tex(p["X"]), tex(p["Y"])))
    for cod in sorted(chave["itens"]):
        e = chave["itens"][cod]
        linhas.append(r"%s & %s & %s & %d & %.2f & %s\\" % (
            tex(cod), tex(e["X"]), tex(e["Y"]), e["estrato"], e["peso"],
            tex(e["fusion"])))
    linhas += [
        r"\end{tabular}\\[5mm]",
        r"\noindent\textbf{Balanço de posição:} TAVERN é o Sistema X em %d "
        r"itens e o Sistema Y em %d.\\[3mm]" % (
            chave["balanco"]["tavern_como_X"],
            chave["balanco"]["tavern_como_Y"]),
        r"\noindent\textbf{Caminho de fusão na amostra:} %s. Os itens "
        r"marcados \texttt{union} não foram produzidos pelo modelo: o guarda "
        r"de fidelidade rejeitou duas tentativas e o fallback determinístico "
        r"assumiu. Eles pertencem à saída do sistema e por isso estão na "
        r"amostra, mas a análise reporta a estimativa com e sem eles."
        % tex(str(amostra["caminho_de_fusao_na_amostra"])),
    ]
    return r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[portuguese]{babel}
\usepackage[margin=20mm]{geometry}
\usepackage{parskip}
\setlength{\parindent}{0pt}
\begin{document}
%s
\end{document}
""" % "\n".join(linhas)


# ------------------------------------------------------------------- main
def compilar(tex_path: Path) -> bool:
    exe = shutil.which("pdflatex")
    if not exe:
        return False
    for _ in range(2):                       # duas passadas: sumário/refs
        r = subprocess.run(
            [exe, "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(tex_path.parent), str(tex_path)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  pdflatex falhou em {tex_path.name}:")
            print("  " + "\n  ".join(r.stdout.splitlines()[-12:]))
            return False
    for ext in (".aux", ".log", ".out"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amostra", type=Path,
                    default=Path("avaliacao_humana/amostra_congelada.json"))
    ap.add_argument("--chave", type=Path,
                    default=Path("avaliacao_humana/chave_cegamento.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("avaliacao_humana/saida"))
    args = ap.parse_args()

    if not args.amostra.exists():
        raise SystemExit(f"{args.amostra} não existe — rode "
                         f"avaliacao_humana/amostragem.py primeiro")
    amostra = json.loads(args.amostra.read_text(encoding="utf-8"))
    chave = json.loads(args.chave.read_text(encoding="utf-8"))
    if chave["digest_por_evento"] != amostra["artefato"]["digest_por_evento"]:
        raise SystemExit("amostra e chave vêm de artefatos diferentes")

    args.out.mkdir(parents=True, exist_ok=True)
    tem_latex = shutil.which("pdflatex") is not None

    n_paginas = 1 + 2 + 1 + 2 * len(amostra["itens"]) + 2
    print(f"{len(amostra['itens'])} itens + piloto -> {n_paginas} páginas "
          f"({n_paginas / 2:.0f} folhas frente e verso)")
    if n_paginas != 26:
        print(f"  AVISO: o plano de impressão pressupõe 26 páginas")

    for av in AVALIADORES:
        t = args.out / f"Caderno_{av}.tex"
        t.write_text(caderno_tex(av, amostra), encoding="utf-8")
        (args.out / f"Caderno_{av}.html").write_text(
            caderno_html(av, amostra), encoding="utf-8")
        ok = compilar(t) if tem_latex else False
        print(f"  Caderno_{av}: .tex .html" + ("  .pdf" if ok else ""))

    g = args.out / "Guia_do_pesquisador.tex"
    g.write_text(guia_tex(amostra, chave), encoding="utf-8")
    ok = compilar(g) if tem_latex else False
    print(f"  Guia_do_pesquisador: .tex" + ("  .pdf" if ok else "")
          + "   CONFIDENCIAL")

    if not tem_latex:
        print("\nNenhum motor LaTeX encontrado (pdflatex). Os .html são "
              "completos e paginados em A4:\n  abra no navegador, "
              "Imprimir -> Salvar como PDF, margens Padrão, cabeçalhos e "
              "rodapés DESLIGADOS.\n  Ver LEIA_ANTES_DE_IMPRIMIR.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
