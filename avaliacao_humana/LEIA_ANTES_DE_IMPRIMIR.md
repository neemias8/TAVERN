# Antes de imprimir

Três cadernos, um por especialista. Cada um tem **26 páginas** e sai em
**13 folhas de papel A4**.

## Configuração da impressão

| Opção | Valor |
|---|---|
| Papel | A4 |
| Escala | **100%** (nunca "ajustar à página") |
| Frente e verso | **sim, virar na borda longa** |
| Margens | Padrão |
| Cabeçalho e rodapé do navegador | **desligados** |
| Cor | indiferente; o caderno é monocromático |

A escala é o item que mais estraga: a 94% de "ajustar à página" as caixas de
marcação encolhem e a folha ganha uma borda branca que confunde na hora de
grampear. Confira na pré-visualização que o rodapé diz **1/26** e que a
última página é a **26/26** — se aparecer 27, alguma coisa transbordou e o
caderno está errado.

## Se você tem LaTeX instalado

```bash
python avaliacao_humana/amostragem.py
python avaliacao_humana/build_booklets.py
```

Saem `Caderno_E01.pdf`, `Caderno_E02.pdf`, `Caderno_E03.pdf` e
`Guia_do_pesquisador.pdf` em `avaliacao_humana/saida/`. Mande os três
primeiros para a gráfica.

## Se você não tem LaTeX

O mesmo comando gera os `.html`, que são o caderno completo já paginado em
A4. Abra `Caderno_E01.html` no navegador, **Ctrl+P**, destino "Salvar como
PDF", e aplique a tabela acima. Repita para E02 e E03.

No Chrome e no Edge, "cabeçalhos e rodapés" fica em *Mais definições*. Deixe
desmarcado: com ele ligado o navegador imprime a URL do arquivo no rodapé de
todas as páginas, o que além de feio revela ao especialista o caminho onde os
arquivos do sistema estão.

## O que NÃO vai para os especialistas

- `Guia_do_pesquisador.pdf` / `.tex` — traz a chave do cegamento, isto é,
  qual dos dois sistemas é o X e qual é o Y em cada item.
- `chave_cegamento.json` — a mesma informação, em formato de máquina.
- `amostra_congelada.json` — contém os rótulos `tavern` / `longest`.

Se algum destes for impresso junto por engano, a coleta daquele especialista
está perdida e não há como recuperá-la depois: a pessoa não consegue
desver qual sistema era qual.

## Montagem

Grampeie no canto superior esquerdo ou encaderne em espiral. Não separe as
folhas: cada item ocupa uma folha inteira, com as fontes bíblicas na frente e
os dois textos com o formulário no verso, e o especialista precisa poder
virar de um lado para o outro à vontade enquanto decide.

> Uma observação sobre esse layout: como as fontes ficam na frente e os
> textos no verso da **mesma** folha, não há como ver os dois ao mesmo tempo
> sem virar o papel. A alternativa seria deslocar tudo em uma página, para
> que fontes e textos caíssem em páginas opostas de uma abertura — ao custo
> de uma folha a mais e de uma página em branco. Ficou como está porque o
> orçamento de 13 folhas era o requisito; se na aplicação piloto o vaivém
> incomodar, é uma linha para mudar em `build_booklets.py`.

## Depois da coleta

1. Abra `registrar_respostas.html` (duplo clique, funciona sem internet).
2. Digite as notas dos três cadernos e exporte
   `respostas_avaliadores.json`.
3. Rode a análise:

```bash
python avaliacao_humana/analisar_respostas.py --respostas avaliacao_humana/respostas_avaliadores.json
```

Saem `resultados.md` e os fragmentos `.tex` prontos para `\input{}` no
Capítulo 10.
