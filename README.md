# NheengatuMT

Tradução automática português ↔ **nheengatu** (`yrl`), a Língua Geral Amazônica —
cooficial em São Gabriel da Cachoeira (AM) e ausente de todos os benchmarks
públicos de MT.

O artigo que descreve este trabalho foi **aceito no WMT 2026** (Conference on
Machine Translation):

> Fabrício Carraro, Rodolfo Joel Zevallos, Rodrigo Gonçalves de Souza,
> Caio Henrique Faustino da Silva e John E. Ortega. *The Constitution Speaks
> Nheengatu: An Open MT System and Reproducible Corpus Pipeline for the
> Amazonian Língua Geral.* WMT 2026.

## O que há aqui

Um sistema aberto de MT e o **primeiro benchmark público** para a língua. O
nheengatu não aparece no NLLB-200, no FLORES+, no MADLAD, no Google Translate
nem em nenhuma das cinco edições do AmericasNLP; até aqui, os únicos resultados
de MT publicados vinham de pesquisa industrial fechada.

### `release/benchmark/` — 682 segmentos, redistribuíveis

| conjunto | segmentos | fonte | por que importa |
|---|---|---|---|
| `dev_const` / `test_const` | 233 / 218 | Constituição Federal em nheengatu (STF/CNJ, 2023) | **artigos inteiros retidos** — nenhum artigo aparece no treino e na avaliação |
| `dev_fala` / `test_fala` | 116 / 115 | fala espontânea transcrita (Reich / FU Berlin) | falantes e sessões **disjuntos**; `test_fala` é de **uso único** |

O conjunto jurídico mede geração de texto novo e difícil; o de fala mede um
registro que projetos em línguas majoritariamente faladas quase nunca avaliam.
São os dois eixos sobre os quais o artigo faz suas afirmações centrais.

### `release/train_aberto/` — 8.880 pares

A fatia do corpus de treino que pode ser redistribuída hoje: Constituição,
treebank UD, fala transcrita, **o dicionário de Marcel T. Avila** (2.745 pares,
publicados com autorização escrita do autor e crédito obrigatório), Tycho Brahe
e catecismo. Os outros **4.145 pares dependem de permissões ainda pendentes** —
para eles o repositório entrega os *scripts de extração*, não o texto. Quem
tiver acesso às fontes reconstrói o corpus inteiro rodando `scripts/`.

Cada camada, com base legal e método de alinhamento, está em
**[`docs/PROVENIENCIA.md`](docs/PROVENIENCIA.md)**.

### Código

| onde | o quê |
|---|---|
| `scripts/` | extração das fontes, alinhamento, construção do corpus, auditorias |
| `cluster/scripts/` | treino (`train_nllb.py`), avaliação (`evaluate_yrl.py`), anotação de terminologia, decodificação restrita, **`term_success.py`** (o scorer de TSR) |
| `resultados/cluster_final/` | hipóteses, métricas e logs dos 43 runs — **todo número do artigo se recomputa daqui, sem precisar dos modelos** |

O texto do artigo não está aqui; sairá pelo canal de publicação. E a avaliação por
falantes só entra quando estiver concluída — publicar o instrumento enquanto ele
está em campo, junto da chave de cegamento, o inutilizaria.


### Modelos

Não ficam aqui — são 32 GB e o GitHub recusa arquivos acima de 100 MB. Vão para
o Hugging Face sob **CC-BY-NC-4.0**, herdada do NLLB-200. Um deles, derivado do
Qwen3.5 (Apache-2.0), é o único sem essa restrição.

## Alinhamento por endereço jurídico

A Constituição foi alinhada por **identidade de estrutura documental**, não por
comprimento nem por similaridade: as duas edições carregam o mesmo esqueleto
legal (artigo, parágrafo, inciso, alínea), então uma máquina de estados valida a
sequência de identificadores enquanto percorre o documento e pareia unidades cujo
endereço completo coincide.

O efeito colateral é o que torna o corpus auditável: **cada par carrega o endereço
jurídico que o produziu**, e qualquer erro é localizável em vez de estatístico.
Foi assim que apareceram erratas do livro impresso — incisos rotulados errado,
renumerações em cascata e um bloco de emenda constitucional ausente da tradução.

## Ética

O projeto segue os princípios CARE. A Academia da Língua Nheengatu foi contatada
e **não respondeu**; silêncio não é tratado como consentimento, e o glossário e as
escolhas ortográficas são **provisórios e derivados das fontes**, não autoritativos.
Qual ortografia o sistema deve produzir é decisão da comunidade, não técnica.

A avaliação por falantes está planejada e **nenhum dado de falante foi analisado**
até aqui.

## Licença

Código sob a licença em `LICENSE`. **Os dados não são de licença única** — cada
camada mantém a sua, e `docs/PROVENIENCIA.md` diz qual é qual. Redistribuir o
conteúdo de `release/` exige respeitar BY-NC-SA nas camadas que a carregam e
creditar o STF/CNJ e os tradutores indígenas na camada da Constituição.
