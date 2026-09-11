# NheengatuMT

**[Português](#português)** · **[English](#english)**

---

## Português

Tradução automática português ↔ **nheengatu** (`yrl`), a Língua Geral Amazônica,
cooficial em São Gabriel da Cachoeira (AM) e ausente de todos os benchmarks
públicos de MT.

O artigo que descreve este trabalho foi **aceito no WMT 2026** (Conference on
Machine Translation):

> Fabrício Carraro, Rodolfo Joel Zevallos, Rodrigo Gonçalves de Souza,
> Caio Henrique Faustino da Silva e John E. Ortega. *The Constitution Speaks
> Nheengatu: An Open MT System and Reproducible Corpus Pipeline for the
> Amazonian Língua Geral.* WMT 2026.

### O que há aqui

Um sistema aberto de MT e o **primeiro benchmark público** para a língua. O
nheengatu não aparece no NLLB-200, no FLORES+, no MADLAD, no Google Translate
nem em nenhuma das cinco edições do AmericasNLP. Até aqui, os únicos resultados
de MT publicados vinham de pesquisa industrial fechada.

#### `release/benchmark/`: 682 segmentos, redistribuíveis

| conjunto | segmentos | fonte | por que importa |
|---|---|---|---|
| `dev_const` / `test_const` | 233 / 218 | Constituição Federal em nheengatu (STF/CNJ, 2023) | **artigos inteiros retidos**. Nenhum artigo aparece no treino e na avaliação |
| `dev_fala` / `test_fala` | 116 / 115 | fala espontânea transcrita (Reich / FU Berlin) | falantes e sessões **disjuntos**; `test_fala` é de **uso único** |

O conjunto jurídico mede geração de texto novo e difícil. O de fala mede um
registro que projetos em línguas majoritariamente faladas quase nunca avaliam.
São os dois eixos sobre os quais o artigo faz suas afirmações centrais.

#### `release/train_aberto/`: 8.880 pares

A fatia do corpus de treino que pode ser redistribuída hoje: Constituição,
treebank UD, fala transcrita, **o dicionário de Marcel T. Avila** (2.745 pares,
publicados com autorização escrita do autor e crédito obrigatório), Tycho Brahe
e catecismo. Os outros **4.145 pares dependem de permissões ainda pendentes**.
Para eles o repositório entrega os *scripts de extração*, não o texto. Quem
tiver acesso às fontes reconstrói o corpus inteiro rodando `scripts/`.

Cada camada, com base legal e método de alinhamento, está em
**[`docs/PROVENIENCIA.md`](docs/PROVENIENCIA.md)**.

#### Código

| onde | o quê |
|---|---|
| `scripts/` | extração das fontes, alinhamento, construção do corpus, auditorias |
| `cluster/scripts/` | treino (`train_nllb.py`), avaliação (`evaluate_yrl.py`), anotação de terminologia, decodificação restrita, **`term_success.py`** (o scorer de TSR) |
| `resultados/cluster_final/` | hipóteses, métricas e logs dos 43 runs. **Todo número do artigo se recomputa daqui, sem precisar dos modelos** |

O texto do artigo não está aqui; sairá pelo canal de publicação. E a avaliação
por falantes só entra quando estiver concluída. Publicar o instrumento enquanto
ele está em campo, junto da chave de cegamento, o inutilizaria.

#### Modelos

Não ficam aqui: são 32 GB e o GitHub recusa arquivos acima de 100 MB. Vão para
o Hugging Face sob **CC-BY-NC-4.0**, herdada do NLLB-200. Um deles, derivado do
Qwen3.5 (Apache-2.0), é o único sem essa restrição.

### Alinhamento por endereço jurídico

A Constituição foi alinhada por **identidade de estrutura documental**, não por
comprimento nem por similaridade: as duas edições carregam o mesmo esqueleto
legal (artigo, parágrafo, inciso, alínea), então uma máquina de estados valida a
sequência de identificadores enquanto percorre o documento e pareia unidades cujo
endereço completo coincide.

O efeito colateral é o que torna o corpus auditável: **cada par carrega o endereço
jurídico que o produziu**, e qualquer erro é localizável em vez de estatístico.
Foi assim que apareceram erratas do livro impresso: incisos rotulados errado,
renumerações em cascata e um bloco de emenda constitucional ausente da tradução.

### Ética

O projeto segue os princípios CARE. A Academia da Língua Nheengatu foi contatada
e **não respondeu**; silêncio não é tratado como consentimento, e o glossário e as
escolhas ortográficas são **provisórios e derivados das fontes**, não autoritativos.
Qual ortografia o sistema deve produzir é decisão da comunidade, não técnica.

A avaliação por falantes está planejada e **nenhum dado de falante foi analisado**
até aqui.

### Licença

Código sob a licença em `LICENSE`. **Os dados não são de licença única**: cada
camada mantém a sua, e `docs/PROVENIENCIA.md` diz qual é qual. Redistribuir o
conteúdo de `release/` exige respeitar BY-NC-SA nas camadas que a carregam e
creditar o STF/CNJ e os tradutores indígenas na camada da Constituição.

---

## English

Machine translation between Portuguese and **Nheengatu** (`yrl`), the Amazonian
Língua Geral, co-official in São Gabriel da Cachoeira (Amazonas, Brazil) and
absent from every public MT benchmark.

The paper describing this work was **accepted at WMT 2026** (Conference on
Machine Translation):

> Fabrício Carraro, Rodolfo Joel Zevallos, Rodrigo Gonçalves de Souza,
> Caio Henrique Faustino da Silva and John E. Ortega. *The Constitution Speaks
> Nheengatu: An Open MT System and Reproducible Corpus Pipeline for the
> Amazonian Língua Geral.* WMT 2026.

### What is here

An open MT system and the **first public benchmark** for the language. Nheengatu
appears in none of NLLB-200, FLORES+, MADLAD or Google Translate, nor in any of
the five editions of AmericasNLP. Until now, the only published MT results came
from closed industrial research.

#### `release/benchmark/`: 682 segments, redistributable

| set | segments | source | why it matters |
|---|---|---|---|
| `dev_const` / `test_const` | 233 / 218 | Federal Constitution in Nheengatu (STF/CNJ, 2023) | **whole articles held out**. No article appears in both training and evaluation |
| `dev_fala` / `test_fala` | 116 / 115 | transcribed spontaneous speech (Reich / FU Berlin) | **disjoint** speakers and sessions; `test_fala` is **single-use** |

The legal set measures generation of new and difficult text. The speech set
measures a register that projects on primarily spoken languages almost never
evaluate. These are the two axes on which the paper makes its central claims.

#### `release/train_aberto/`: 8,880 pairs

The slice of the training corpus that can be redistributed today: the
Constitution, the UD treebank, transcribed speech, **Marcel T. Avila's
dictionary** (2,745 pairs, published with the author's written permission and
mandatory credit), Tycho Brahe and a catechism. The remaining **4,145 pairs
depend on permissions that are still pending**. For those the repository ships
the *extraction scripts*, not the text. Anyone with legitimate access to the
sources can rebuild the full corpus by running `scripts/`.

Every layer, with its legal basis and alignment method, is documented in
**[`docs/PROVENIENCIA.md`](docs/PROVENIENCIA.md)** (in Portuguese).

#### Code

| where | what |
|---|---|
| `scripts/` | source extraction, alignment, corpus construction, audits |
| `cluster/scripts/` | training (`train_nllb.py`), evaluation (`evaluate_yrl.py`), terminology annotation, constrained decoding, **`term_success.py`** (the TSR scorer) |
| `resultados/cluster_final/` | hypotheses, metrics and logs from all 43 runs. **Every number in the paper can be recomputed from here, without the models** |

The text of the paper is not here; it will appear through the publication
channel. The speaker evaluation will only be added once it is complete.
Publishing the instrument while it is in the field, together with its blinding
key, would render it useless.

#### Models

They are not here: 32 GB, and GitHub rejects files above 100 MB. They go to
Hugging Face under **CC-BY-NC-4.0**, inherited from NLLB-200. One of them,
derived from Qwen3.5 (Apache-2.0), is the only one without that restriction.

### Alignment by legal address

The Constitution was aligned by **identity of document structure**, not by length
and not by similarity: both editions carry the same legal skeleton (article,
paragraph, *inciso* and *alínea*, the numbered item and lettered sub-item of
Brazilian legal drafting), so a state machine validates the sequence of
identifiers as it walks the document and pairs units whose full address matches.

The side effect is what makes the corpus auditable: **every pair carries the
legal address that produced it**, and any error is locatable rather than
statistical. That is how errata in the printed book surfaced: mislabelled items,
cascading renumberings and an entire constitutional amendment block missing from
the translation.

### Ethics

The project follows the CARE principles. The Academia da Língua Nheengatu was
contacted and **did not respond**; silence is not treated as consent, and the
glossary and orthographic choices are **provisional and source-derived**, not
authoritative. Which orthography the system should produce is a decision for the
community, not a technical one.

The speaker evaluation is planned and **no speaker data has been analysed** so
far.

### Licence

Code under the licence in `LICENSE`. **The data is not under a single licence**:
each layer keeps its own, and `docs/PROVENIENCIA.md` says which is which.
Redistributing the contents of `release/` requires honouring BY-NC-SA on the
layers that carry it, and crediting the STF/CNJ and the Indigenous translators
on the Constitution layer.
