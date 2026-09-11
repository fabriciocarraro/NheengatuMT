# Corpus paralelo PT ↔ Nheengatu (yrl) — Constituição Federal do Brasil

**Gerado em:** 01/08/2026

**Fontes:**
- **PT:** `Constituicao-PT-BR-EC128-2022.pdf` — Senado Federal, *texto constitucional compilado até a EC nº 128/2022* ([BDSF id/603955](https://www2.senado.leg.br/bdsf/handle/id/603955), obtido via Wayback Machine). **Esta é a base contemporânea da tradução** — a troca da edição de 2016 por esta eliminou 84 das 87 divergências de conteúdo e deu contraparte PT às 207 unidades de ECs pós-2016 que antes ficavam sem par.
- **YRL:** `Constituicao-Nheengatu.pdf` — *"Mundu sa Turusu" waá...*, tradução oficial ao Nheengatu lançada pelo STF/CNJ em jul/2023 (diagramada sobre o mesmo template editorial do Senado).
- A edição PT de 2016 (`Constituicao-PT-BR.pdf`) permanece no projeto como fonte histórica; o pipeline roda com qualquer das duas (`align.py pt` | `align.py pt2022`).

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `constituicao_pt_yrl.tsv` | **2.496 pares limpos** (`id`, `tipo`, `pt`, `yrl`, `flags`) — caput 267, §§ 713, incisos 1.155, alíneas 263, cabeçalhos 97, fecho 1. ~358 mil caracteres PT / ~323 mil YRL. |
| `divergencias.tsv` | 3 pares divergentes + 54 unidades sem par, classificados e anotados. |
| `constituicao_pt_yrl_full.jsonl` | Tudo, com metadados: `yrl_raw` (como impresso) e `yrl` (normalizado), páginas de origem, veredito da verificação no Planalto, notas, remapeamentos. |

Scripts reproduzíveis em `scripts/` (`extract_lines.py`/`extract_pt2022.py` → `parse_units.py` → `align.py` → `verify_planalto.py` → `build_corpus.py`); intermediários e logs em `work/`.

## Garantias de correspondência (método)

1. **Extração** (PyMuPDF): remoção de cabeçalhos correntes, rodapés e notas editoriais ("NE:"); junção de hifenização por *soft hyphen* (PT) e por hífen lexical conferido caso a caso (NG, 59); fusão de fragmentos da mesma linha visual separados por expoentes de nota; normalização do hífen não separável (U+2011) da edição 2022.
2. **Parse com máquina de estados validada por sequência**: um marcador só é aceito se couber na sequência (Art. n+1, § k+1, inciso romano seguinte, alínea seguinte; sufixos -A/-B/-C em artigos, §§ e incisos). Referências no meio do texto ("...§ 4º, 150, II...") são rejeitadas; romanos malformados do livro (`VIX`→IX, `XLVIX`→XLIX) só são aceitos quando não são romanos válidos. Inventários de cabeçalhos idênticos nos dois lados: 9 títulos, 33 capítulos, 50 seções, 5 subseções.
3. **Alinhamento por identidade estrutural** (`art5.caput.inc79` ↔ `art5.caput.inc79`), com pareamento **por conteúdo** (flag `pareado_por_conteudo`, 5 casos) onde o livro NG renumerou após omitir uma unidade.
4. **Validações por par**: dígitos citados, razão de comprimento (outliers por MAD), texto vazio.
5. **Verificação externa (Planalto, `constituicao.htm` com redações históricas)**: 2.401 unidades verificadas (2.398 pareadas + os 3 pares divergentes; contagens brutas em `work/planalto_check.jsonl`): 2.207 idênticas ao texto vigente em 2026, 112 idênticas a menos de pontuação, 30 mudaram apenas por ECs de 2023+ (posteriores à base — ex.: reforma tributária EC 132/2023), 21 anomalias de busca/variação editorial Senado×Planalto resolvidas por revisão manual e 31 curtas demais para busca confiável (baixo risco). **Nenhuma unidade PT desta base foi substituída por EC ≤ 2022** — confirmação empírica (pós-revisão manual das anomalias) de que a edição EC 128/2022 é a base da tradução.

## Divergências restantes (3 pares, fora do TSV limpo)

| id | Motivo |
|---|---|
| `art39.caput` | ADI 2.135: a edição imprime a redação da EC 19/98 (aplicação suspensa); o livro NG traduziu **as duas** redações (a 2ª preservada em `art39.caput.bis`). Requer decisão humana. |
| `art73.par1.inc1` | NG traz "60 akayu" onde a redação (EC 122/2022) diz "menos de setenta" — não corresponde a nenhuma redação. |
| `art101.caput` | NG traz "65 akayú" (redação **pré**-EC 122/2022); a base traz "menos de setenta". O livro usou snapshots diferentes: o art. 111-A já traz "70" (pós-EC 122). |

## Unidades sem par (54)

- **31 `pt_revogado_ng_omite`**: dispositivos que a edição PT imprime como "(Revogado)" e o livro NG omite (art. 192 quase inteiro, EC 40/2003; art. 166 §14 I-IV; etc.).
- **22 `ng_omite_livro`** — omissões reais do livro NG, confirmadas por leitura: art. 4º X (asilo político), art. 21 XXIII d, art. 22 XI, art. 68 caput (os §§ foram impressos sob o art. 67 e remapeados), art. 72 §§ 1º-2º, art. 84 XXVII, art. 85 PU, art. 92 § 1º, art. 102 I i, art. 139 I, art. 201 § 5º, **art. 163 VIII + alíneas a-e + PU e art. 164-A caput/PU (bloco da EC 109/2021 pulado pelos tradutores)**, art. 100 § 21 IV (conteúdo fundido no fim do III).
- **1 `ng_extra_livro`**: `art39.caput.bis`.

## Erros do próprio livro NG — detectados e tratados

Correções cirúrgicas documentadas (`scripts/parse_units.py`, logs em `work/`): art. 93 VIII-A e art. 109 V-A rotulados "VIII"/"V"; art. 128 §5º II b impressa como "c)"; art. 39 impresso duas vezes; art. 22 XI / art. 92 §1º / art. 139 I-II / art. 201 §5º omitidos com renumeração em cascata (repareados por conteúdo); fragmento em português não traduzido no art. 194; "Parágrafo único" com 10 grafias diferentes, 18 delas coladas no fim da unidade anterior (divididas automaticamente, cada uma logada).

Pares mantidos no TSV com flag **`erro_conteudo_livro`** (6): art. 41 §2º (referência espúria ao art. 142), art. 130-A §2º II ("art. 39" por 37), art. 167-E ("art. 197" por 167), art. 201 §8 ("55 akayu" resolvendo mal "reduzido em 5 anos"), art. 235 I ("10" por dezessete), art. 100 §21 III (fusão com o IV).

## Normalização do NG e peculiaridades do PT

`yrl` normalizado / `yrl_raw` como impresso. Glifos unificados (contagens da base atual): ῖ→ĩ (483), ū→ũ (250), ē→ẽ (223), ī→ĩ (83), ´ solto removido (27), À→Á, Ì→Í, ữ→ũ, Â→Ã e unitários. O **ñ** de "diñeru" (153×) e as oscilações internas (Kãmaára/Kâmaára, ta/tá) foram **mantidos** — são características da publicação. No PT, typos da edição do Senado preservados e anotados (ex.: "8.000.0001" no art. 29-A VI), além de 21 variações de pontuação Senado×Planalto (classe `fonte_editorial`).

## Flags no TSV (atenção, não erro)

- `razao_comprimento_atipica` (159): a tradução condensa unidades longas; pares válidos, porém mais "livres".
- `digitos_divergem` (19): quase todos benignos (PT por extenso × NG em algarismos); os problemáticos têm também `erro_conteudo_livro`.
- `pareado_por_conteudo` (5): renumerações do livro corrigidas.
- `erro_conteudo_livro` (6): ver acima.

## Recomendações de uso

1. Para treino de MT: usar o TSV inteiro; opcionalmente separar os 159 `razao_comprimento_atipica` para um estágio de menor peso e excluir os 6 `erro_conteudo_livro` do conjunto de avaliação.
2. Segmentação em frases: viável onde as contagens de sentenças casarem 1:1 (próximo passo natural).
3. O par do art. 39 e os dois casos da EC 122 estão em `divergencias.tsv` para decisão humana.
4. Ortografia do Nheengatu: `yrl` uniformiza apenas glifos inequívocos; normalização linguística mais profunda é decisão de projeto (manter `yrl_raw` ao lado).
