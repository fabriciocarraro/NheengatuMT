# Dados v2 (raw2/norm2/uni2) — limpeza de artefatos de extração (03/08/2026)

Gerados por `scripts/build_data_v2.py` a partir de `{raw,norm,uni}`; auditoria
completa em `work/relatorio_dados_v2.json`. Motivação: o 1.3B reproduziu
`{o mesmo que:` e `SESEWÁRA` na saída (RESULTADOS_FASE1, Adendo 4) — "o modelo
aprende os erros do corpus" (IBM).

## Regras (espírito do reparo dirigido: só transformação inequívoca; resto é logado)

| Regra | O quê | Alterações (por variante) |
|---|---|---|
| R1 metatexto | Remove `{...}`/truncados do dicionário de Ávila (`{o mesmo que:`, `{v. tb.:`), `[sic]`, `[lit. ...]`; marcadores inequívocos dispensam a trava de tamanho | 225 pares (208 train + 17 dev/test extra) |
| R2 elipse | Remove `[...]`/`[…]` dos dois lados + conserto de pontuação órfã (`", ."`→`"."`, `":."`→`":"`, vírgula final órfã etc.) | 1.298 pares (1.155 train + 143 dev/test extra) |
| R3 CAPS | Rebaixa tokens do lado yrl em caixa alta (217 tipos: SESEWÁRA, ITA, ASUI, KUÁ...), preservando numerais romanos, pares de cabeçalho (fonte PT em caixa alta) e 1 par de ênfase paralela legítima ("NÃO É BOM BATER NA PRÓPRIA ESPOSA" ↔ "TI PURANGA YANUPÁ YANÉ RIMIRIKÚ") | 180 pares em raw2/norm2; **110 em uni2** (o reparo dirigido da `uni` já havia corrigido ~70 ocorrências) |

Total: **1.620 pares alterados** em raw2/norm2 e **1.550** em uni2
(norm2: train 1.450, dev_const 12, dev_extra 71, test_const 6, test_extra 81;
fra-yrl intocado, byte-idêntico). Contagem de linhas idêntica à v1 em todos os
arquivos; campos além de `src`/`tgt` intactos; estado final de CAPS idêntico
nas 3 variantes (restam só os 5 tokens da ênfase paralela).

**Não tocados (36 casos, no log de revisão)**: glosas `[lit. ...]` cujo corte
removeria >50% do texto — conteúdo real, decisão para revisão humana.

## Auditoria profunda (03/08/2026, pós-geração)

- **Vazamento treino↔dev/test: ZERO** nas 3 variantes, nos dois lados — o
  risco mais sério da limpeza (pares que só diferiam pelo metatexto colidirem
  entre treino e avaliação) **não se materializou**.
- **Duplicatas de fonte no treino**: 383 (v1) → 394 (v2). Os 27 grupos novos
  são da classe benigna pré-existente "mesma fonte PT, outra tradução/grafia"
  (entradas cruzadas do dicionário via "o mesmo que"). Bônus: pares como
  `Anhutẽ`/`Anhunté` são material contrastivo natural para o experimento de
  tags ortográficas (docs/PESQUISA_ORTOGRAFIA_TAGS.md).
- **Elipses internas reais** (texto dos dois lados do `[...]`): 159 pares no
  train + 20 em dev/test. A remoção cola fragmentos, mas de forma **simétrica
  e paralela** nos dois lados (a omissão é a mesma) — aceitável para MT;
  documentado, não revertido.
- Pontuação: dois defeitos detectados na 1ª geração (11 vírgulas finais
  órfãs; `":."` colado) — corrigidos no `tidy()` e regenerado; varredura final
  sem resíduos novos.
- EOL: CRLF, **consistente com a v1** (mesma convenção de todo o pacote).
- Estrutura: ordem, contagens, `id`/`grupo`/`*_lang` byte-idênticos; nenhuma
  string curta/vazia criada; `uni2` comprovadamente deriva de `uni` (197
  linhas const com `puranga` preservadas).
- Volume removido (norm2): ~13,1k caracteres no PT, ~10,2k no yrl (~0,9% do
  texto) — quase tudo aparato de dicionário.

## Ressalva de comparabilidade (importante)

As **referências de dev/test também mudaram** (170 linhas de avaliação por
variante). chrF medido sobre v2 **não é comparável ponto-a-ponto** aos números
v1 (Adendos 2 e 4) — mesma situação já documentada para a variante `uni`.
Modelos v2 se comparam entre si; a ponte com v1 é qualitativa.

## Uso

```bash
# enviar ao cluster (da máquina local):
scp -r cluster/data/raw2 cluster/data/norm2 cluster/data/uni2 \
    USUARIO@HOST:$ROOT/data/
# treinar: as variantes são selecionadas por variável de ambiente
VAR=norm2 MIX=none STEPS=20000 EVAL=1000
```

Sequência acordada: v2 entra no início da **Fase 2** (junto com porMT e
extrações do backlog), depois de fechada a série v1 (uni + seeds 2-3), para
não quebrar a comparabilidade da Fase 1.
