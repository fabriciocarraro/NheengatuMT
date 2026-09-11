# Série v3 (`norm3`) — norm2 + backlog auditado da Fase 2

Gerada por `scripts/build_data_v3.py` em 04/08/2026. Base: `norm2` intocada.

## O que entrou (3.408 pares novos de treino; todos auditados por agentes com evidência — RESULTADOS_FASE2)

| Bloco | grupo | Pares | Origem e auditoria |
|---|---|---|---|
| Refubium (fala espontânea) | `refubium_fala` | 1.156 treino (+231 eval) | EAFs Reich/FU Berlin, alinhamento temporal; auditoria 200/1.575, filtro code-switch (−42); typos do PT: 319 correções mínimas validadas (registro coloquial preservado) |
| Melgueiro (exemplos) | `melgueiro2022_ex` | 1.726 | Tese UnB 2022, caminhada token a token; auditoria 240 vs fonte, 31 correções; só `pt_completo`. (O grupo `melgueiro2022` SEM sufixo são as 144 linhas antigas do agregado — proveniências distintas.) |
| LEETRA Kariamã | `leetra_kariama` | 275 | Realinhamento semântico por agentes (layout PT-antes-do-yrl!), validação literal contra a fonte |
| LEETRA Tapajoara | `leetra_tapajoara` | 173 | idem (dialeto do Tapajós) |
| LEETRA leitura/escrita | `leetra_leitura` | 52 | DP auditado par a par; CAPS→sentence-case (52 linhas) |
| LEETRA Kabari | `leetra_kabari` | 26 | DP auditado par a par |

Descartes na fusão: 111 duplicatas exatas (backchannels repetidos entre
sessões + 2 Melgueiro), 1 vazamento p/ conjunto de avaliação (excluído).

## Conjuntos novos de avaliação — registro oral transcrito ("fala" = texto de transcrição, não áudio)

- `dev_fala.por-yrl.jsonl` (116) e `test_fala.por-yrl.jsonl` (115).
- **Falantes e sessões disjuntos do treino**: holdout do grupo de sessão
  inteiro `ESC03_MRPM04` (268 pares). Antes do split alternado por tarefa:
  dedupe por conteúdo DENTRO do holdout (backchannels repetidos cairiam nos
  dois lados, correlacionando dev e teste — achado da auditoria de
  integração); depois, guarda anti-vazamento derruba itens cujo yrl é
  idêntico a falas curtas de outros falantes no treino.
- Disciplina: `test_fala` segue a regra do teste único (gastar uma vez, no
  fim). `dev_fala` pode ser usado em desenvolvimento.

## Arquivos auxiliares adicionados DEPOIS do build (não são parte da série)

Experimentos posteriores depositaram em `norm3/` arquivos de token pairs
alternativos, todos inertes até serem passados explicitamente por flag
(`LEXV=`/`--extra_train`): `train.lexesp-yrl.jsonl` (lex+54 espaciais,
pack de fechamento — rejeitado), `train.lex2-yrl.jsonl` (lex+4.259
verbetes — rejeitado), `train.orf-yrl.jsonl` (430 sintéticos brutos do
A4, COM campo alvos_ok; NÃO usar direto), `train.lexorf-yrl.jsonl`
(lex+233 sintéticos filtrados — em validação). Documentação:
RESULTADOS_FASE2 (pack de fechamento e camera-ready). Os arquivos da
série original permanecem os 10 listados na auditoria abaixo.

## Auditoria de integração (04/08/2026) — APROVADA

Checagens automatizadas (todas OK): dev/test const/extra e
fra/lex/porMT **byte a byte iguais ao norm2**; 9.617 primeiras linhas do
train literais do norm2; sem CRLF; blocos novos sem lado vazio, sem par
identidade, direção src=PT/tgt=yrl verificada por sinal lexical (0
suspeitos), ids únicos; eval de fala 100% da sessão holdout, treino sem
nenhum id dela, 0 vazamentos de yrl, dev∩test vazio; 319/319 typos no
lugar certo; sentence-case restrito a `leetra_leitura`. Também verificado
no código: `train_nllb.py` carrega devs de early stopping por NOME FIXO
(dev_const+dev_extra) — `dev_fala` não altera o treino — e o treino não
faz glob de `train.*` (extra_train só entra por flag do pack4).

## Convenções herdadas (e por que a comparação continua válida)

- **dev/test const e extra são CÓPIAS BYTE A BYTE do norm2** → chrF++ de
  runs norm3 é diretamente comparável aos Experimentos 1-2.
- Normalização yrl: mesmo charmap de glifos do `align.py` (aplicado:
  ī→ĩ 35×, ´ 8×, ữ→ũ 6×, û→ũ 1×); espaços múltiplos; espaço antes de
  pontuação. PT: só espaços. Texto bruto preservado nos arquivos-fonte de
  `work/` (refubium/melgueiro/leetra_pares.jsonl).
- Esquema de linha idêntico; campos extras inexistentes (grupo/id como
  sempre). `train.fra-yrl`, `train.lex-yrl`, `train.porMT-yrl` copiados sem
  mudança — flags ANNOT/LEX/PORMT do pack4 funcionam igual com `VAR=norm3`.

## Licenças por bloco (PARECER_LICENCAS se aplica)

- Refubium: CC BY-NC-SA 4.0 (DOI 10.17169/refubium-39406) — treino/paper ok;
  camada publicada do corpus herda BY-NC-SA.
- Melgueiro: tese pública UnB — citação obrigatória; publicação em massa dos
  pares na camada aberta: avaliar (autor é candidato a contato via Academia).
- LEETRA: autorização "citar partes com crédito" (treino/paper ok);
  publicação dos pares no corpus aberto aguarda confirmação do grupo
  (e-mail enviado 04/08/2026).

## Registro/domínio

Sem tags de domínio nesta série (decisão: medir primeiro o efeito puro dos
dados; tags de registro/dialeto/ortografia = experimento próprio, ver
PESQUISA_ORTOGRAFIA_TAGS). Os grupos (`grupo`) preservam a proveniência
para tagging futuro sem reconstruir nada.
