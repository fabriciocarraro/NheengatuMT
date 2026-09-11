# Proveniência e direitos, camada por camada

As 14 fontes que compõem o corpus, com o que cada uma é, como foi alinhada, sob
que base legal está e se pode ser redistribuída. É esta tabela que decide o que
entra em `release/` e o que fica de fora, e é a mesma informação que um revisor
pediu para o camera-ready.

**Status em 09/08/2026.** Cinco permissões continuam pendentes; enquanto isso o
repositório entrega **scripts de extração** para essas camadas, não o texto.

## Resumo

| | pares | |
|---|---|---|
| redistribuíveis hoje | **6.135** treino + **682** avaliação | em `release/` |
| autorizadas para uso, escopo de publicação a confirmar | 526 (LEETRA) | fora, por cautela |
| permissão pendente | 6.364 treino + 743 avaliação + 925 FR | fora; só o extrator |

## As camadas

### ✅ Redistribuíveis

**`constituicao` — 2.031 treino + 451 avaliação**
Constituição da República Federativa do Brasil traduzida ao nheengatu, *"Mundu
Sa Turusu Waá"*, STF e CNJ, Brasília, 2023, 193 pp., ISBN 978-65-5972-113-9.
Língua original **português**; lado PT é a **edição oficial**, texto originalmente
escrito em português. Traduzida por quinze tradutores indígenas sob revisão
editorial. Ortografia da **norma escolar do Rio Negro**. Alinhamento por
**identidade de estrutura documental** (artigo, parágrafo, inciso, alínea) — cada
par carrega o endereço jurídico que o produziu, o que torna todo erro localizável.
Base: **ato oficial, art. 8º, IV, da Lei 9.610/1998** — sem proteção autoral.
Redistribuível com crédito.

**`treebank_ud` — 2.484 treino + 240 avaliação**
UD_Nheengatu-CompLin (de Alencar, PROPOR 2024). Registro misto. Ortografia
**acadêmica de Navarro e Ávila**. Base: **CC BY-NC-SA 4.0**. Redistribuível
mantendo BY-NC-SA.

**`refubium_fala` — 1.156 treino + 231 avaliação**
Fala espontânea transcrita, corpus Reich / FU Berlin, DOI
10.17169/refubium-39406. Língua original **yrl**; o lado PT é **tradução livre de
campo escrita pelos transcritores** — português regional do Alto Rio Negro e/ou
L2, não português padrão (ver as Limitations do paper). Unidade de alinhamento é
o **turno delimitado por pausa**, não a sentença. Ortografia: **transcrição de
campo**, uma quinta tradição além das quatro escritas. Alinhamento por
**sobreposição temporal ≥50%** entre as trilhas ELAN. Base: **CC BY-NC-SA 4.0**.
Redistribuível.

**`rm_tycho` — 441 treino + 48 avaliação**
Plataforma Tycho Brahe / DACILAT (FAPESP), textos históricos do século XIX.
Ortografia **oitocentista**. Base: **uso livre declarado**, API pública.
Redistribuível.

**`catecismo1944` — 23 treino**
Catecismo de 1944, registro religioso. Base: **domínio público provável pela
idade**. Redistribuível; risco baixo.

### ⚠️ Autorizada para uso, escopo de publicação a confirmar

**`leetra_*` — 526 treino** (kariama 275, tapajoara 173, leitura 52, kabari 26)
Quatro livros bilíngues da Revista Leetra Indígena / UFSCar. **Dois dialetos que
nenhuma outra fonte cobre: Içana (Baniwa) e Tapajós.** Língua original **yrl**;
lado PT é **tradução livre**. Alinhamento: o método por comprimento **falhou** —
num dos livros o PT vem **antes** da história em yrl — e foi substituído por
**realinhamento semântico** com validação literal anti-alucinação (99,6% aprovado).
Base: **autorização escrita de 04/08/2026** para *"citar partes com crédito à
Revista Leetra Indígena e aos autores"*. Isso cobre treino e paper; **publicar os
pares no corpus aberto exige confirmação de escopo**, pedida e ainda sem resposta.
Fora do `release/` por cautela — um e-mail de distância de entrar.

### ❌ Permissão pendente — só o extrator vai no repositório

| fonte | pares | o que é | pendência |
|---|---|---|---|
| `avila2021` | 2.745 + 284 | Dicionário da tese de Marcel T. Avila (USP, 2021). Lado PT é **glosa de verbete**, não texto corrido: o yrl é primário | ✅ **Autorização escrita do autor, 10/08/2026** — sem licença nomeada; permissão do titular com **crédito obrigatório**. Os 2.745 do treino entram em `release/`; os 284 dos conjuntos `extra` não, porque esses conjuntos contêm outras fontes ainda restritas |
| `melgueiro2022(_ex)` | 1.870 + 16 | Frases-exemplo da tese de Melgueiro (UnB, 2022). Lado PT é **glosa** | © autor; **nenhum pedido enviado** |
| `gurgel` | 1.562 + 144 | Compilação didática em repositório GitHub | Repo **sem licença** = todos os direitos reservados |
| `rm_navarro` | 187 + 11 | Curso de Língua Geral (Navarro, USP) | © autor; mesmo e-mail do Ávila |
| `trevisan_fr` | 925 | *Le Petit Prince* em yrl (USP, 2017). Língua original **francês**; alvo yrl humano. Entra em todo run via `--extra_train` | © autor; **nenhum pedido enviado** |

## Por que os conjuntos `extra` ficam inteiros de fora

`dev_extra` e `test_extra` têm **62% de pares de fonte restrita** (Ávila, Gurgel,
Melgueiro, Navarro). Publicar só a fatia aberta produziria um conjunto que não é
o do paper e com o qual ninguém poderia comparar. Ficam retidos inteiros — e com
eles as hipóteses guardadas sobre eles, que embutem `src` e `ref`.

**O que sobra é o que importa:** o conjunto jurídico *article-disjoint* e o de
fala, ambos 100% redistribuíveis, são os dois eixos sobre os quais o paper faz
suas afirmações centrais.

## Os recursos terminológicos: o que entra, o que não, e por quê

Todos foram **minerados dos pares de treino**, o que inclui fontes restritas. A
linha divisória não é a fonte: é **o que cada arquivo contém**.

**Entram** — `glossario_fase2.jsonl` (904 entradas), `bitermos_auditados.jsonl`
(920), `bitermos_rejeitados.jsonl` (132) e `lemas_yrl.tsv`. Medido: o lado
português tem **no máximo 3 palavras** em todos eles, e nenhuma entrada com 4 ou
mais. São pares de termo — palavra ou expressão curta e seu equivalente —, mais
próximos de um fato sobre a língua do que da expressão protegida de uma obra. A
tabela de lemas vem do treebank (CC BY-NC-SA). Se algum detentor de direitos
discordar, a camada sai.

**NÃO entra** — `lexico_pt_yrl_v0.jsonl`. Apesar do nome, **não é um léxico de
termos**: das suas 1.592 entradas, **700 (44%) têm 4 ou mais palavras do lado
português**, porque foi construído colhendo *pares curtos* do treino, e "curto"
inclui sentença. Exemplos: *"A arraia me ferrou"* (de `avila2021`), *"A camisa
toda retalhada"* (de `melgueiro2022`). Publicá-lo seria redistribuir frases de
obra protegida sob o rótulo de terminologia. Fica fora, e é **regenerável por
`scripts/build_lexico.py`** por quem tiver acesso às fontes.

⚠ **Consequência para o TSR:** o léxico de scoring do paper é a *união* de
`glossario_fase2.jsonl` com `lexico_pt_yrl_v0.jsonl`. Com só o primeiro, os
valores de TSR **não reproduzem exatamente** os publicados. Reproduzi-los exige
rodar `build_lexico.py` sobre os dados completos. Isso é uma limitação real do
release por camadas e está declarada aqui em vez de descoberta por quem tentar.
