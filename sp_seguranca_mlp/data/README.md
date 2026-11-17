# Dados – Veículos subtraídos 2025 (SSP-SP)

Documentação da base utilizada no projeto. Números obtidos por perfilamento do CSV em `raw/` (244.049 linhas × 54 colunas).

## 1. Arquivos

| Arquivo | Origem | Descrição |
|---|---|---|
| `raw/VeiculosSubtraidos_2025.xlsx` | SSP-SP (transparência ativa) | Planilha original (66 MB) com 3 abas: `METODOLOGIA`, `DICIONARIO DE DADOS` e `VEICULOS_2025` |
| `raw/VeiculosSubtraidos_2025.csv` | Aba `VEICULOS_2025` | 244.049 registros, UTF-8, separador `,` |
| `raw/dicionario_dados.csv` | Aba `DICIONARIO DE DADOS` | Descrição de cada campo (`Campos,Descrição`) |
| `processed/` | Gerado pelo notebook | Capital filtrada, deduplicada e amostrada (15k) |

`raw/` não é versionado no Git.

## 2. Fonte e metodologia

- Extraída dos sistemas **R.D.O.** e **S.P.J.** (boletins de ocorrência do Estado de SP). O S.P.J. substituiu o R.D.O. entre 2022 e 2023.
- Base **bruta**: sem filtros, padronização ou consistência. **Não é estatística oficial** da SSP; uso exploratório/analítico.
- Campos pessoais protegidos pela LAI e LGPD. Em alguns tipos de local, `LOGRADOURO` é substituído por "VEDAÇÃO DA DIVULGAÇÃO DOS DADOS RELATIVOS".
- ~60% dos boletins são registrados fora do local do fato, então município de registro ≠ município da ocorrência.

## 3. Granularidade (ponto crítico)

Cada linha é um **vínculo** (veículo, pessoa, natureza criminal ou objeto) de um boletim, não um boletim.

| Medida | Base completa | Capital (`NOME_MUNICIPIO_CIRC == "S.PAULO"`) |
|---|---|---|
| Linhas | 244.049 | 99.118 |
| Boletins únicos (`NOME_DELEGACIA` + `ANO_BO` + `NUM_BO`) | 133.706 | 54.700 |
| Veículos distintos (`CONT_VEICULO`) | ~138 mil | 56.477 |

Há ~1,8 linha por boletim na capital (um boletim com várias naturezas ou veículos gera várias linhas). Não há linhas 100% idênticas, mas há muita repetição de informação.

**Consequência para o modelo:** amostrar 15 mil linhas sem deduplicar coloca o mesmo boletim em treino e teste e gera **vazamento de dados**. Defina a unidade de análise (boletim ou veículo), deduplique antes da amostragem e, se mantiver vários registros por boletim, faça o split por grupo (`GroupShuffleSplit` por boletim).

## 4. Recorte da capital

- `NOME_MUNICIPIO_CIRC` (local do fato): **99.118** linhas, valor único `S.PAULO`.
- `NOME_MUNICIPIO` (local do registro): 98.954 linhas. Só 0,3% das linhas da capital diferem entre os dois campos.
- Para o objetivo de segurança por local, prefira `NOME_MUNICIPIO_CIRC`. Hoje `CITY_COL` em `src/data_prep.py` aponta para `NOME_MUNICIPIO` (ajustar).
- Dados da capital: 1.722 bairros distintos (inclui variações de grafia) e 94 delegacias de circunscrição.

## 5. Variável-alvo candidata

`DESCR_OCORRENCIA_VEICULO` (3 classes, sem nulos). Distribuição na capital:

| Classe | Linhas | % |
|---|---|---|
| Localizado / Entregue | 52.952 | 53,4% |
| Furtado | 35.769 | 36,1% |
| Roubado | 10.397 | 10,5% |

- Desbalanceamento moderado, em especial na classe `Roubado`. Usar estratificação, `class_weight` e métricas como F1 macro.
- **Alternativa mais útil para o app:** prever **Furtado vs. Roubado** (46.166 linhas), descartando "Localizado / Entregue", que é o desfecho posterior, não o tipo de crime.
- **Risco de vazamento:** `RUBRICA`, `DESDOBRAMENTO`, `CIRCUNSTANCIA`, `DESCR_CONDUTA` e `DESC_LEI` descrevem a natureza jurídica e quase determinam o alvo (ex.: `Furto (art. 155)` ≈ `Furtado`). Não use como entrada.

## 6. Dicionário resumido e qualidade (base completa)

| Grupo | Colunas | Observações |
|---|---|---|
| Identificação | `ID_DELEGACIA`, `NOME_DELEGACIA`, `NUM_BO`, `ANO_BO`, `VERSAO`, `CONT_VEICULO` | Chave do boletim: delegacia + ano + número |
| Registro (administrativo) | `NOME_DEPARTAMENTO`, `NOME_SECCIONAL`, `NOME_MUNICIPIO`, `DESCRICAO_APRESENTACAO`, `DATAHORA_*`, `DATA_COMUNICACAO_BO`, `MES_REGISTRO_BO`, `ANO_REGISTRO_BO` | Local de registro, não do fato |
| Circunscrição (local do fato) | `*_CIRC`, `CMD`, `BTL`, `CIA`, `CD_IBGE` | Sem nulos |
| Tempo do fato | `DATA_OCORRENCIA_BO`, `HORA_OCORRENCIA` (12,3% nulos), `DESCR_PERIODO` (**87,7% nulos**) | Derivar hora, dia da semana e mês |
| Local | `DESCR_TIPOLOCAL`, `DESCR_SUBTIPOLOCAL`, `BAIRRO` (1,7%), `CEP` (10,8%), `LOGRADOURO`, `NUMERO_LOGRADOURO` (9,6%), `DESC_NATUREZA_LOCAL` (5,3%) | Via pública = 77% |
| Geografia | `LATITUDE`, `LONGITUDE` (4,1% nulos) | Há coordenadas `0` (6.330 na capital) = inválidas |
| Veículo | `DESCR_TIPO_VEICULO` (23), `DESCR_MARCA_VEICULO` (6.755), `DESCR_COR_VEICULO` (17), `ANO_FABRICACAO` (0,6%), `ANO_MODELO` (**28,5%**), `PLACA_VEICULO` | Marca com altíssima cardinalidade |
| Natureza criminal | `RUBRICA` (159), `DESC_LEI`, `DESDOBRAMENTO` (95,7%), `CIRCUNSTANCIA` (79,5%), `DESCR_CONDUTA` (37,3%) | Possível vazamento do alvo |
| Flags | `AUTORIA_BO` (C/D/S), `FLAG_FLAGRANTE`, `FLAG_ATO_INFRACIONAL`, `FLAG_INTOLERANCIA`, `FLAG_STATUS`, `TIPO_INTOLERANCIA` | `FLAG_STATUS` (só `CONSUMADO`) e `TIPO_INTOLERANCIA` (100% nulo) não têm informação |

Descrições completas em `raw/dicionario_dados.csv`.

## 7. Problemas conhecidos e tratamento sugerido

| Problema | Evidência | Tratamento |
|---|---|---|
| Registros repetidos por boletim | 54.700 boletins em 99.118 linhas | Deduplicar e/ou split por grupo |
| Coordenadas inválidas | `LATITUDE`/`LONGITUDE` = 0; nulos 4,1% | Tratar como nulo; descartar ou imputar |
| Anos fora do recorte | `ANO_BO` < 2025 em 84 linhas (25 na capital); `DATA_OCORRENCIA_BO` mínima 2005 | Filtrar ou criar feature "defasagem registro − fato" |
| Colunas sem variância | `FLAG_STATUS`, `ANO_REGISTRO_BO`, `TIPO_INTOLERANCIA` | Remover |
| Alta cardinalidade | `DESCR_MARCA_VEICULO` (6.755), `BAIRRO`, `LOGRADOURO` | Agrupar em top-N + "Outros", ou usar embeddings |
| Nulos altos | `DESCR_PERIODO` 87,7%, `ANO_MODELO` 28,5% | Derivar período da `HORA_OCORRENCIA`; indicador de ausência |
| Colunas de identificação | `NUM_BO`, `PLACA_VEICULO`, `CONT_VEICULO`, `LOGRADOURO` | Não usar como feature (ID/dado pessoal) |
| Mudança de sistema R.D.O. → S.P.J. | Documentado na metodologia | Só 2025 aqui, impacto baixo |

## 8. Features sugeridas para o MLP

- **Tempo:** hora (0–23, com codificação cíclica seno/cosseno), dia da semana, mês, fim de semana.
- **Local:** `DESCR_TIPOLOCAL`, `DESCR_SUBTIPOLOCAL` (agrupado), bairro (top-N), latitude/longitude válidas, `NOME_DELEGACIA_CIRC`.
- **Veículo:** tipo, cor, idade do veículo (`ANO_BO − ANO_FABRICACAO`), marca agrupada.
- **Contexto:** `AUTORIA_BO`, `DESCRICAO_APRESENTACAO`, `FLAG_FLAGRANTE`.

## 9. Pipeline proposto (raw → processed)

1. Ler `raw/VeiculosSubtraidos_2025.csv`.
2. Filtrar `NOME_MUNICIPIO_CIRC == "S.PAULO"`.
3. Remover coordenadas inválidas e colunas sem variância.
4. Definir a unidade de análise e deduplicar.
5. Definir o alvo (ver seção 5).
6. Amostrar 15 mil registros, estratificados pelo alvo.
7. Salvar em `processed/` e ajustar encoder/scaler **só no treino**.

