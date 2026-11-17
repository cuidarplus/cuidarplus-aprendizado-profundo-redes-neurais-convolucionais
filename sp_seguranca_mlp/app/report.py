"""Abas de Documentação Técnica (referência do sistema) e Relatório Técnico (tópicos do edital)."""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from torch import nn

from ui_theme import br

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS_DIR = ROOT / "reports" / "screenshots"

INTEGRANTES = [
    ("José Ribeiro dos Santos Neto", "567692"),
    ("Danilo Roberto Anterio dos Santos", "566966"),
]

SOURCE_URL = "https://www.ssp.sp.gov.br/estatistica/consultas"

# Funil medido na execução de src/data_prep.py sobre o CSV bruto de 2025.
FUNNEL = [
    ("Base bruta (estado de SP)", 244_049, "54 colunas; cada linha é um vínculo de um boletim, não um boletim"),
    ("Capital", 99_118, '`NOME_MUNICIPIO_CIRC == "S.PAULO"` (local do fato, não do registro)'),
    ("Apenas Furtado ou Roubado", 46_166, "Descarta `Localizado / Entregue`, que não é subtração"),
    ("Uma linha por boletim", 39_502, "Deduplicação por `NOME_DELEGACIA` + `ANO_BO` + `NUM_BO`, mantendo a primeira"),
    ("Amostra estratificada", 15_000, "Proporção do alvo preservada: 83,3% furto e 16,7% roubo"),
]

FEATURES = {
    "NOME_SECCIONAL_CIRC": ("Seccional da área do fato", "Sem alteração"),
    "NOME_DELEGACIA_CIRC": ("Delegacia da área do fato", "Top-20 do treino + OUTROS"),
    "BTL": ("Batalhão da PM da área do fato", "Sem alteração"),
    "CIA": ("Companhia da PM da área do fato", "Sem alteração"),
    "PERIODO_FATO": ("`HORA_OCORRENCIA` e `DESCR_PERIODO`",
                     "Hora em 4 faixas de 6 h; sem hora, usa o período declarado; senão, Incerto"),
    "DESCR_TIPOLOCAL": ("Tipo de local", "Sem alteração"),
    "DESCR_SUBTIPOLOCAL": ("Subtipo de local", "Sem alteração"),
    "BAIRRO": ("Bairro do fato", "Caixa alta, Top-20 do treino + OUTROS"),
    "DESCR_TIPO_VEICULO": ("Tipo de veículo", "Sem alteração"),
    "MARCA_VEICULO": ("`DESCR_MARCA_VEICULO`",
                      "Extrai a marca (`I/VW JETTA` vira `VW`), unifica grafias, Top-20 + OUTROS"),
    "DESCR_COR_VEICULO": ("Cor do veículo", "Sem alteração"),
    "LATITUDE": ("Latitude do fato", "Fora do município vira nulo; imputa mediana do bairro ou da cidade"),
    "LONGITUDE": ("Longitude do fato", "Fora do município vira nulo; imputa mediana do bairro ou da cidade"),
    "IDADE_VEICULO": ("`ANO_BO` e `ANO_FABRICACAO`", "Ano do boletim menos ano de fabricação; imputa mediana do treino"),
}

EXCLUDED = [
    ("Natureza jurídica", "`RUBRICA`, `DESC_LEI`, `DESDOBRAMENTO`, `CIRCUNSTANCIA`, `DESCR_CONDUTA`",
     "Descrevem o crime e revelam o alvo diretamente"),
    ("Identificadores", "`NUM_BO`, `PLACA_VEICULO`, `CONT_VEICULO`, `LOGRADOURO`, `NUMERO_LOGRADOURO`",
     "Identificam o boletim, o veículo ou o endereço exato"),
    ("Pós-fato", "`FLAG_FLAGRANTE`, `AUTORIA_BO`, `FLAG_STATUS`, `DESCRICAO_APRESENTACAO`, `VERSAO`",
     "Só são conhecidos depois da investigação ou do trâmite do registro"),
    ("Datas de registro", "`DATAHORA_REGISTRO_BO`, `DATA_COMUNICACAO_BO`, `DATAHORA_IMPRESSAO_BO`, "
     "`MES_REGISTRO_BO`, `ANO_REGISTRO_BO`", "São do registro, não do fato"),
    ("Delegacia de registro", "`ID_DELEGACIA`, `NOME_DELEGACIA`, `NOME_DEPARTAMENTO`, `NOME_SECCIONAL`, "
     "`NOME_MUNICIPIO`", "A Delegacia Eletrônica concentra 62% dos boletins e tem 10,6% de roubos, contra 25,3% "
     "nas delegacias do DECAP: o canal de registro é consequência do tipo de crime"),
    ("Ano do modelo", "`ANO_MODELO`", "Nulo em 9.214 dos 9.230 boletins da Delegacia Eletrônica: a ausência "
     "do campo reproduz o vazamento do canal de registro"),
    ("Sem sinal", "`ANO_BO`, `CD_IBGE`, `TIPO_INTOLERANCIA`, `FLAG_INTOLERANCIA`, `FLAG_ATO_INFRACIONAL`, "
     "`DESC_NATUREZA_LOCAL`, `LOGRADOURO_VERSAO`, `CEP`, `CMD`", "Constantes, quase constantes ou redundantes"),
]

ARCH_RATIONALE = {
    0: "Ponto de partida: a menor rede capaz de aprender interações entre as features. Sem BatchNorm e com "
       "dropout leve, serve de referência para medir se mais capacidade ajuda.",
    1: "Dobra a largura e adiciona uma segunda camada para capturar interações de ordem maior. BatchNorm "
       "estabiliza o treino da rede mais profunda e o dropout sobe para 0,3 para conter o aumento de parâmetros.",
    2: "Testa o limite de capacidade: três camadas em funil e sete vezes mais parâmetros que a Arq. 1. O "
       "dropout sobe para 0,4 porque o risco de memorizar o treino cresce com o tamanho da rede.",
}

METRIC_LABELS = {"accuracy": "Acurácia", "precision_macro": "Precision (macro)",
                 "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)"}


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def md_table(headers: list[str], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def integer(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def layer_rows(model) -> list[tuple]:
    """Uma linha por camada da rede: tipo, dimensões e parâmetros treináveis."""
    rows = []
    for module in list(model.hidden) + [model.output]:
        params = sum(p.numel() for p in module.parameters())
        if isinstance(module, nn.Linear):
            desc = f"{module.in_features} → {module.out_features}"
            formula = f"{module.in_features} × {module.out_features} pesos + {module.out_features} vieses"
        elif isinstance(module, nn.Dropout):
            desc, formula = f"p = {br(module.p, 'g')}", "Ativo só no treino"
        elif isinstance(module, nn.BatchNorm1d):
            desc, formula = f"{module.num_features} canais", "Escala e deslocamento por neurônio"
        else:
            desc, formula = "Elemento a elemento", "Sem parâmetros"
        rows.append((type(module).__name__, desc, integer(params), formula))
    return rows


def describe_arch(config: dict) -> tuple[str, str, str, str]:
    hidden = config["hidden_dims"]
    return (str(len(hidden)), " → ".join(map(str, hidden)), "Sim" if config["batch_norm"] else "Não",
            br(config["dropout"][0], "g"))


def curve_chart(history: dict, key: str, label: str, best_epoch: int, palette: dict) -> alt.Chart:
    df = pd.DataFrame({"Época": history["epoch"], "Treino": history[f"train_{key}"],
                       "Validação": history[f"val_{key}"]}).melt("Época", var_name="Conjunto", value_name=label)
    lines = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("Época:Q", axis=alt.Axis(tickMinStep=1, grid=False)),
        y=alt.Y(f"{label}:Q", scale=alt.Scale(zero=False), axis=alt.Axis(format=".2f")),
        color=alt.Color("Conjunto:N", scale=alt.Scale(domain=["Treino", "Validação"], range=palette["series"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Época:Q", "Conjunto:N", alt.Tooltip(f"{label}:Q", format=".4f")],
    )
    best = alt.Chart(pd.DataFrame({"Época": [best_epoch]})).mark_rule(
        strokeDash=[4, 4], color=palette["muted"]).encode(x="Época:Q", tooltip=alt.value("Melhor época (val_loss)"))
    return (lines + best).properties(height=210)


def confusion_html(cm: list[list[int]], palette: dict) -> str:
    """Matriz de confusão com intensidade proporcional ao % da classe real."""
    names = ["Furtado", "Roubado"]
    html = "<table class='cm'><tr><th></th>" + "".join(f"<th>Previsto {n}</th>" for n in names) + "</tr>"
    for i, row in enumerate(cm):
        html += f"<tr><th>Real {names[i]}</th>"
        for value in row:
            pct = value / sum(row)
            color = "#FFFFFF" if pct > 0.5 else palette["text"]
            html += (f"<td style='background: rgba({palette['heat']}, {0.12 + 0.88 * pct:.2f}); color: {color}'>"
                     f"<b>{integer(value)}</b><br><span>{br(pct, '.1%')}</span></td>")
        html += "</tr>"
    return html + "</table>"


def section(icon: str, title: str):
    box = st.container(border=True)
    box.subheader(f":material/{icon}: {title}")
    return box


# --------------------------------------------------------------------------- #
# Aba: Documentação Técnica
# --------------------------------------------------------------------------- #
def render_docs(meta: dict, encoder, model, checkpoint: dict, experiments: dict) -> None:
    st.caption("Referência de como o sistema funciona: do dado bruto à resposta da aplicação.")

    st.subheader(":material/account_tree: PIPELINE")
    st.markdown(md_table(
        ["Etapa", "Código", "Entrada", "Saída"],
        [("1. Preparação", "`src/data_prep.py` (`run`)", "`data/raw/VeiculosSubtraidos_2025.csv`",
          "`data/processed/sample_15k.csv`"),
         ("2. Features", "`src/data_prep.py` (`build_features`, `fit_preprocessor`)", "Amostra de 15k",
          "14 features, `preprocess_meta.pkl`"),
         ("3. Codificação", "`src/data_prep.py` (`fit_transformers`)", "Features do treino",
          "`encoder.pkl`, `scaler.pkl`, matriz de 218 colunas"),
         ("4. Treino", "`notebooks/01_modelagem_mlp.ipynb`, `src/architecture.py`", "Tensores de treino e validação",
          "`best_mlp_model.pt`, `experiment_results.json`"),
         ("5. Inferência", "`app/main.py`", "Formulário da sidebar", "Probabilidade e classe")],
    ))

    st.subheader(":material/filter_alt: FUNIL DE DADOS")
    st.markdown(md_table(["Etapa", "Linhas", "Regra"], [(e, integer(n), r) for e, n, r in FUNNEL]))

    st.subheader(":material/view_column: FEATURES DE ENTRADA")
    widths = dict(zip(meta["categorical_cols"], (len(c) for c in encoder.categories_)))
    rows = []
    for name, (origin, transform) in FEATURES.items():
        categorical = name in widths
        rows.append((f"`{name}`", origin, transform, "One-Hot" if categorical else "StandardScaler",
                     widths.get(name, 1)))
    st.markdown(md_table(["Feature", "Origem", "Transformação", "Codificação", "Colunas"], rows))
    st.caption(f"Total: {sum(r[-1] for r in rows)} colunas na entrada da rede. Categoria desconhecida na "
               "inferência gera um vetor de zeros (`handle_unknown='ignore'`).")

    st.subheader(":material/shield: COLUNAS EXCLUÍDAS E PREVENÇÃO DE VAZAMENTO")
    st.markdown(md_table(["Grupo", "Colunas", "Motivo"], EXCLUDED))
    st.markdown(
        "Além das exclusões, a divisão em treino, validação e teste é feita **antes** de qualquer estatística: "
        "categorias Top-20, medianas de imputação, encoder e scaler são aprendidos somente no treino. A "
        "deduplicação por boletim impede que o mesmo boletim apareça em dois conjuntos."
    )

    col_arch, col_train = st.columns(2)
    with col_arch:
        st.subheader(":material/hub: REDE EM PRODUÇÃO")
        st.markdown(md_table(["Camada", "Dimensão", "Parâmetros", "Observação"], layer_rows(model)))
        st.caption(f"{checkpoint['architecture_name']}: {integer(model.n_parameters())} parâmetros treináveis. "
                   "A saída é um logit; a sigmoid é aplicada na inferência.")
    with col_train:
        st.subheader(":material/model_training: HIPERPARÂMETROS")
        t = experiments["training"]
        st.markdown(md_table(
            ["Parâmetro", "Valor"],
            [("Otimizador", t["optimizer"]), ("Taxa de aprendizado", br(t["learning_rate"], "g")),
             ("Função de perda", f"`{t['loss']}`"), ("`pos_weight`", br(t["pos_weight"], ".4f")),
             ("Batch size", t["batch_size"]), ("Limite de épocas", t["max_epochs"]),
             ("Early stopping", f"`{t['early_stopping_monitor']}`, paciência {t['early_stopping_patience']}"),
             ("Semente", t["seed"])],
        ))

    st.subheader(":material/inventory_2: ARTEFATOS EM `models/`")
    st.markdown(md_table(
        ["Arquivo", "Conteúdo", "Usado em"],
        [("`best_mlp_model.pt`", "Pesos (`state_dict`), configuração da rede, limiares, métricas de teste",
          "Inferência"),
         ("`encoder.pkl`", "`OneHotEncoder` ajustado no treino", "Inferência"),
         ("`scaler.pkl`", "`StandardScaler` ajustado no treino", "Inferência"),
         ("`preprocess_meta.pkl`", "Listas Top-20, medianas de imputação e tabelas de apoio do formulário",
          "Inferência e formulário"),
         ("`experiment_results.json`", "Histórico, métricas e matrizes de confusão das três arquiteturas",
          "Documentação e relatório")],
    ))

    st.subheader(":material/bolt: FLUXO DE INFERÊNCIA")
    st.markdown(
        "1. O formulário monta um `DataFrame` de uma linha com as 14 features. Seccional, batalhão e companhia "
        "são derivados da delegacia de circunscrição; as coordenadas vêm da mediana do bairro ou da delegacia.\n"
        "2. `apply_preprocessor` agrupa categorias fora do Top-20 em OUTROS e imputa nulos.\n"
        "3. `transform_features` aplica o scaler e o encoder e concatena em um vetor `float32` de 218 posições.\n"
        "4. O vetor vira tensor e passa pela rede com `model.eval()` e `torch.no_grad()`.\n"
        "5. A sigmoid converte o logit em probabilidade de roubo, comparada ao limiar escolhido no slider."
    )

    st.subheader(":material/terminal: REPRODUÇÃO")
    st.code(
        "cd sp_seguranca_mlp\n"
        "python -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "python src/data_prep.py                       # gera data/processed/sample_15k.csv\n"
        "jupyter nbconvert --to notebook --execute --inplace notebooks/01_modelagem_mlp.ipynb\n"
        "streamlit run app/main.py",
        language="bash",
    )


# --------------------------------------------------------------------------- #
# Aba: Relatório Técnico (10 tópicos do edital)
# --------------------------------------------------------------------------- #
def render_report(meta: dict, encoder, model, checkpoint: dict, experiments: dict, palette: dict) -> None:
    archs = experiments["architectures"]
    best_name = experiments["best"]
    best = archs[best_name]
    training = experiments["training"]
    split = experiments["split"]

    with section("flag", "APRESENTAÇÃO DO PROJETO"):
        col_a, col_b = st.columns([3, 2])
        col_a.markdown(
            "**Classificador de ocorrências: Roubo vs Furto de veículos.**\n\n"
            "**Objetivo:** treinar uma rede neural Multilayer Perceptron que, a partir das características de "
            "uma ocorrência de veículo subtraído na cidade de São Paulo (onde, quando e qual veículo), indique "
            "se o perfil é mais compatível com **Roubo** (com violência ou grave ameaça) ou com **Furto** (sem "
            "violência), e disponibilizar o modelo em uma aplicação web."
        )
        col_b.markdown("**Integrantes**")
        col_b.markdown(md_table(["Nome completo", "RM"], INTEGRANTES))

    with section("database", "PROBLEMA E FONTE DE DADOS"):
        st.markdown(
            "**Problema:** classificação binária supervisionada. Alvo `DESCR_OCORRENCIA_VEICULO`: "
            "Furtado (0) ou Roubado (1).\n\n"
            f"**Fonte pública:** Secretaria da Segurança Pública de São Paulo, base *Veículos subtraídos 2025*, "
            f"em [{SOURCE_URL.removeprefix('https://')}]({SOURCE_URL}). Extraída dos boletins de ocorrência do "
            "estado (sistemas R.D.O. e S.P.J.)."
        )
        st.markdown(md_table(["Recorte", "Registros", "Critério"], [(e, integer(n), r) for e, n, r in FUNNEL]))
        st.markdown(
            "**Atributos:** a base tem 54 colunas; 20 foram descartadas na preparação e 14 features entram no "
            "modelo, em três grupos: local (delegacia, seccional, batalhão e companhia da área, bairro, tipo e "
            "subtipo de local, coordenadas), tempo (período do dia) e veículo (tipo, marca, cor e idade).\n\n"
            "**Justificativa:** é uma fonte oficial e aberta, com volume suficiente para treinar uma rede "
            "neural, alvo bem definido pela legislação e atributos mistos (categóricos e numéricos) que exigem "
            "as etapas de codificação e escalonamento pedidas no projeto. O recorte da capital mantém um "
            "contexto urbano único e a deduplicação por boletim garante observações independentes."
        )

    with section("cleaning_services", "PREPARAÇÃO DOS DADOS"):
        col_a, col_b = st.columns(2)
        n_onehot = sum(len(c) for c in encoder.categories_)
        col_a.markdown(
            "**Variáveis escolhidas (14)**\n"
            f"- {len(meta['categorical_cols'])} categóricas: seccional, delegacia, batalhão e companhia da área "
            "do fato, período, tipo e subtipo de local, bairro, tipo, marca e cor do veículo.\n"
            f"- {len(meta['numeric_cols'])} numéricas: latitude, longitude e idade do veículo.\n\n"
            "**Tratamento**\n"
            "- Período do fato: hora em quatro faixas; sem hora, usa o período declarado ou Incerto.\n"
            "- Marca extraída do texto do modelo; bairro, marca e delegacia reduzidos a Top-20 + OUTROS.\n"
            "- Coordenadas fora do município tratadas como nulas e imputadas pela mediana do bairro.\n"
            "- Idade do veículo: ano do boletim menos ano de fabricação; mediana do treino nos nulos."
        )
        col_b.markdown(
            "**Codificação e escalonamento**\n"
            f"- `OneHotEncoder` nas categóricas: {n_onehot} colunas.\n"
            f"- `StandardScaler` nas numéricas: {len(meta['numeric_cols'])} colunas.\n"
            f"- Entrada da rede: **{checkpoint['config']['input_dim']} colunas**.\n\n"
            "**Separação dos conjuntos (estratificada pelo alvo)**"
        )
        col_b.markdown(md_table(
            ["Conjunto", "Registros", "Roubos", "% roubo"],
            [(nome, integer(split[k]["n"]), integer(split[k]["roubo"]), br(split[k]["roubo"] / split[k]["n"], ".1%"))
             for nome, k in [("Treino (70%)", "treino"), ("Validação (15%)", "validacao"), ("Teste (15%)", "teste")]],
        ))
        st.markdown(
            "**Prevenção de vazamento:** 20 colunas removidas por revelarem o alvo (natureza jurídica), "
            "identificarem o boletim ou só existirem após o fato. A delegacia de registro também saiu, porque a "
            "Delegacia Eletrônica concentra furtos. Todas as estatísticas são aprendidas apenas no treino."
        )

    with section("hub", "ARQUITETURA DA MLP"):
        col_a, col_b = st.columns(2)
        col_a.markdown(f"**Modelo escolhido: {best_name}**")
        col_a.markdown(md_table(["Camada", "Dimensão", "Parâmetros", "Observação"], layer_rows(model)))
        col_a.caption(f"Total: {integer(model.n_parameters())} parâmetros treináveis.")
        with col_b:
            st.markdown("**Funcionamento**")
            st.latex(r"h = \mathrm{ReLU}(W_1 x + b_1) \qquad z = W_2\,\mathrm{Dropout}(h) + b_2 \qquad "
                     r"P(\text{roubo}) = \sigma(z) = \frac{1}{1 + e^{-z}}")
            st.markdown(
                "- **Entrada:** vetor de 218 valores (3 numéricos padronizados e 215 indicadores One-Hot).\n"
                "- **Camada oculta:** 64 neurônios totalmente conectados com ativação ReLU, que introduz a não "
                "linearidade necessária para combinar local, horário e veículo.\n"
                "- **Dropout:** desliga 10% dos neurônios a cada passo de treino para reduzir overfitting.\n"
                "- **Saída:** um neurônio linear (logit); a sigmoid o converte em probabilidade de roubo.\n"
                "- **Implementação:** classe `MLP` em PyTorch, parametrizável em número de camadas, neurônios, "
                "dropout, ativação e BatchNorm."
            )

    with section("model_training", "CONFIGURAÇÃO DO TREINAMENTO"):
        col_a, col_b = st.columns(2)
        col_a.markdown(md_table(
            ["Item", "Configuração"],
            [("Função de perda", f"`{training['loss']}` com `pos_weight = {br(training['pos_weight'], '.2f')}`"),
             ("Otimizador", training["optimizer"]),
             ("Taxa de aprendizado", br(training["learning_rate"], "g")),
             ("Batch size", training["batch_size"]),
             ("Épocas", f"Limite de {training['max_epochs']}, com early stopping"),
             ("Early stopping", f"Monitora a perda de validação, paciência de "
                                f"{training['early_stopping_patience']} épocas, restaura os melhores pesos")],
        ))
        col_b.markdown(
            f"- **Perda ponderada:** entropia cruzada binária sobre o logit. Como só 16,7% dos registros são "
            f"roubos, cada roubo pesa {br(training['pos_weight'], '.2f')} vezes mais (razão entre furtos e roubos "
            "no treino), evitando que a rede aprenda a prever sempre furto.\n"
            "- **Épocas efetivas:** "
            + "; ".join(f"{name.split(' - ')[0]} parou na época {a['epochs_trained']} (melhor: {a['best_epoch']})"
                        for name, a in archs.items()) + ".\n"
            "- **Comparação justa:** as três arquiteturas usam a mesma configuração, a mesma semente e os mesmos "
            "conjuntos; só a rede muda."
        )

    with section("science", "ARQUITETURAS TESTADAS"):
        rows = []
        for i, (name, a) in enumerate(archs.items()):
            layers, neurons, batch_norm, dropout = describe_arch(a["config"])
            rows.append((name.split(" (")[0], layers, neurons, batch_norm, dropout, integer(a["n_parameters"]),
                         ARCH_RATIONALE[i]))
        st.markdown(md_table(
            ["Arquitetura", "Camadas ocultas", "Neurônios", "BatchNorm", "Dropout", "Parâmetros",
             "Justificativa da alteração"], rows))
        st.caption("Ativação ReLU nas três. Variam profundidade, largura, regularização e normalização.")

    with section("leaderboard", "COMPARAÇÃO E ESCOLHA DO MODELO"):
        rows = []
        for name, a in archs.items():
            mark = "**" if name == best_name else ""
            rows.append((f"{mark}{name}{mark}", *(f"{mark}{br(a['test_metrics'][k], '.4f')}{mark}" for k in METRIC_LABELS),
                         br(a["val_metrics"]["f1_macro"], ".4f")))
        base = experiments["majority_baseline_test"]
        rows.append(("Referência: prever sempre Furtado", br(base["accuracy"], ".4f"), "—", "—",
                     br(base["f1_macro"], ".4f"), "—"))
        st.markdown(md_table(["Arquitetura", *(f"{v} teste" for v in METRIC_LABELS.values()), "F1 (macro) validação"],
                             rows))
        report = experiments["best_class_report_test"]["Roubado"]
        st.markdown(
            f"**Modelo escolhido: {best_name}.** O critério foi o maior F1 macro na **validação**; o teste não "
            "participou da escolha e confirma a mesma ordem. A acurácia não serve de critério: prever sempre "
            f"Furtado atinge {br(base['accuracy'], '.1%')} de acurácia com F1 macro de "
            f"{br(base['f1_macro'], '.1%')}. O modelo escolhido recupera {br(report['recall'], '.1%')} dos roubos "
            f"com precisão de {br(report['precision'], '.1%')} nessa classe. Entre redes de desempenho próximo, a "
            "mais simples também é preferível por generalizar melhor e custar menos."
        )

    with section("monitoring", "CURVAS, MATRIZ DE CONFUSÃO E OVERFITTING"):
        st.markdown("**Curvas de treino e validação** (linha tracejada: época com a menor perda de validação)")
        for key, label in [("loss", "Loss"), ("acc", "Acurácia")]:
            for col, (name, a) in zip(st.columns(len(archs)), archs.items()):
                col.markdown(f"<span class='eyebrow'>{name.split(' (')[0]} · {label}</span>", unsafe_allow_html=True)
                col.altair_chart(curve_chart(a["history"], key, label, a["best_epoch"], palette), width="stretch")
        st.markdown("**Matrizes de confusão no teste** (contagem e % da classe real, limiar 0,50)")
        for col, (name, a) in zip(st.columns(len(archs)), archs.items()):
            suffix = " · escolhida" if name == best_name else ""
            col.markdown(f"<span class='eyebrow'>{name.split(' (')[0]}{suffix}</span>", unsafe_allow_html=True)
            col.markdown(confusion_html(a["confusion_matrix_test"], palette), unsafe_allow_html=True)
        others = [a for n, a in archs.items() if n != best_name]
        last = {n: a["history"] for n, a in archs.items()}
        st.markdown(
            "**Análise**\n"
            f"- **Overfitting nas redes maiores:** as Arq. 2 e 3 atingem a menor perda de validação na época "
            f"{others[0]['best_epoch']}. Depois disso a perda de treino segue caindo "
            f"(até {br(min(h['train_loss'][-1] for n, h in last.items() if n != best_name), '.2f')}) enquanto a de "
            f"validação sobe (até {br(max(h['val_loss'][-1] for n, h in last.items() if n != best_name), '.2f')}): "
            "a rede passa a memorizar o treino.\n"
            f"- **Arq. 1 generaliza melhor:** as curvas se afastam devagar e a melhor época é a "
            f"{best['best_epoch']}, de {best['epochs_trained']} treinadas.\n"
            "- **Sem underfitting:** as três superam com folga a referência de classe majoritária e a perda de "
            "treino cai desde a primeira época.\n"
            "- **Early stopping:** restaurou em cada rede os pesos da melhor época, então o overfitting observado "
            "nas curvas não contamina os modelos avaliados.\n"
            "- **Erro dominante:** furtos classificados como roubo, efeito esperado do peso maior dado à classe "
            "minoritária."
        )

    with section("web", "APLICAÇÃO WEB"):
        shots = [(SCREENSHOTS_DIR / f"classificador_{t}.png", f"Classificador, tema {label}")
                 for t, label in [("escuro", "escuro"), ("claro", "claro")]]
        shots = [(path, caption) for path, caption in shots if path.exists()]
        if shots:
            for col, (path, caption) in zip(st.columns(len(shots)), shots):
                col.image(str(path), caption=caption, width="stretch")
        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Funcionalidades**\n"
            "- Formulário na barra lateral com as características da ocorrência (quando, onde e veículo).\n"
            "- Seccional, batalhão, companhia e coordenadas preenchidos automaticamente a partir da delegacia e "
            "do bairro.\n"
            "- Classe prevista em destaque e probabilidades de Roubo e Furto em percentual.\n"
            "- Limiar de decisão ajustável, com efeito imediato sobre a classe.\n"
            "- Aviso de interpretação, conferência dos dados enviados ao modelo, tema claro e escuro.\n"
            "- Documentação técnica e este relatório na própria aplicação."
        )
        col_b.markdown(
            "**Integração com a MLP**\n"
            "- Streamlit, executando o mesmo código de pré-processamento usado no treino (`src/data_prep.py`).\n"
            "- Carrega `best_mlp_model.pt` e recria a rede com `MLP.from_config`; carrega também o encoder, o "
            "scaler e os metadados salvos no treino.\n"
            "- A cada clique: uma linha de dados, One-Hot e padronização, tensor de 218 posições, rede em modo "
            "de avaliação, sigmoid e comparação com o limiar.\n"
            "- Verificação: com os artefatos salvos, a aplicação reproduz o F1 macro de teste do notebook."
        )

    with section("summarize", "CONCLUSÕES, LIMITAÇÕES E USO DE IA GENERATIVA"):
        tuned = experiments["threshold"]
        col_a, col_b = st.columns(2)
        col_a.markdown(
            "**Conclusões**\n"
            f"- Uma MLP de uma camada oculta atinge F1 macro de {br(best['test_metrics']['f1_macro'], '.3f')} no "
            f"teste, contra {br(experiments['majority_baseline_test']['f1_macro'], '.3f')} da referência.\n"
            "- Mais profundidade não ajudou: com 10.500 exemplos e entradas esparsas, as redes maiores "
            "memorizam o treino em poucas épocas.\n"
            f"- O limiar de decisão importa: {br(tuned['tuned'], '.2f')} eleva o F1 macro de teste para "
            f"{br(tuned['test_metrics_tuned']['f1_macro'], '.3f')}.\n"
            "- A etapa mais decisiva foi a prevenção de vazamento: o canal de registro do boletim denunciaria o "
            "tipo de crime e inflaria as métricas.\n\n"
            "**Limitações**\n"
            "- Não prevê risco de crime; classifica uma ocorrência que já existe.\n"
            "- Metade das previsões de roubo são furtos (precisão de cerca de 50% na classe Roubado).\n"
            "- As probabilidades não são calibradas, por causa do peso dado aos roubos.\n"
            "- Base bruta de boletins, sem consistência oficial, restrita à capital e a 2025; só crimes "
            "registrados.\n"
            "- Uma única divisão e uma única semente: a diferença entre as Arq. 1 e 2 é pequena e pode estar "
            "dentro da variação do treino.\n"
            "- A categoria Incerto do período só ocorre fora da Delegacia Eletrônica, um resíduo do canal de "
            "registro."
        )
        col_b.markdown(
            "**Como a IA generativa foi utilizada**\n"
            "- **Papel:** assistente de programação. O grupo definiu o problema, o recorte dos dados, as regras "
            "contra vazamento, as arquiteturas a comparar e os critérios de avaliação.\n"
            "- **Código em conjunto:** o grupo partiu de uma implementação própria anterior e trabalhou no "
            "código junto com a ferramenta, especificando cada etapa, revisando e ajustando o que era gerado. "
            "A IA acelerou a escrita do pipeline de dados, do laço de treino e da interface.\n"
            "- **Controle:** o trabalho foi dividido em etapas, e cada uma só avançou depois de revisão técnica "
            "e aprovação do grupo. Sugestões da ferramenta foram aceitas, ajustadas ou recusadas caso a caso.\n"
            "- **Documentação:** apoio na redação, com revisão do grupo.\n"
            "- **Validação:** todos os resultados vêm de código executado e reproduzível "
            "(`experiment_results.json`); nenhum número do relatório foi gerado por texto."
        )
