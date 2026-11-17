"""Aplicação Streamlit: classifica uma ocorrência hipotética de veículo como Roubo ou Furto.

Uso (a partir da raiz do repositório):
    streamlit run app/main.py
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

import report  # noqa: E402
from src import data_prep as dp  # noqa: E402
from src.architecture import MLP  # noqa: E402
from ui_theme import br, inject_css, theme_selector  # noqa: E402

PERIODO_LABELS = {
    "Madrugada": "Madrugada (0h às 5h59)",
    "Manhã": "Manhã (6h às 11h59)",
    "Tarde": "Tarde (12h às 17h59)",
    "Noite": "Noite (18h às 23h59)",
    "Incerto": "Hora incerta",
}

st.set_page_config(page_title="Roubo vs Furto de Veículos", page_icon=":material/directions_car:", layout="wide")

@st.cache_resource
def load_pipeline():
    encoder, scaler, meta = dp.load_artifacts()
    checkpoint = torch.load(dp.MODELS_DIR / "best_mlp_model.pt", map_location="cpu")
    model = MLP.from_config(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    experiments = json.loads((dp.MODELS_DIR / "experiment_results.json").read_text())
    return encoder, scaler, meta, model, checkpoint, experiments


def options_for(column: str, encoder, meta: dict) -> list[str]:
    """Categorias que o encoder conhece para a coluna, com OUTROS por último."""
    cats = list(encoder.categories_[meta["categorical_cols"].index(column)])
    return sorted(c for c in cats if c != dp.OTHER) + ([dp.OTHER] if dp.OTHER in cats else [])


def predict_proba(row: dict, encoder, scaler, meta: dict, model: MLP) -> float:
    """a) DataFrame de 1 linha -> b) encoder + scaler -> c) tensor + modelo -> d) sigmoid."""
    df = dp.apply_preprocessor(pd.DataFrame([row]), meta)
    x = torch.from_numpy(dp.transform_features(df, meta, encoder, scaler))
    model.eval()
    with torch.no_grad():
        logit = model(x)
    return float(torch.sigmoid(logit).item())


encoder, scaler, meta, model, checkpoint, experiments = load_pipeline()
circ_lookup = meta["circ_lookup"]
top_bairros = meta["top_categories"]["BAIRRO"]
tuned_threshold = checkpoint["threshold_tuned"]

# --------------------------------------------------------------------------- #
# Sidebar: características da ocorrência e parâmetros do modelo
# --------------------------------------------------------------------------- #
with st.sidebar:
    palette = theme_selector()
    inject_css(palette)
    st.header(":material/fact_check: OCORRÊNCIA")

    st.subheader(":material/schedule: QUANDO")
    periodo = st.selectbox("Período do fato", list(PERIODO_LABELS), index=3, format_func=PERIODO_LABELS.get)

    st.subheader(":material/location_on: ONDE")
    bairro = st.selectbox(
        "Bairro", sorted(top_bairros) + [dp.OTHER],
        help="Os 20 bairros com mais ocorrências no treino; os demais entram como OUTROS.",
    )
    delegacias = sorted(circ_lookup)
    default_delegacia = meta["bairro_default_delegacia"].get(bairro, delegacias[0])
    delegacia = st.selectbox(
        "Delegacia de circunscrição", delegacias, index=delegacias.index(default_delegacia),
        help="Delegacia responsável pela área do fato. A sugestão inicial é a mais frequente para o bairro "
             "escolhido. Seccional, batalhão e companhia da PM são derivados dela.",
    )
    tipos = options_for("DESCR_TIPOLOCAL", encoder, meta)
    tipo_local = st.selectbox("Tipo de local", tipos, index=tipos.index("Via Pública"))
    subtipo_local = st.selectbox("Subtipo de local", meta["subtipo_by_tipo"][tipo_local])

    circ = circ_lookup[delegacia]
    if bairro in meta["coord_median_by_bairro"]:
        auto_coords, coord_source = meta["coord_median_by_bairro"][bairro], f"mediana do bairro {bairro}"
    else:
        auto_coords, coord_source = circ, f"mediana da área da delegacia {delegacia}"
    manual_coords = st.checkbox("Informar coordenadas manualmente")
    if manual_coords:
        latitude = st.number_input("Latitude", *dp.LAT_RANGE, float(auto_coords["LATITUDE"]), format="%.6f")
        longitude = st.number_input("Longitude", *dp.LON_RANGE, float(auto_coords["LONGITUDE"]), format="%.6f")
        coord_source = "informada manualmente"
    else:
        latitude, longitude = float(auto_coords["LATITUDE"]), float(auto_coords["LONGITUDE"])
        st.caption(f":material/my_location: {latitude:.4f}, {longitude:.4f} ({coord_source})")

    st.subheader(":material/directions_car: VEÍCULO")
    tipos_veiculo = options_for("DESCR_TIPO_VEICULO", encoder, meta)
    tipo_veiculo = st.selectbox("Tipo de veículo", tipos_veiculo, index=tipos_veiculo.index("Automovel"))
    marcas = options_for("MARCA_VEICULO", encoder, meta)
    marca = st.selectbox(
        "Marca", marcas, index=marcas.index("FIAT"),
        help="As 20 marcas mais frequentes no treino; as demais entram como OUTROS.",
    )
    cores = options_for("DESCR_COR_VEICULO", encoder, meta)
    cor = st.selectbox("Cor", cores, index=cores.index("Prata"))
    ano_fabricacao = st.number_input("Ano de fabricação", 1950, meta["reference_year"], 2019, step=1)

    st.header(":material/tune: PARÂMETROS DO MODELO")
    threshold = st.slider("Limiar de decisão", 0.0, 1.0, 0.5, 0.01)
    st.caption(
        f"Probabilidade de roubo a partir da qual a ocorrência é classificada como Roubo. "
        f"O padrão é 0,50; o valor {br(tuned_threshold, ".2f")} maximiza o F1-Score macro na validação."
    )
    clicked = st.button("Classificar Ocorrência", type="primary", icon=":material/bolt:", width="stretch")

row = {
    "NOME_SECCIONAL_CIRC": circ["NOME_SECCIONAL_CIRC"],
    "NOME_DELEGACIA_CIRC": delegacia,
    "BTL": circ["BTL"],
    "CIA": circ["CIA"],
    "PERIODO_FATO": periodo,
    "DESCR_TIPOLOCAL": tipo_local,
    "DESCR_SUBTIPOLOCAL": subtipo_local,
    "BAIRRO": bairro,
    "DESCR_TIPO_VEICULO": tipo_veiculo,
    "MARCA_VEICULO": marca,
    "DESCR_COR_VEICULO": cor,
    "LATITUDE": latitude,
    "LONGITUDE": longitude,
    "IDADE_VEICULO": float(meta["reference_year"] - ano_fabricacao),
}

# --------------------------------------------------------------------------- #
# Página principal
# --------------------------------------------------------------------------- #
st.title("Classificador de Ocorrências: Roubo vs Furto de Veículos")
st.caption("Rede neural MLP treinada com boletins de ocorrência de veículos subtraídos na cidade de São Paulo em 2025.")

tab_classificador, tab_docs, tab_relatorio = st.tabs([
    ":material/analytics: CLASSIFICADOR",
    ":material/description: DOCUMENTAÇÃO TÉCNICA",
    ":material/slideshow: RELATÓRIO TÉCNICO",
])

with tab_classificador:
    st.warning(
        "**Como interpretar o resultado.** Este modelo **não prevê a chance de um crime acontecer** em um local, "
        "horário ou com um veículo. Ele parte do pressuposto de que a ocorrência já existe e indica apenas se uma "
        "**ocorrência hipotética** com essas características tem um perfil mais compatível com **Roubo** "
        "(subtração com violência ou grave ameaça) ou com **Furto** (sem violência), com base nos registros de "
        "2025. É um exercício acadêmico e não deve orientar decisões de segurança pessoal ou policial.",
        icon=":material/warning:",
    )

    if clicked:
        st.session_state["result"] = {"row": row, "prob": predict_proba(row, encoder, scaler, meta, model)}

    result = st.session_state.get("result")
    if result is None:
        st.info("Preencha as características no menu lateral e clique em **Classificar Ocorrência**.",
                icon=":material/arrow_back:")
    elif result["row"] != row:
        st.info("As características foram alteradas. Clique em **Classificar Ocorrência** para atualizar o "
                "resultado.", icon=":material/sync:")
    else:
        p_roubo = result["prob"]
        p_furto = 1 - p_roubo
        classe = "Roubo" if p_roubo >= threshold else "Furto"
        comparacao = "maior ou igual ao" if p_roubo >= threshold else "menor que o"

        st.subheader(":material/speed: RESULTADO")
        col_classe, col_roubo, col_furto = st.columns([2, 1, 1])
        col_classe.markdown(
            f"<div class='hero {classe.lower()}'>"
            f"<div class='eyebrow'>Perfil mais compatível com</div>"
            f"<div class='classe'>{classe}</div>"
            f"<div class='regra'>Probabilidade de roubo ({br(p_roubo, ".1%")}) {comparacao} limiar de decisão "
            f"({br(threshold, ".2f")}).</div></div>",
            unsafe_allow_html=True,
        )
        col_roubo.metric(":material/warning: Roubo", f"{br(p_roubo, ".1%")}")
        col_roubo.progress(p_roubo)
        col_furto.metric(":material/lock_open: Furto", f"{br(p_furto, ".1%")}")
        col_furto.progress(p_furto)

        st.subheader(":material/table_rows: DADOS ENVIADOS AO MODELO")
        grupos = dp.apply_preprocessor(pd.DataFrame([row]), meta).iloc[0]
        resumo = pd.DataFrame(
            [
                ("Período do fato", PERIODO_LABELS[periodo]),
                ("Bairro", bairro),
                ("Delegacia de circunscrição", delegacia
                 + (" (agrupada em OUTROS pelo modelo)" if grupos["NOME_DELEGACIA_CIRC"] == dp.OTHER else "")),
                ("Seccional (derivada)", row["NOME_SECCIONAL_CIRC"]),
                ("Batalhão / Companhia PM (derivados)", f"{row['BTL']} / {row['CIA']}"),
                ("Tipo de local", tipo_local),
                ("Subtipo de local", subtipo_local),
                ("Latitude, Longitude", f"{latitude:.5f}, {longitude:.5f} ({coord_source})"),
                ("Tipo de veículo", tipo_veiculo),
                ("Marca", marca),
                ("Cor", cor),
                ("Ano de fabricação", f"{ano_fabricacao} (idade: {int(row['IDADE_VEICULO'])} anos)"),
            ],
            columns=["Característica", "Valor"],
        )
        st.dataframe(resumo, hide_index=True, width="stretch")

with tab_docs:
    report.render_docs(meta, encoder, model, checkpoint, experiments)

with tab_relatorio:
    report.render_report(meta, encoder, model, checkpoint, experiments, palette)
