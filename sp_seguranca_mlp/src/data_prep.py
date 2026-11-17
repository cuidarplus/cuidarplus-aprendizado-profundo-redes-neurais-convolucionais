"""Fase 1 - Pipeline de preparação de dados (Furtado x Roubado, capital de SP).

Uso (a partir da raiz sp_seguranca_mlp/):
    python src/data_prep.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "VeiculosSubtraidos_2025.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "sample_15k.csv"

CITY_COL = "NOME_MUNICIPIO_CIRC"
CAPITAL = "S.PAULO"
SOURCE_TARGET_COL = "DESCR_OCORRENCIA_VEICULO"
TARGET_COL = "ALVO"
TARGET_MAP = {"Furtado": 0, "Roubado": 1}
BO_KEY = ["NOME_DELEGACIA", "ANO_BO", "NUM_BO"]
SAMPLE_SIZE = 15_000
RANDOM_STATE = 42

# Colunas que descrevem a natureza jurídica do crime (revelam o alvo diretamente).
TARGET_LEAK_COLS = ["RUBRICA", "DESC_LEI", "DESDOBRAMENTO", "CIRCUNSTANCIA", "DESCR_CONDUTA"]
# Identificadores de BO / veículo / endereço exato.
ID_COLS = ["NUM_BO", "PLACA_VEICULO", "CONT_VEICULO", "LOGRADOURO", "NUMERO_LOGRADOURO"]
# Informação conhecida apenas após o fato (investigação / trâmite do registro).
POST_FACT_COLS = ["FLAG_FLAGRANTE", "AUTORIA_BO", "FLAG_STATUS", "DESCRICAO_APRESENTACAO", "VERSAO"]
# Datas de registro, comunicação e impressão do BO (não são do fato).
REGISTRATION_DATE_COLS = [
    "DATAHORA_REGISTRO_BO",
    "DATA_COMUNICACAO_BO",
    "DATAHORA_IMPRESSAO_BO",
    "MES_REGISTRO_BO",
    "ANO_REGISTRO_BO",
]
DROP_COLS = TARGET_LEAK_COLS + ID_COLS + POST_FACT_COLS + REGISTRATION_DATE_COLS


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NULL"], low_memory=False)


def filter_capital(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[CITY_COL] == CAPITAL]


def filter_target(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas Furtado/Roubado e cria o alvo binário (0/1)."""
    df = df[df[SOURCE_TARGET_COL].isin(TARGET_MAP)].copy()
    df[TARGET_COL] = df[SOURCE_TARGET_COL].map(TARGET_MAP).astype("int8")
    return df


def deduplicate_bo(df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por BO, para que o mesmo BO não apareça em treino e teste."""
    return df.drop_duplicates(subset=BO_KEY, keep="first")


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_COLS)


def stratified_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Amostra exatamente `n` linhas preservando a proporção do alvo."""
    if len(df) <= n:
        return df.reset_index(drop=True)
    sample, _ = train_test_split(df, train_size=n, stratify=df[TARGET_COL], random_state=RANDOM_STATE)
    return sample.reset_index(drop=True)


def run() -> pd.DataFrame:
    df = load_raw()
    print(f"Bruto:                        {df.shape}")
    df = filter_capital(df)
    print(f"Capital ({CAPITAL}):            {df.shape}")
    df = filter_target(df)
    print(f"Furtado/Roubado:              {df.shape}")
    df = deduplicate_bo(df)
    print(f"Deduplicado por BO:           {df.shape}")
    df = drop_leaky_columns(df)
    print(f"Sem vazamento/identificadores: {df.shape}")
    sample = stratified_sample(df)
    print(f"Amostra estratificada:        {sample.shape}")
    print("\nDistribuição do alvo (população -> amostra):")
    print(
        pd.DataFrame(
            {
                "populacao": df[TARGET_COL].value_counts(normalize=True),
                "amostra": sample[TARGET_COL].value_counts(normalize=True),
                "n_amostra": sample[TARGET_COL].value_counts(),
            }
        ).round(4)
    )
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(PROCESSED_PATH, index=False)
    print(f"\nSalvo em {PROCESSED_PATH.relative_to(ROOT)}")
    return sample


if __name__ == "__main__":
    run()


# =============================================================================
# Fase 2 - Engenharia de features, split e transformadores
# (reutilizado pelo notebook e pela aplicação Streamlit)
# =============================================================================
MODELS_DIR = ROOT / "models"

CATEGORICAL_COLS = [
    "NOME_SECCIONAL_CIRC",
    "NOME_DELEGACIA_CIRC",
    "BTL",
    "CIA",
    "PERIODO_FATO",
    "DESCR_TIPOLOCAL",
    "DESCR_SUBTIPOLOCAL",
    "BAIRRO",
    "DESCR_TIPO_VEICULO",
    "MARCA_VEICULO",
    "DESCR_COR_VEICULO",
]
NUMERIC_COLS = ["LATITUDE", "LONGITUDE", "IDADE_VEICULO"]
# Colunas agrupadas em Top-N + "OUTROS".
TOP_N_COLS = ["BAIRRO", "MARCA_VEICULO", "NOME_DELEGACIA_CIRC"]
TOP_N = 20
REFERENCE_YEAR = 2025  # ANO_BO da base; referência para a idade do veículo
OTHER = "OUTROS"

# A ausência de ANO_MODELO coincide quase perfeitamente com o registro na
# Delegacia Eletrônica (9.214 de 9.230 linhas na amostra), ou seja, uma flag de
# nulo reintroduz o vazamento administrativo removido. Por padrão a idade usa
# ANO_FABRICACAO (99,4% preenchido) e nenhuma flag é criada.
# True = especificação literal: idade só por ANO_MODELO + mediana + flag de nulo.
USE_ANO_MODELO_FLAG = False
FLAG_COL = "ANO_MODELO_NULO"

PERIODO_FROM_DESCR = {
    "De madrugada": "Madrugada",
    "Pela manhã": "Manhã",
    "A tarde": "Tarde",
    "A noite": "Noite",
}
PERIODO_BINS = ["Madrugada", "Manhã", "Tarde", "Noite"]  # 0-5, 6-11, 12-17, 18-23
UNCERTAIN = "Incerto"

BRAND_ALIASES = {
    "CHEV": "CHEVROLET",
    "GM": "CHEVROLET",
    "VOLKSWAGEN": "VW",
    "VOLKSWAGEM": "VW",
    "VOLKWAGEN": "VW",
    "MMC": "MITSUBISHI",
    "M.BENZ": "MERCEDES",
    "MERCEDES-BENZ": "MERCEDES",
    "RE": "ROYAL ENFIELD",
}
# Limites aproximados do município; fora disso a coordenada é inválida (ex.: 0,0).
LAT_RANGE = (-24.1, -23.3)
LON_RANGE = (-46.9, -46.3)


def passthrough_cols() -> list[str]:
    return [FLAG_COL] if USE_ANO_MODELO_FLAG else []


def derive_periodo(hora: pd.Series, descr_periodo: pd.Series) -> pd.Series:
    """Consolida HORA_OCORRENCIA e DESCR_PERIODO em um único período do dia."""
    hour = pd.to_numeric(hora.astype("string").str.slice(0, 2), errors="coerce")
    from_hour = pd.cut(hour, bins=[0, 6, 12, 18, 24], right=False, labels=PERIODO_BINS).astype("object")
    return from_hour.fillna(descr_periodo.map(PERIODO_FROM_DESCR)).fillna(UNCERTAIN)


def extract_brand(model: pd.Series) -> pd.Series:
    """'I/VW JETTA 2.0T' -> 'VW'; 'GM/ONIX' -> 'CHEVROLET'; 'CITROEN PICASSO' -> 'CITROEN'."""
    s = model.astype("string").str.upper().str.strip()
    s = s.str.replace(r"^(I|IMP|H)/", "", regex=True)  # prefixo de importado
    s = s.str.split("/").str[0].str.strip()
    known_two_words = s.str.startswith("ROYAL ENFIELD", na=False)
    s = s.where(known_two_words, s.str.split().str[0])
    return s.replace(BRAND_ALIASES).fillna(OTHER).astype("object")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transformações sem estado (linha a linha). Mantém apenas features + ALVO."""
    out = pd.DataFrame(index=df.index)
    for col in ["NOME_SECCIONAL_CIRC", "NOME_DELEGACIA_CIRC", "BTL", "CIA",
                "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "DESCR_TIPO_VEICULO", "DESCR_COR_VEICULO"]:
        out[col] = df[col].astype("object")
    out["BAIRRO"] = df["BAIRRO"].astype("string").str.upper().str.strip().astype("object")
    out["PERIODO_FATO"] = derive_periodo(df["HORA_OCORRENCIA"], df["DESCR_PERIODO"])
    out["MARCA_VEICULO"] = extract_brand(df["DESCR_MARCA_VEICULO"])

    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    valid = lat.between(*LAT_RANGE) & lon.between(*LON_RANGE)
    out["LATITUDE"] = lat.where(valid)
    out["LONGITUDE"] = lon.where(valid)

    ano_modelo = pd.to_numeric(df["ANO_MODELO"], errors="coerce")
    if USE_ANO_MODELO_FLAG:
        out[FLAG_COL] = ano_modelo.isna().astype("float32")
        ano_veiculo = ano_modelo
    else:
        ano_veiculo = pd.to_numeric(df["ANO_FABRICACAO"], errors="coerce").fillna(ano_modelo)
    out["IDADE_VEICULO"] = (df["ANO_BO"] - ano_veiculo).clip(lower=0)

    if TARGET_COL in df.columns:
        out[TARGET_COL] = df[TARGET_COL].astype("int64")
    return out


def split_data(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15):
    """Treino/validação/teste estratificados pelo alvo (70/15/15 por padrão)."""
    train, tmp = train_test_split(
        df, test_size=val_size + test_size, stratify=df[TARGET_COL], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        tmp, test_size=test_size / (val_size + test_size), stratify=tmp[TARGET_COL], random_state=RANDOM_STATE
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fit_preprocessor(train: pd.DataFrame) -> dict:
    """Aprende, SOMENTE no treino, as categorias Top-N e as medianas de imputação."""
    top = {c: train[c].value_counts().head(TOP_N).index.tolist() for c in TOP_N_COLS}
    bairro = train["BAIRRO"].where(train["BAIRRO"].isin(top["BAIRRO"]))
    coords = train[["LATITUDE", "LONGITUDE"]]
    # Tabelas de apoio para a aplicação: a partir da delegacia de circunscrição
    # derivam-se seccional, batalhão, companhia e uma coordenada típica.
    mode = lambda x: x.mode().iat[0]  # noqa: E731
    circ = train.groupby("NOME_DELEGACIA_CIRC").agg(
        NOME_SECCIONAL_CIRC=("NOME_SECCIONAL_CIRC", mode), BTL=("BTL", mode), CIA=("CIA", mode),
        LATITUDE=("LATITUDE", "median"), LONGITUDE=("LONGITUDE", "median"),
    )
    circ[["LATITUDE", "LONGITUDE"]] = circ[["LATITUDE", "LONGITUDE"]].fillna(coords.median())
    return {
        "circ_lookup": circ.to_dict("index"),
        "bairro_default_delegacia": train.groupby(bairro)["NOME_DELEGACIA_CIRC"].agg(mode).to_dict(),
        "subtipo_by_tipo": train.groupby("DESCR_TIPOLOCAL")["DESCR_SUBTIPOLOCAL"]
        .agg(lambda x: x.value_counts().index.tolist()).to_dict(),
        "reference_year": REFERENCE_YEAR,
        "top_categories": top,
        "coord_median_by_bairro": coords.groupby(bairro).median().dropna().to_dict("index"),
        "coord_median_city": coords.median().to_dict(),
        "idade_median": float(train["IDADE_VEICULO"].median()),
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "passthrough_cols": passthrough_cols(),
    }


def apply_preprocessor(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Agrupa categorias raras em OUTROS e imputa nulos com as estatísticas do treino."""
    df = df.copy()
    for col, keep in meta["top_categories"].items():
        df[col] = df[col].where(df[col].isin(keep), OTHER)
    for col in meta["categorical_cols"]:
        df[col] = df[col].fillna(OTHER)
    by_bairro = meta["coord_median_by_bairro"]
    for col in ["LATITUDE", "LONGITUDE"]:
        bairro_median = df["BAIRRO"].map({b: v[col] for b, v in by_bairro.items()})
        df[col] = df[col].fillna(bairro_median).fillna(meta["coord_median_city"][col])
    df["IDADE_VEICULO"] = df["IDADE_VEICULO"].fillna(meta["idade_median"])
    return df


def fit_transformers(train: pd.DataFrame, meta: dict):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(train[meta["categorical_cols"]])
    scaler = StandardScaler().fit(train[meta["numeric_cols"]])
    return encoder, scaler


def transform_features(df: pd.DataFrame, meta: dict, encoder, scaler) -> np.ndarray:
    """Matriz final: [numéricas escalonadas | flags | one-hot]."""
    parts = [scaler.transform(df[meta["numeric_cols"]])]
    if meta["passthrough_cols"]:
        parts.append(df[meta["passthrough_cols"]].to_numpy())
    parts.append(encoder.transform(df[meta["categorical_cols"]]))
    return np.hstack(parts).astype(np.float32)


def feature_names(meta: dict, encoder) -> list[str]:
    return (
        list(meta["numeric_cols"])
        + list(meta["passthrough_cols"])
        + encoder.get_feature_names_out(meta["categorical_cols"]).tolist()
    )


def save_artifacts(encoder, scaler, meta: dict, models_dir: Path = MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, models_dir / "encoder.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(meta, models_dir / "preprocess_meta.pkl")


def load_artifacts(models_dir: Path = MODELS_DIR):
    return (
        joblib.load(models_dir / "encoder.pkl"),
        joblib.load(models_dir / "scaler.pkl"),
        joblib.load(models_dir / "preprocess_meta.pkl"),
    )
