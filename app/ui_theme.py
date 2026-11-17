"""Tema claro/escuro, CSS e utilitários visuais da aplicação."""
import streamlit as st

THEMES = {
    "dark": {
        "config": {"base": "dark", "primaryColor": "#ED145B", "backgroundColor": "#0B0B0E",
                   "secondaryBackgroundColor": "#15151A", "textColor": "#F2F2F3"},
        "border": "#2A2A31", "muted": "#9A9AA3",
        "series": ["#3987E5", "#D95926"],  # treino, validação
        "heat": "57, 135, 229",
    },
    "light": {
        "config": {"base": "light", "primaryColor": "#D40F4F", "backgroundColor": "#FFFFFF",
                   "secondaryBackgroundColor": "#F3F3F6", "textColor": "#111114"},
        "border": "#D9D9E0", "muted": "#5C5C68",
        "series": ["#2A78D6", "#EB6834"],
        "heat": "42, 120, 214",
    },
}
THEME_LABELS = {"dark": ":material/dark_mode: Escuro", "light": ":material/light_mode: Claro"}


def br(value: float, spec: str) -> str:
    """Formata número com vírgula decimal (pt-BR)."""
    return format(value, spec).replace(".", ",")


def theme_selector() -> dict:
    """Seletor de tema na sidebar. Aplica as cores no Streamlit e devolve a paleta ativa.

    As opções de tema são do servidor, não da sessão: em uma implantação com vários
    usuários simultâneos, a troca feita por um deles vale para todos.
    """
    current = st.get_option("theme.base") or "dark"
    choice = st.segmented_control(
        "Tema", list(THEMES), default=current, format_func=THEME_LABELS.get, key="tema", width="stretch"
    ) or current
    if any(st.get_option(f"theme.{k}") != v for k, v in THEMES[choice]["config"].items()):
        for key, value in THEMES[choice]["config"].items():
            st._config.set_option(f"theme.{key}", value)
        st.rerun()
    theme = THEMES[choice]
    return {"name": choice, "primary": theme["config"]["primaryColor"],
            "surface": theme["config"]["secondaryBackgroundColor"], "text": theme["config"]["textColor"],
            "border": theme["border"], "muted": theme["muted"], "series": theme["series"], "heat": theme["heat"]}


def inject_css(p: dict) -> None:
    st.markdown(
        f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  #MainMenu, .stAppDeployButton, footer {{ display: none; }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  h1, h2, h3 {{ text-transform: uppercase; letter-spacing: .05em; font-weight: 800; }}
  h1 {{ font-style: italic; letter-spacing: .02em; padding-bottom: .2rem; }}
  h1::after {{ content: ""; display: block; width: 72px; height: 4px; margin-top: .6rem;
               background: {p['primary']}; transform: skewX(-20deg); }}

  section[data-testid="stSidebar"] {{ border-right: 1px solid {p['border']}; }}
  section[data-testid="stSidebar"] h2 {{ font-size: 1.05rem; }}
  section[data-testid="stSidebar"] h3 {{ font-size: .8rem; color: {p['muted']}; margin-top: .6rem;
                                         padding-top: .9rem; border-top: 1px solid {p['border']}; }}
  label[data-testid="stWidgetLabel"] p {{ font-size: .78rem; text-transform: uppercase;
                                          letter-spacing: .06em; color: {p['muted']}; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {p['border']}; }}
  .stTabs [data-baseweb="tab"] p {{ font-size: .85rem; font-weight: 700; letter-spacing: .1em;
                                    text-transform: uppercase; }}

  [data-testid="stBaseButton-primary"] {{
      border: 1px solid {p['primary']}; border-radius: 4px; min-height: 2.9rem;
      transition: box-shadow .2s ease, transform .15s ease, filter .2s ease; }}
  [data-testid="stBaseButton-primary"] p {{ font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }}
  [data-testid="stBaseButton-primary"]:hover {{
      filter: brightness(1.12); transform: translateY(-1px);
      box-shadow: 0 0 0 1px {p['primary']}, 0 0 22px {p['primary']}99; }}
  [data-testid="stBaseButton-primary"]:active {{ transform: translateY(0); box-shadow: 0 0 8px {p['primary']}99; }}
  [data-testid="stBaseButton-primary"]:focus-visible {{ outline: 2px solid {p['text']}; outline-offset: 2px; }}

  [data-testid="stMetric"] {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: 6px;
                              padding: .9rem 1.1rem; transition: border-color .2s ease; }}
  [data-testid="stMetric"]:hover {{ border-color: {p['primary']}; }}
  [data-testid="stMetricLabel"] p {{ font-size: .75rem; font-weight: 700; letter-spacing: .1em;
                                     text-transform: uppercase; color: {p['muted']}; }}
  [data-testid="stMetricValue"] {{ font-weight: 800; font-variant-numeric: tabular-nums; }}

  .hero {{ background: {p['surface']}; border: 1px solid {p['border']}; border-left: 4px solid {p['border']};
           border-radius: 6px; padding: 1.1rem 1.4rem; height: 100%; }}
  .hero.roubo {{ border-left-color: {p['primary']}; }}
  .hero .eyebrow, .eyebrow {{ font-size: .75rem; font-weight: 700; letter-spacing: .12em;
                              text-transform: uppercase; color: {p['muted']}; }}
  .hero .classe {{ font-size: 3.4rem; font-weight: 800; font-style: italic; line-height: 1.05;
                   text-transform: uppercase; letter-spacing: .02em; margin: .15rem 0 .35rem; }}
  .hero.roubo .classe {{ color: {p['primary']}; text-shadow: 0 0 24px {p['primary']}55; }}
  .hero .regra {{ font-size: .85rem; color: {p['muted']}; }}

  table.cm {{ border-collapse: separate; border-spacing: 2px; width: 100%; font-variant-numeric: tabular-nums; }}
  table.cm th {{ font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
                 color: {p['muted']}; padding: .3rem .4rem; text-align: center; border: 0; }}
  table.cm td {{ text-align: center; padding: .9rem .4rem; border: 0; border-radius: 4px; line-height: 1.25; }}
  table.cm td b {{ font-size: 1.25rem; }}
  table.cm td span {{ font-size: .8rem; }}

  @media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; }} }}
</style>
""",
        unsafe_allow_html=True,
    )
