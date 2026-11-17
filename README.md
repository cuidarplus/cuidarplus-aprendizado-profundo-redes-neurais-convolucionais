# SP Segurança – Classificação Roubo vs Furto de Veículos com MLP

Rede neural Multilayer Perceptron (PyTorch) que classifica uma ocorrência de veículo subtraído na cidade de São Paulo como **Furtado (0)** ou **Roubado (1)**, a partir da base "Veículos subtraídos 2025" da SSP-SP, com aplicação web em Streamlit.

## Estrutura

```
data/raw/            dataset original (não versionado)
data/processed/      sample_15k.csv – capital, deduplicado por boletim, amostra estratificada
notebooks/           01_modelagem_mlp.ipynb – features, split, treino de 3 MLPs, avaliação, exportação
src/data_prep.py     pipeline de dados, engenharia de features, encoder e scaler
src/architecture.py  classe MLP parametrizável
models/              best_mlp_model.pt, encoder.pkl, scaler.pkl, preprocess_meta.pkl, experiment_results.json
app/main.py          aplicação Streamlit (classificador)
app/report.py        abas de documentação técnica e relatório técnico
app/ui_theme.py      temas claro/escuro e CSS
.streamlit/          tema inicial da aplicação
reports/screenshots/ imagens da interface usadas no relatório
```

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Coloque `VeiculosSubtraidos_2025.csv` em `data/raw/`.
2. Gere a amostra: `python src/data_prep.py`
3. Treine e exporte os artefatos: `jupyter nbconvert --to notebook --execute --inplace notebooks/01_modelagem_mlp.ipynb`
4. Rode o app a partir desta pasta: `streamlit run app/main.py`

Os passos 2 e 3 são determinísticos (semente 42). O app precisa ser iniciado a partir da raiz do repositório para encontrar `.streamlit/config.toml`.

## Resultado

Modelo escolhido: uma camada oculta de 64 neurônios, com F1 macro de 0,751 no teste (referência de classe majoritária: 0,455). Detalhes nas abas **Documentação Técnica** e **Relatório Técnico** do app.

## App publicado

Link: https://seguranca-sp-ml-global-solution-2026-jose-danilo.streamlit.app/