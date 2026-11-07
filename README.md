# Detección de Fraude Financiero con Machine Learning

Proyecto final de **Minería de Datos**. Sistema inteligente para detectar transacciones fraudulentas con tarjeta de crédito usando machine learning y aprendizaje profundo, desarrollado con la metodología **CRISP-DM**.

## Dataset

**Credit Card Fraud Detection** — Kaggle ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).

| Característica | Valor |
|---|---|
| Transacciones | 284.807 (tarjetas europeas, septiembre de 2013, 2 días) |
| Fraudes | 492 (~0,172 %) |
| Variables | `Time`, `V1`–`V28` (componentes PCA anonimizados), `Amount` |
| Objetivo | `Class` (1 = fraude, 0 = normal) |

Descarga manual: bajar `creditcard.csv` desde Kaggle y colocarlo en `data/raw/creditcard.csv`.

Descarga con la CLI de Kaggle (requiere `kaggle.json` configurado):

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Técnicas implementadas

| Requerimiento | Implementación |
|---|---|
| Balanceo de clases | SMOTE (`imbalanced-learn`) dentro de `imblearn.Pipeline`, solo sobre entrenamiento. Alternativa: `class_weight` / `scale_pos_weight` |
| Bagging | Random Forest |
| Boosting | XGBoost |
| Red neuronal profunda | DNN con TensorFlow/Keras |
| Detección de anomalías | Autoencoder entrenado solo con transacciones normales; umbral sobre error de reconstrucción |
| Métricas | Precision, Recall, F1, ROC-AUC, PR-AUC (Average Precision), matriz de confusión |
| Interpretabilidad | SHAP sobre XGBoost |
| Umbral de decisión | Optimizado por modelo según costo de negocio (FN más caro que FP) |

## Estructura del proyecto

```
fraud-detection/
├── data/
│   ├── raw/                 # creditcard.csv original (no versionado)
│   ├── processed/           # train/val/test sin escalar (Parquet, no versionado)
│   └── demo/                # muestra liviana de prueba para la app (versionada)
├── notebooks/
│   ├── 01_comprension_negocio_y_datos.ipynb   # CRISP-DM fases 1 y 2 (EDA)
│   ├── 02_preparacion_datos.ipynb             # fase 3
│   ├── 03_modelado.ipynb                      # fase 4
│   ├── 04_evaluacion.ipynb                    # fase 5
│   └── referencia/                            # notebook monolítico original (base del proyecto)
├── src/
│   ├── config.py               # rutas, semillas, esquema del dataset y colores
│   ├── data_preprocessing.py   # carga, limpieza, escalado, split estratificado
│   ├── train_models.py         # entrenamiento de RF, XGBoost, DNN y Autoencoder
│   ├── evaluate.py             # métricas, curvas, umbrales, SHAP
│   └── predict.py              # inferencia con modelos guardados
├── models/                  # modelos, preprocesador, registro, umbrales y puntajes
├── app/
│   ├── app.py               # aplicación Streamlit (navegación entre páginas)
│   ├── paginas/             # inicio, dashboard, monitoreo, sospechosas, predicción
│   ├── utils.py             # caché de modelos/datos y estilo de gráficos
│   ├── api.py               # API REST (FastAPI)
│   └── requirements.txt     # dependencias de ejecución (Docker / Streamlit Cloud)
├── tests/                   # pruebas del predictor, la API y la app
├── reports/figures/         # figuras para el documento técnico
├── docs/                    # resúmenes por fase, DESPLIEGUE.md y artículo IEEE
├── Dockerfile · docker-compose.yml
├── requirements.txt         # entorno completo de desarrollo
└── README.md
```

## Instalación

Requisito: **Python 3.11** (TensorFlow 2.18 no publica versiones para Python 3.13 o superior).

```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name fraud-detection --display-name "Python 3.11 (fraud-detection)"
```

```bash
# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Los notebooks de `notebooks/` recorren las fases con explicaciones y figuras. El mismo pipeline se reproduce desde la terminal:

```bash
# 1. Preparación (limpieza, variables, división estratificada, escalado)
python -m src.data_preprocessing

# 2. Entrenamiento de todos los modelos (~30 min con 12 núcleos)
python -m src.train_models

# 3. Evaluación: umbrales, selección del mejor modelo y models/config_despliegue.json
python -m src.evaluate

# 4. Aplicación web  ->  http://localhost:8501
streamlit run app/app.py --server.address localhost

# 5. API REST  ->  http://localhost:8000/docs
uvicorn app.api:app --port 8000

# 6. Pruebas
python -m pytest

# 7. Docker (app + API)
docker compose up --build
```

Detalles de arquitectura, monitoreo en tiempo real, Streamlit Community Cloud y plan de mantenimiento: [docs/DESPLIEGUE.md](docs/DESPLIEGUE.md).

## Resultados (conjunto de prueba, 95 fraudes)

| Modelo | Precision | Recall | F1 | PR-AUC | Ahorro vs. sin modelo |
|---|---|---|---|---|---|
| **XGBoost (seleccionado)** | **0,830** | **0,821** | **0,825** | **0,829** | **71,2 %** |
| Stacking (RF + XGB) | 0,750 | 0,821 | 0,784 | 0,817 | 70,9 % |
| Random Forest + SMOTE | 0,745 | 0,800 | 0,772 | 0,813 | 70,9 % |
| DNN | 0,811 | 0,768 | 0,789 | 0,780 | 69,5 % |
| Regresión Logística + SMOTE | 0,787 | 0,779 | 0,783 | 0,695 | 71,0 % |
| Autoencoder | 0,735 | 0,263 | 0,388 | 0,493 | 9,3 % |

Umbral de decisión de cada modelo elegido en validación (mínimo costo con Precision ≥ 0,70; 5 € de revisión por alerta).

## Metodología CRISP-DM

| Fase | Entregable |
|---|---|
| 1. Comprensión del negocio | Problema, costo de los errores y criterios de éxito (Recall ≥ 0,80, Precision ≥ 0,70, PR-AUC ≥ 0,80) |
| 2. Comprensión de los datos | EDA: calidad, desbalance, `Amount`, `Time`, correlaciones, separabilidad (KS) y atípicos |
| 3. Preparación de los datos | Duplicados, `Amount_log`, hora cíclica, división 60/20/20, escalado y SMOTE solo en train, experimento de fuga |
| 4. Modelado | Regresión Logística, RF, XGBoost, Stacking, DNN y Autoencoder; con y sin balanceo; `RandomizedSearchCV` |
| 5. Evaluación | Umbral por costo de negocio, métricas con IC 95 %, curvas ROC y PR, matrices de confusión, SHAP |
| 6. Despliegue | App Streamlit (dashboard, alertas en tiempo real, sospechosas, predicción), API FastAPI, Docker |

Resúmenes de cada fase para el documento técnico: `docs/resumen_fases_1_2.md`, `docs/resumen_fase_3.md` … `docs/resumen_fase_6.md`.

## Reproducibilidad

Todas las fuentes de aleatoriedad usan `random_state=42` (NumPy, scikit-learn, XGBoost, TensorFlow).

## Referencia del dataset

A. Dal Pozzolo, O. Caelen, R. A. Johnson y G. Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification," en *IEEE Symposium Series on Computational Intelligence (SSCI)*, 2015.
