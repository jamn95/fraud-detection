"""
Carga y preparación de datos (CRISP-DM fases 2 y 3).

Las funciones de este módulo se reutilizan en los notebooks, en el entrenamiento
y en la app web, para que el preprocesamiento sea idéntico en todos los casos.

Ejecución completa de la preparación desde la raíz del proyecto:
    python -m src.data_preprocessing
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config import (
    COLUMNAS_CRUDAS, COLUMNAS_V, DATA_PROCESSED, DATA_RAW, FEATURES, MODELS_DIR,
    RANDOM_STATE, TARGET, TEST_SIZE, VAL_SIZE,
)

RUTA_PREPROCESADOR = MODELS_DIR / "preprocesador.joblib"
RUTA_METADATA = DATA_PROCESSED / "metadata_preparacion.json"


# ==========================================================================
# Carga
# ==========================================================================
def cargar_datos(ruta: Path = DATA_RAW) -> pd.DataFrame:
    """Lee creditcard.csv y comprueba que tenga las columnas esperadas."""
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descarga creditcard.csv desde "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud y colócalo en data/raw/."
        )
    df = pd.read_csv(ruta)
    faltantes = set(COLUMNAS_CRUDAS + [TARGET]) - set(df.columns)
    if faltantes:
        raise ValueError(f"Al dataset le faltan columnas: {sorted(faltantes)}")
    return df


# ==========================================================================
# Limpieza
# ==========================================================================
def limpiar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Elimina duplicados exactos ANTES de dividir: si una transacción repetida
    quedara a la vez en entrenamiento y en prueba, el modelo "memorizaría" la
    respuesta y las métricas saldrían infladas.

    Se conserva el índice original (número de fila en creditcard.csv) como
    identificador de la transacción, para poder rastrearla en el dashboard.
    """
    duplicados = df.duplicated(keep="first")
    limpio = df.loc[~duplicados].copy()
    limpio.index.name = "id_transaccion"
    info = {
        "filas_originales": int(len(df)),
        "duplicados_eliminados": int(duplicados.sum()),
        "fraudes_duplicados_eliminados": int(df.loc[duplicados, TARGET].sum()),
        "filas_finales": int(len(limpio)),
        "fraudes_finales": int(limpio[TARGET].sum()),
        "nulos": int(limpio.isnull().sum().sum()),
    }
    return limpio, info


# ==========================================================================
# Feature engineering
# ==========================================================================
def hora_del_ciclo(time_segundos: pd.Series) -> pd.Series:
    """
    Convierte Time (segundos desde la primera transacción) en una hora 0-23.

    El dataset no trae la hora real del reloj, solo segundos relativos al inicio
    del registro. Se asume que el registro empieza a medianoche; el patrón diario
    (menos transacciones de madrugada) es consistente con ese supuesto.
    """
    return (time_segundos // 3600) % 24


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering. Agrega:
      - Hour_sin, Hour_cos: hora en forma cíclica, para que la hora 23 quede
        "cerca" de la hora 0 (con la hora lineal parecerían extremos opuestos).
      - Amount_log: log(1 + Amount), reduce la fuerte asimetría del monto.
    """
    df = df.copy()
    hora = hora_del_ciclo(df["Time"])
    df["Hour_sin"] = np.sin(2 * np.pi * hora / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * hora / 24)
    df["Amount_log"] = np.log1p(df["Amount"])
    return df


def preparar_para_prediccion(df_crudo: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte transacciones en formato original (Time, V1..V28, Amount) en la
    matriz de variables que esperan los modelos. Se usa en producción (app/API).
    El escalado lo hace el preprocesador guardado, no esta función.
    """
    faltantes = set(COLUMNAS_CRUDAS) - set(df_crudo.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas en la transacción: {sorted(faltantes)}")
    return crear_variables(df_crudo[COLUMNAS_CRUDAS])[FEATURES]


# ==========================================================================
# División
# ==========================================================================
def dividir_datos(df: pd.DataFrame, test_size: float = TEST_SIZE, val_size: float = VAL_SIZE,
                  random_state: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """
    División ESTRATIFICADA en entrenamiento / validación / prueba (60/20/20).

      - Entrenamiento: ajustar escaladores, SMOTE y modelos.
      - Validación: early stopping y elección del umbral de decisión.
      - Prueba: solo para la evaluación final. Conserva la proporción real de fraude.

    Se devuelven los DataFrames completos (todas las columnas), porque la
    evaluación de costo y el monitoreo en tiempo real necesitan Amount y Time.
    """
    temporal, prueba = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    val_relativo = val_size / (1 - test_size)   # 0.20 / 0.80 = 0.25 del temporal
    entrenamiento, validacion = train_test_split(
        temporal, test_size=val_relativo, stratify=temporal[TARGET], random_state=random_state
    )
    return {"train": entrenamiento, "val": validacion, "test": prueba}


def separar_xy(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa la matriz de variables (FEATURES) y la etiqueta (Class)."""
    return datos[FEATURES], datos[TARGET]


# ==========================================================================
# Escalado
# ==========================================================================
def construir_preprocesador() -> ColumnTransformer:
    """
    Escalador por tipo de variable (se ajusta SOLO con entrenamiento):
      - V1..V28 -> StandardScaler: ya están centradas pero con varianzas distintas;
        igualarlas ayuda a las redes neuronales. A los árboles no les afecta.
      - Amount_log -> RobustScaler: usa mediana e IQR, así los montos extremos
        no deforman la escala.
      - Hour_sin / Hour_cos -> sin cambios: ya están en [-1, 1].
    La salida conserva el orden y los nombres de FEATURES (DataFrame de pandas).
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ("estandar", StandardScaler(), COLUMNAS_V),
            ("robusto", RobustScaler(), ["Amount_log"]),
            ("ciclicas", "passthrough", ["Hour_sin", "Hour_cos"]),
        ],
        verbose_feature_names_out=False,
    )
    return preprocesador.set_output(transform="pandas")


# ==========================================================================
# Balanceo de clases
# ==========================================================================
def crear_smote(sampling_strategy: float = 0.1, k_neighbors: int = 5) -> SMOTE:
    """
    SMOTE genera fraudes sintéticos interpolando entre un fraude real y uno de
    sus k vecinos fraudulentos más cercanos.
    sampling_strategy = fraudes / legítimas después del balanceo
    (0.1 -> 1 fraude por cada 10 legítimas; 1.0 -> 50/50).
    """
    return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors,
                 random_state=RANDOM_STATE)


def pipeline_con_balanceo(modelo, sampling_strategy: float | None = None) -> ImbPipeline:
    """
    Pipeline preprocesador -> [SMOTE] -> modelo.

    Con imblearn.Pipeline, SMOTE solo se ejecuta en fit(), es decir, sobre los
    datos de entrenamiento de cada pliegue de la validación cruzada. En predict()
    se omite, así validación y prueba se evalúan con su distribución real.
    Si sampling_strategy es None no se aplica SMOTE (útil para comparar con
    class_weight / scale_pos_weight).
    """
    pasos = [("prep", construir_preprocesador())]
    if sampling_strategy is not None:
        pasos.append(("smote", crear_smote(sampling_strategy)))
    pasos.append(("modelo", modelo))
    return ImbPipeline(pasos)


# ==========================================================================
# Persistencia
# ==========================================================================
def guardar_splits(splits: dict[str, pd.DataFrame], carpeta: Path = DATA_PROCESSED) -> None:
    """Guarda cada conjunto en Parquet (sin escalar; el escalado vive en el pipeline)."""
    for nombre, datos in splits.items():
        datos.to_parquet(Path(carpeta) / f"{nombre}.parquet")


def cargar_splits(carpeta: Path = DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    """Carga los conjuntos generados por la Fase 3."""
    carpeta = Path(carpeta)
    faltan = [n for n in ("train", "val", "test") if not (carpeta / f"{n}.parquet").exists()]
    if faltan:
        raise FileNotFoundError(
            f"Faltan {faltan} en {carpeta}. Ejecuta antes: python -m src.data_preprocessing"
        )
    return {n: pd.read_parquet(carpeta / f"{n}.parquet") for n in ("train", "val", "test")}


def resumen_splits(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Filas, fraudes y porcentaje de fraude por conjunto."""
    return pd.DataFrame({
        nombre: {
            "Filas": len(d),
            "Fraudes": int(d[TARGET].sum()),
            "% fraude": round(d[TARGET].mean() * 100, 4),
        }
        for nombre, d in splits.items()
    }).T.astype({"Filas": int, "Fraudes": int})


def guardar_artefactos_preparacion(splits: dict[str, pd.DataFrame], preprocesador,
                                   info_limpieza: dict) -> dict:
    """Guarda splits, preprocesador ajustado y metadatos de la preparación."""
    guardar_splits(splits)
    joblib.dump(preprocesador, RUTA_PREPROCESADOR)
    metadata = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "limpieza": info_limpieza,
        "splits": resumen_splits(splits).to_dict(orient="index"),
        "preprocesador": str(RUTA_PREPROCESADOR.relative_to(MODELS_DIR.parent)),
    }
    RUTA_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def crear_datos_demo(n_legitimas: int = 4905, semilla: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Muestra liviana del conjunto de PRUEBA para la app (monitoreo y predicción):
    todos sus fraudes + n_legitimas legítimas, en formato original (sin variables
    derivadas). Con 95 fraudes y 4 905 legítimas, el fraude es ~1,9 % del flujo
    simulado (en la realidad es ~0,17 %), para que la demostración muestre alertas.
    """
    from src.config import DATA_DEMO

    prueba = cargar_splits()["test"]
    demo = pd.concat([
        prueba[prueba[TARGET] == 1],
        prueba[prueba[TARGET] == 0].sample(n_legitimas, random_state=semilla),
    ])[COLUMNAS_CRUDAS + [TARGET]].sample(frac=1, random_state=semilla)
    DATA_DEMO.parent.mkdir(parents=True, exist_ok=True)
    demo.to_parquet(DATA_DEMO)
    return demo


def ejecutar_preparacion() -> dict:
    """Fase 3 completa: carga -> limpieza -> variables -> división -> escalado -> guardado."""
    df, info = limpiar_datos(cargar_datos())
    df = crear_variables(df)
    splits = dividir_datos(df)
    X_train, _ = separar_xy(splits["train"])
    preprocesador = construir_preprocesador().fit(X_train)   # solo con entrenamiento
    return guardar_artefactos_preparacion(splits, preprocesador, info)


if __name__ == "__main__":
    meta = ejecutar_preparacion()
    print("Preparación completada.")
    print(json.dumps(meta["limpieza"], indent=2, ensure_ascii=False))
    print(pd.DataFrame(meta["splits"]).T)
