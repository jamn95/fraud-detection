"""
Evaluación (CRISP-DM fase 5).

Reglas para no "elegir mirando el examen":
  - Los umbrales de decisión se eligen con el conjunto de VALIDACIÓN.
  - El mejor modelo se elige con un criterio fijado de antemano sobre VALIDACIÓN.
  - El conjunto de PRUEBA solo se usa para reportar el desempeño final.

Modelo de costo (supuesto de negocio, ajustable):
  - Cada alerta se revisa manualmente: cuesta COSTO_REVISION euros (sea fraude o no).
  - Cada fraude no detectado (falso negativo) cuesta su monto (Amount).
  costo_total = Σ Amount(FN) + COSTO_REVISION · (número de alertas)

Política de umbral "de negocio": el umbral que minimiza el costo total en
validación, con la restricción Precision ≥ PRECISION_MINIMA (calidad mínima de
las alertas que puede absorber el equipo de revisión).

Ejecución completa desde la raíz del proyecto:
    python -m src.evaluate
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from src.config import FEATURES, MODELOS, MODELS_DIR, RANDOM_STATE, TARGET

COSTO_REVISION = 5.0          # € por alerta revisada
PRECISION_MINIMA = 0.70       # restricción de la política de umbral
CRITERIOS_EXITO = {"Recall": 0.80, "Precision": 0.70, "PR-AUC": 0.80}
TOLERANCIA_EMPATE = 0.005     # diferencia de PR-AUC que se considera empate
# Orden de complejidad (para desempatar se prefiere el más simple / rápido)
COMPLEJIDAD = {"Regresión Logística": 1, "XGBoost": 2, "Random Forest": 3, "DNN": 4,
               "Autoencoder": 5, "Stacking (RF + XGB)": 6}
RUTA_CONFIG_DESPLIEGUE = MODELS_DIR / "config_despliegue.json"


# ==========================================================================
# Costo y umbrales
# ==========================================================================
def costo_negocio(y, montos, pred, costo_revision: float = COSTO_REVISION) -> dict:
    """Desglose económico de un conjunto de predicciones (0/1)."""
    y, montos, pred = np.asarray(y), np.asarray(montos, dtype=float), np.asarray(pred).astype(bool)
    fraude = y == 1
    perdido = montos[fraude & ~pred].sum()
    alertas = int(pred.sum())
    return {
        "Fraude evitado (€)": float(montos[fraude & pred].sum()),
        "Fraude perdido (€)": float(perdido),
        "Alertas": alertas,
        "Costo revisión (€)": alertas * costo_revision,
        "Costo total (€)": float(perdido + alertas * costo_revision),
    }


def umbral_f1(y, puntaje) -> float:
    """Umbral que maximiza el F1."""
    p, r, t = precision_recall_curve(y, puntaje)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.nanargmax(f1[:-1])])


def curva_costo(y, puntaje, montos, costo_revision: float = COSTO_REVISION) -> pd.DataFrame:
    """
    Costo, precision, recall y alertas para cada umbral candidato. Los candidatos
    son los puntajes de los fraudes: entre dos de ellos solo cambian falsos
    positivos, así que el costo mínimo siempre cae en uno de estos valores.
    """
    y, puntaje, montos = np.asarray(y), np.asarray(puntaje), np.asarray(montos, dtype=float)
    filas = []
    for t in np.unique(puntaje[y == 1]):
        pred = puntaje >= t
        tp = int((pred & (y == 1)).sum())
        filas.append({
            "Umbral": float(t),
            "Costo total (€)": float(montos[(y == 1) & ~pred].sum() + costo_revision * pred.sum()),
            "Precision": tp / max(int(pred.sum()), 1),
            "Recall": tp / int((y == 1).sum()),
            "Alertas": int(pred.sum()),
        })
    return pd.DataFrame(filas)


def umbral_negocio(y, puntaje, montos, costo_revision: float = COSTO_REVISION,
                   precision_min: float | None = PRECISION_MINIMA) -> float:
    """Umbral de mínimo costo con Precision ≥ precision_min (si es alcanzable)."""
    curva = curva_costo(y, puntaje, montos, costo_revision)
    if precision_min is not None and (curva["Precision"] >= precision_min).any():
        curva = curva[curva["Precision"] >= precision_min]
    return float(curva.loc[curva["Costo total (€)"].idxmin(), "Umbral"])


def calcular_umbrales(puntajes_val: pd.DataFrame, modelos, costo_revision: float = COSTO_REVISION,
                      precision_min: float = PRECISION_MINIMA) -> dict:
    """Umbral de negocio y umbral F1 de cada modelo, ambos elegidos en validación."""
    y, montos = puntajes_val[TARGET], puntajes_val["Amount"]
    return {m: {"negocio": umbral_negocio(y, puntajes_val[m], montos, costo_revision, precision_min),
                "f1": umbral_f1(y, puntajes_val[m])} for m in modelos}


# ==========================================================================
# Métricas
# ==========================================================================
def evaluar_en_umbral(y, puntaje, montos, umbral: float,
                      costo_revision: float = COSTO_REVISION) -> dict:
    """Métricas de clasificación, matriz de confusión y costo para un umbral."""
    y, puntaje = np.asarray(y), np.asarray(puntaje)
    pred = (puntaje >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y, puntaje),
        "PR-AUC": average_precision_score(y, puntaje),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        **costo_negocio(y, montos, pred, costo_revision),
    }


def tabla_evaluacion(puntajes: pd.DataFrame, umbrales: dict, modelos, tipo_umbral: str = "negocio",
                     costo_revision: float = COSTO_REVISION) -> pd.DataFrame:
    """Tabla comparativa de modelos en un conjunto, con el umbral indicado."""
    y, montos = puntajes[TARGET], puntajes["Amount"]
    sin_modelo = float(montos[y == 1].sum())
    filas = []
    for m in modelos:
        fila = {"Modelo": m, "Umbral": umbrales[m][tipo_umbral],
                **evaluar_en_umbral(y, puntajes[m], montos, umbrales[m][tipo_umbral], costo_revision)}
        fila["Ahorro vs. sin modelo (%)"] = 100 * (sin_modelo - fila["Costo total (€)"]) / sin_modelo
        filas.append(fila)
    return pd.DataFrame(filas).set_index("Modelo")


def intervalos_bootstrap(y, puntaje, umbral: float, n: int = 1000, nivel: float = 0.95,
                         semilla: int = RANDOM_STATE) -> dict:
    """
    Intervalos de confianza por bootstrap (remuestreo con reemplazo del conjunto
    de prueba). Con solo ~95 fraudes en prueba, las métricas tienen bastante
    incertidumbre y conviene reportarla.
    """
    y, puntaje = np.asarray(y), np.asarray(puntaje)
    rng = np.random.default_rng(semilla)
    valores = {"PR-AUC": [], "ROC-AUC": [], "Precision": [], "Recall": [], "F1": []}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        yb, sb = y[idx], puntaje[idx]
        if yb.sum() == 0:
            continue
        pb = sb >= umbral
        valores["PR-AUC"].append(average_precision_score(yb, sb))
        valores["ROC-AUC"].append(roc_auc_score(yb, sb))
        valores["Precision"].append(precision_score(yb, pb, zero_division=0))
        valores["Recall"].append(recall_score(yb, pb))
        valores["F1"].append(f1_score(yb, pb))
    alfa = (1 - nivel) / 2 * 100
    return {k: (float(np.percentile(v, alfa)), float(np.percentile(v, 100 - alfa)))
            for k, v in valores.items()}


def seleccionar_mejor_modelo(puntajes_val: pd.DataFrame, modelos) -> tuple[str, pd.DataFrame]:
    """
    Criterio fijado de antemano (sobre VALIDACIÓN):
      1. Mayor PR-AUC en validación (no depende del umbral).
      2. Si varios quedan a menos de TOLERANCIA_EMPATE del mejor, se elige el
         de menor complejidad (más rápido de entrenar, servir y explicar).
    """
    tabla = pd.DataFrame({
        "PR-AUC (val)": {m: average_precision_score(puntajes_val[TARGET], puntajes_val[m]) for m in modelos},
        "Complejidad": {m: COMPLEJIDAD.get(m, 99) for m in modelos},
    }).sort_values("PR-AUC (val)", ascending=False)
    mejor_pr = tabla["PR-AUC (val)"].max()
    empatados = tabla[tabla["PR-AUC (val)"] >= mejor_pr - TOLERANCIA_EMPATE]
    tabla["Empate con el mejor"] = tabla.index.isin(empatados.index)
    return empatados["Complejidad"].idxmin(), tabla


def verificar_criterios(fila_test: pd.Series, ahorro: float) -> dict:
    """Compara el desempeño en prueba con los criterios de éxito de la Fase 1."""
    resultado = {f"{k} ≥ {v:.2f}": bool(fila_test[k] >= v) for k, v in CRITERIOS_EXITO.items()}
    resultado["Costo menor que sin modelo"] = bool(ahorro > 0)
    return resultado


# ==========================================================================
# Interpretabilidad
# ==========================================================================
def explicar_shap(pipeline_xgb, X: pd.DataFrame):
    """
    Valores SHAP del XGBoost (en log-odds). El pipeline se separa en su
    preprocesador y su modelo: SHAP se calcula sobre las variables escaladas,
    pero se muestran los valores originales para que el gráfico sea legible
    (el escalado es monótono, así que el orden de los valores no cambia).
    """
    import shap

    X_s = pipeline_xgb.named_steps["prep"].transform(X[FEATURES])
    explicador = shap.TreeExplainer(pipeline_xgb.named_steps["modelo"])
    valores = explicador(X_s)
    valores.data = X[FEATURES].to_numpy()
    return valores


# ==========================================================================
# Despliegue
# ==========================================================================
def guardar_config_despliegue(mejor_modelo: str, umbrales: dict, tabla_test: pd.DataFrame,
                              costo_revision: float = COSTO_REVISION,
                              precision_min: float = PRECISION_MINIMA) -> dict:
    """Configuración que usan la app y la API: modelo por defecto, umbrales y métricas."""
    columnas = ["Umbral", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TP", "FP", "FN",
                "Alertas", "Costo total (€)", "Ahorro vs. sin modelo (%)"]
    config = {
        "fecha_evaluacion": datetime.now().isoformat(timespec="seconds"),
        "mejor_modelo": mejor_modelo,
        "features": FEATURES,
        "costo_revision": costo_revision,
        "precision_minima": precision_min,
        "umbrales": umbrales,
        "metricas_prueba": json.loads(tabla_test[columnas].round(6).to_json(orient="index", force_ascii=False)),
    }
    RUTA_CONFIG_DESPLIEGUE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return config


def cargar_config_despliegue() -> dict:
    return json.loads(RUTA_CONFIG_DESPLIEGUE.read_text(encoding="utf-8"))


def evaluar_todo() -> dict:
    """Fase 5 sin gráficos: umbrales, tabla de prueba, selección y configuración."""
    from src.train_models import cargar_puntajes

    puntajes = cargar_puntajes()
    umbrales = calcular_umbrales(puntajes["val"], MODELOS)
    tabla = tabla_evaluacion(puntajes["test"], umbrales, MODELOS)
    mejor, _ = seleccionar_mejor_modelo(puntajes["val"], MODELOS)
    print(tabla[["Precision", "Recall", "F1", "PR-AUC", "Costo total (€)"]].round(4))
    print("Mejor modelo:", mejor)
    return guardar_config_despliegue(mejor, umbrales, tabla)


if __name__ == "__main__":
    evaluar_todo()
