"""
Modelado (CRISP-DM fase 4).

Modelos:
  - Regresión Logística (línea base)
  - Random Forest (bagging)
  - XGBoost (boosting)
  - Stacking (RF + XGB con meta-modelo logístico; idea del ejemplo de clase)
  - Red neuronal profunda (DNN, Keras)
  - Autoencoder para detección de anomalías (Keras, solo transacciones legítimas)

Procedimiento para los modelos de scikit-learn / XGBoost:
  1. Comparar tres estrategias de balanceo (sin balanceo, pesos de clase, SMOTE)
     con hiperparámetros por defecto, midiendo PR-AUC en VALIDACIÓN.
  2. Con la estrategia ganadora, ajustar hiperparámetros con RandomizedSearchCV y
     validación cruzada estratificada SOLO sobre entrenamiento. El escalador y SMOTE
     viven dentro del pipeline, así que se reajustan en cada pliegue (sin fuga).
  3. Guardar el modelo y sus puntajes en validación y prueba. El conjunto de prueba
     no se usa para ninguna decisión: se evalúa en la Fase 5.

Ejecución completa desde la raíz del proyecto:
    python -m src.train_models
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src.config import BALANCEOS, MODELS_DIR, RANDOM_STATE, TARGET, fijar_semillas
from src.data_preprocessing import crear_smote, pipeline_con_balanceo

RUTA_REGISTRO = MODELS_DIR / "registro_modelos.json"
RUTA_PUNTAJES = MODELS_DIR / "puntajes"          # puntajes val/test por modelo (Fase 5)

# Cómo implementa cada estrategia de balanceo un modelo de scikit-learn / XGBoost
VARIANTES = {
    "Sin balanceo": {"smote": None, "pesos": False},
    "Pesos de clase": {"smote": None, "pesos": True},
    "SMOTE": {"smote": 0.1, "pesos": False},
}
assert list(VARIANTES) == BALANCEOS

TIPOS = {"logistica": "Regresión Logística", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def razon_desbalance(y) -> float:
    """Legítimas por cada fraude (base para scale_pos_weight)."""
    y = np.asarray(y)
    return float((y == 0).sum() / (y == 1).sum())


# ==========================================================================
# Modelos de scikit-learn / XGBoost
# ==========================================================================
def crear_estimador(tipo: str, pesos: bool, ratio: float, **params):
    """Crea el estimador con hiperparámetros base razonables."""
    if tipo == "logistica":
        base = dict(max_iter=3000, random_state=RANDOM_STATE)
        if pesos:
            base["class_weight"] = "balanced"
        return LogisticRegression(**{**base, **params})
    if tipo == "random_forest":
        # max_samples < 1 hace que cada árbol use una submuestra bootstrap: más
        # diversidad entre árboles y entrenamiento ~2,5 veces más rápido
        base = dict(n_estimators=100, max_samples=0.5, min_samples_leaf=2,
                    max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE)
        if pesos:
            base["class_weight"] = "balanced_subsample"
        return RandomForestClassifier(**{**base, **params})
    if tipo == "xgboost":
        base = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=3, tree_method="hist",
                    eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE)
        if pesos:
            base["scale_pos_weight"] = ratio
        return XGBClassifier(**{**base, **params})
    raise ValueError(f"Tipo de modelo desconocido: {tipo}")


def crear_pipeline(tipo: str, variante: str, ratio: float, **params):
    """Pipeline escalador -> [SMOTE] -> modelo para una estrategia de balanceo."""
    cfg = VARIANTES[variante]
    return pipeline_con_balanceo(crear_estimador(tipo, cfg["pesos"], ratio, **params), cfg["smote"])


def puntaje(modelo, X) -> np.ndarray:
    """Probabilidad de fraude."""
    return modelo.predict_proba(X)[:, 1]


def comparar_balanceo(tipos, X_train, y_train, X_val, y_val):
    """
    Paso 1: entrena cada modelo con las tres estrategias de balanceo y mide en
    validación. Devuelve (tabla de resultados, modelos entrenados).
    """
    ratio = razon_desbalance(y_train)
    filas, modelos = [], {}
    for tipo in tipos:
        for variante in VARIANTES:
            inicio = time.perf_counter()
            modelo = crear_pipeline(tipo, variante, ratio).fit(X_train, y_train)
            p = puntaje(modelo, X_val)
            filas.append({
                "Modelo": TIPOS[tipo], "Balanceo": variante,
                "PR-AUC (val)": average_precision_score(y_val, p),
                "ROC-AUC (val)": roc_auc_score(y_val, p),
                "Tiempo (s)": time.perf_counter() - inicio,
            })
            modelos[(TIPOS[tipo], variante)] = modelo
    return pd.DataFrame(filas), modelos


def espacio_busqueda(tipo: str, variante: str, ratio: float) -> dict:
    """Distribuciones de hiperparámetros para RandomizedSearchCV."""
    espacio = {}
    if tipo == "logistica":
        espacio["modelo__C"] = loguniform(1e-3, 1e1)          # inverso de la regularización
    elif tipo == "random_forest":
        espacio.update({
            "modelo__n_estimators": randint(150, 301),
            "modelo__max_depth": [None, 12, 20],
            "modelo__min_samples_leaf": randint(1, 6),
            "modelo__max_features": ["sqrt", 0.3],
            "modelo__max_samples": [0.3, 0.5],
        })
    elif tipo == "xgboost":
        espacio.update({
            "modelo__n_estimators": randint(200, 801),
            "modelo__max_depth": randint(3, 8),
            "modelo__learning_rate": loguniform(0.01, 0.2),
            "modelo__subsample": uniform(0.6, 0.4),            # 0.6 – 1.0
            "modelo__colsample_bytree": uniform(0.5, 0.5),     # 0.5 – 1.0
            "modelo__min_child_weight": randint(1, 8),
            "modelo__gamma": uniform(0, 2),
            "modelo__reg_lambda": loguniform(0.1, 10),
        })
        if variante == "Pesos de clase":
            # El peso "teórico" (≈ 599) suele ser demasiado agresivo; se prueban valores menores
            espacio["modelo__scale_pos_weight"] = [np.sqrt(ratio), ratio / 10, ratio / 4, ratio]
    if VARIANTES[variante]["smote"] is not None:
        espacio["smote__sampling_strategy"] = [0.05, 0.1, 0.25, 0.5]
    return espacio


def ajustar_hiperparametros(tipo: str, variante: str, X, y, n_iter: int,
                            cv_folds: int = 3, n_jobs: int = 1) -> RandomizedSearchCV:
    """
    Paso 2: búsqueda aleatoria con validación cruzada estratificada sobre
    entrenamiento. Métrica: PR-AUC (average_precision), adecuada para el
    desbalance. refit=True reentrena la mejor configuración con todo el train.
    """
    ratio = razon_desbalance(y)
    busqueda = RandomizedSearchCV(
        estimator=crear_pipeline(tipo, variante, ratio),
        param_distributions=espacio_busqueda(tipo, variante, ratio),
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
    )
    return busqueda.fit(X, y)


def resumen_busqueda(busqueda: RandomizedSearchCV, top: int = 5) -> pd.DataFrame:
    """Mejores configuraciones de una búsqueda, con media y desviación del PR-AUC en CV."""
    res = pd.DataFrame(busqueda.cv_results_)
    columnas = [c for c in res.columns if c.startswith("param_")]
    tabla = res[columnas + ["mean_test_score", "std_test_score", "mean_fit_time"]]
    tabla = tabla.rename(columns=lambda c: c.replace("param_modelo__", "").replace("param_", ""))
    tabla = tabla.rename(columns={"mean_test_score": "PR-AUC CV (media)",
                                  "std_test_score": "PR-AUC CV (desv.)",
                                  "mean_fit_time": "Tiempo ajuste (s)"})
    return tabla.sort_values("PR-AUC CV (media)", ascending=False).head(top)


def limpiar_parametros(params: dict) -> dict:
    """Convierte los parámetros a tipos JSON (para el registro)."""
    limpio = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        limpio[k] = v
    return limpio


def crear_stacking(modelo_rf, modelo_xgb, cv_folds: int = 3) -> StackingClassifier:
    """
    Stacking: los pipelines ya ajustados de RF y XGBoost se clonan (sin entrenar)
    como modelos base. StackingClassifier genera sus probabilidades fuera de
    pliegue con validación cruzada estratificada, y una Regresión Logística
    aprende a combinarlas. SMOTE/pesos de cada pipeline base se respetan.
    """
    return StackingClassifier(
        estimators=[("rf", clone(modelo_rf)), ("xgb", clone(modelo_xgb))],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )


# ==========================================================================
# Redes neuronales (Keras). Reciben datos YA escalados con el preprocesador.
# ==========================================================================
def construir_dnn(n_entradas: int, capas=(64, 32, 16), dropout: float = 0.3, lr: float = 1e-3):
    """
    Red densa: capas ocultas ReLU con Batch Normalization y Dropout (contra el
    sobreajuste) y salida sigmoide = probabilidad de fraude.
    """
    import keras
    from keras import layers

    modelo = keras.Sequential([keras.Input(shape=(n_entradas,))], name="dnn_fraude")
    for i, unidades in enumerate(capas):
        modelo.add(layers.Dense(unidades, activation="relu"))
        if i < len(capas) - 1:
            modelo.add(layers.BatchNormalization())
            modelo.add(layers.Dropout(dropout))
    modelo.add(layers.Dense(1, activation="sigmoid"))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                   loss="binary_crossentropy",
                   metrics=[keras.metrics.AUC(curve="PR", name="pr_auc")])
    return modelo


def entrenar_dnn(X_train_s, y_train, X_val_s, y_val, capas=(64, 32, 16), dropout=0.3, lr=1e-3,
                 balanceo="Sin balanceo", smote_ratio=0.1, epocas=60, batch=2048, paciencia=6):
    """
    Entrena la DNN con early stopping sobre el PR-AUC de validación.
    balanceo: "Sin balanceo" | "Pesos de clase" (class_weight) | "SMOTE" (solo train).
    """
    import keras

    fijar_semillas(incluir_tensorflow=True)
    X_tr = np.asarray(X_train_s, dtype="float32")
    y_tr = np.asarray(y_train)
    pesos = None
    if balanceo == "SMOTE":
        X_tr, y_tr = crear_smote(smote_ratio).fit_resample(X_tr, y_tr)
    elif balanceo == "Pesos de clase":
        n = len(y_tr)   # pesos "balanced": n / (2 · n_clase)
        pesos = {0: n / (2 * (y_tr == 0).sum()), 1: n / (2 * (y_tr == 1).sum())}

    modelo = construir_dnn(X_tr.shape[1], capas, dropout, lr)
    parada = keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=paciencia,
                                           restore_best_weights=True)
    historial = modelo.fit(
        X_tr, y_tr,
        validation_data=(np.asarray(X_val_s, dtype="float32"), np.asarray(y_val)),
        epochs=epocas, batch_size=batch, class_weight=pesos, callbacks=[parada], verbose=0,
    )
    return modelo, historial.history


def puntaje_keras(modelo, X_s) -> np.ndarray:
    return modelo.predict(np.asarray(X_s, dtype="float32"), batch_size=8192, verbose=0).ravel()


def construir_autoencoder(n_entradas: int, capas=(20, 10), lr: float = 1e-3):
    """
    Autoencoder simétrico: el codificador comprime la transacción hasta el cuello
    de botella (última capa de `capas`) y el decodificador la reconstruye.
    """
    import keras
    from keras import layers

    modelo = keras.Sequential([keras.Input(shape=(n_entradas,))], name="autoencoder_fraude")
    for unidades in capas:                      # codificador
        modelo.add(layers.Dense(unidades, activation="relu"))
    for unidades in reversed(capas[:-1]):       # decodificador (espejo)
        modelo.add(layers.Dense(unidades, activation="relu"))
    modelo.add(layers.Dense(n_entradas, activation="linear"))
    modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return modelo


def entrenar_autoencoder(X_normal_train_s, X_normal_val_s, capas=(20, 10), lr=1e-3,
                         epocas=80, batch=1024, paciencia=6):
    """Entrena SOLO con transacciones legítimas: aprende a reconstruir lo "normal"."""
    import keras

    fijar_semillas(incluir_tensorflow=True)
    X_tr = np.asarray(X_normal_train_s, dtype="float32")
    X_va = np.asarray(X_normal_val_s, dtype="float32")
    modelo = construir_autoencoder(X_tr.shape[1], capas, lr)
    parada = keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                           restore_best_weights=True)
    historial = modelo.fit(X_tr, X_tr, validation_data=(X_va, X_va), epochs=epocas,
                           batch_size=batch, callbacks=[parada], verbose=0)
    return modelo, historial.history


def error_reconstruccion(modelo, X_s) -> np.ndarray:
    """Error cuadrático medio por transacción = puntaje de anomalía."""
    X = np.asarray(X_s, dtype="float32")
    reconstruido = modelo.predict(X, batch_size=8192, verbose=0)
    return np.mean(np.square(X - reconstruido), axis=1)


# ==========================================================================
# Persistencia
# ==========================================================================
def cargar_registro() -> dict:
    if RUTA_REGISTRO.exists():
        return json.loads(RUTA_REGISTRO.read_text(encoding="utf-8"))
    return {}


def guardar_modelo(nombre: str, modelo, tipo: str, archivo: str, **info) -> Path:
    """
    Guarda un modelo y lo anota en models/registro_modelos.json.
    tipo: "pipeline"    -> scikit-learn/imblearn; recibe FEATURES sin escalar
          "dnn"         -> Keras; recibe FEATURES escaladas con preprocesador.joblib
          "autoencoder" -> Keras; puntaje = error de reconstrucción
    """
    ruta = MODELS_DIR / archivo
    if tipo in ("dnn", "autoencoder"):
        modelo.save(ruta)
    else:
        joblib.dump(modelo, ruta, compress=3)
    registro = cargar_registro()
    registro[nombre] = {"archivo": archivo, "tipo": tipo, **info}
    RUTA_REGISTRO.write_text(json.dumps(registro, ensure_ascii=False, indent=2, default=str),
                             encoding="utf-8")
    return ruta


def guardar_puntajes(puntajes_val: dict, puntajes_test: dict, val: pd.DataFrame,
                     test: pd.DataFrame) -> None:
    """
    Guarda los puntajes de cada modelo en validación y prueba, junto con la
    etiqueta real y los datos que necesita el análisis de costo (Amount, Time).
    """
    RUTA_PUNTAJES.mkdir(parents=True, exist_ok=True)
    for nombre, puntajes, datos in (("val", puntajes_val, val), ("test", puntajes_test, test)):
        tabla = datos[["Time", "Amount", TARGET]].copy()
        for modelo, valores in puntajes.items():
            tabla[modelo] = np.asarray(valores, dtype="float64")
        tabla.to_parquet(RUTA_PUNTAJES / f"puntajes_{nombre}.parquet")


def cargar_puntajes() -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(RUTA_PUNTAJES / f"puntajes_{n}.parquet") for n in ("val", "test")}


# ==========================================================================
# Ejecución completa (misma secuencia que notebooks/03_modelado.ipynb)
# ==========================================================================
def entrenar_todo(n_iter: dict | None = None) -> None:
    import keras  # noqa: F401  (carga TensorFlow una vez)
    from src.data_preprocessing import RUTA_PREPROCESADOR, cargar_splits, separar_xy

    n_iter = n_iter or {"logistica": 10, "random_forest": 8, "xgboost": 20}
    fijar_semillas()
    splits = cargar_splits()
    X_train, y_train = separar_xy(splits["train"])
    X_val, y_val = separar_xy(splits["val"])
    X_test, _ = separar_xy(splits["test"])
    p_val, p_test = {}, {}

    # 1) Comparación de balanceo
    tabla, modelos_var = comparar_balanceo(list(TIPOS), X_train, y_train, X_val, y_val)
    print(tabla.round(4).to_string(index=False))
    for (modelo, variante), m in modelos_var.items():
        p_val[f"{modelo} · {variante}"] = puntaje(m, X_val)
        p_test[f"{modelo} · {variante}"] = puntaje(m, X_test)

    # 2) Ajuste de hiperparámetros con la mejor estrategia de cada modelo
    finales = {}
    for tipo, nombre in TIPOS.items():
        sub = tabla[tabla["Modelo"] == nombre]
        variante = sub.loc[sub["PR-AUC (val)"].idxmax(), "Balanceo"]
        busqueda = ajustar_hiperparametros(tipo, variante, X_train, y_train, n_iter[tipo],
                                           n_jobs=-1 if tipo == "logistica" else 1)
        finales[nombre] = busqueda.best_estimator_
        p_val[nombre], p_test[nombre] = puntaje(busqueda.best_estimator_, X_val), puntaje(busqueda.best_estimator_, X_test)
        guardar_modelo(nombre, busqueda.best_estimator_, "pipeline", f"{tipo}.joblib",
                       balanceo=variante, params=limpiar_parametros(busqueda.best_params_),
                       pr_auc_cv=float(busqueda.best_score_),
                       pr_auc_val=float(average_precision_score(y_val, p_val[nombre])))
        print(f"{nombre}: {variante} | PR-AUC CV = {busqueda.best_score_:.4f}")

    # 3) Stacking
    stacking = crear_stacking(finales["Random Forest"], finales["XGBoost"]).fit(X_train, y_train)
    p_val["Stacking (RF + XGB)"], p_test["Stacking (RF + XGB)"] = puntaje(stacking, X_val), puntaje(stacking, X_test)
    guardar_modelo("Stacking (RF + XGB)", stacking, "pipeline", "stacking.joblib",
                   pr_auc_val=float(average_precision_score(y_val, p_val["Stacking (RF + XGB)"])))

    # 4) Redes neuronales con datos escalados
    prep = joblib.load(RUTA_PREPROCESADOR)
    Xs_tr, Xs_val, Xs_te = prep.transform(X_train), prep.transform(X_val), prep.transform(X_test)
    # Configuraciones elegidas en notebooks/03_modelado.ipynb (búsqueda sobre validación)
    cfg_dnn = {"capas": (32, 16), "dropout": 0.4, "lr": 1e-3}
    dnn, _ = entrenar_dnn(Xs_tr, y_train, Xs_val, y_val, balanceo="Sin balanceo", **cfg_dnn)
    p_val["DNN"], p_test["DNN"] = puntaje_keras(dnn, Xs_val), puntaje_keras(dnn, Xs_te)
    guardar_modelo("DNN", dnn, "dnn", "dnn.keras", balanceo="Sin balanceo",
                   params={**cfg_dnn, "capas": list(cfg_dnn["capas"])},
                   pr_auc_val=float(average_precision_score(y_val, p_val["DNN"])))
    capas_ae = (28, 14, 7)
    ae, _ = entrenar_autoencoder(Xs_tr[y_train.values == 0], Xs_val[y_val.values == 0], capas=capas_ae)
    p_val["Autoencoder"], p_test["Autoencoder"] = error_reconstruccion(ae, Xs_val), error_reconstruccion(ae, Xs_te)
    guardar_modelo("Autoencoder", ae, "autoencoder", "autoencoder.keras",
                   params={"capas": list(capas_ae)},
                   pr_auc_val=float(average_precision_score(y_val, p_val["Autoencoder"])))

    guardar_puntajes(p_val, p_test, splits["val"], splits["test"])
    print("Modelos y puntajes guardados en", MODELS_DIR)


if __name__ == "__main__":
    entrenar_todo()
