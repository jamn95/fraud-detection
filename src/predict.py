"""
Inferencia (CRISP-DM fase 6).

`DetectorFraude` carga los artefactos de models/ y aplica exactamente el mismo
preprocesamiento del entrenamiento. Lo usan la app de Streamlit y la API.

Uso desde la terminal (raíz del proyecto):
    python -m src.predict transacciones.csv --modelo XGBoost --salida resultado.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import COLUMNAS_CRUDAS, FEATURES, MODELS_DIR, MODELOS
from src.data_preprocessing import hora_del_ciclo, preparar_para_prediccion

NIVELES_RIESGO = ["BAJO", "MEDIO", "ALTO"]
FACTOR_RIESGO_MEDIO = 0.5   # MEDIO: puntaje entre 50 % del umbral y el umbral

ACCIONES = {
    "ALTO": "Bloquear la transacción y contactar al titular",
    "MEDIO": "Enviar a revisión manual o pedir verificación adicional (3-D Secure)",
    "BAJO": "Aprobar",
}


def nivel_riesgo(puntaje, umbral: float) -> np.ndarray:
    """ALTO si supera el umbral; MEDIO si supera la mitad del umbral; BAJO en otro caso."""
    puntaje = np.asarray(puntaje, dtype=float)
    return np.select([puntaje >= umbral, puntaje >= umbral * FACTOR_RIESGO_MEDIO],
                     ["ALTO", "MEDIO"], default="BAJO")


class DetectorFraude:
    """Punto único de inferencia: carga perezosa de modelos y puntuación."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.config = json.loads((self.models_dir / "config_despliegue.json").read_text(encoding="utf-8"))
        self.registro = json.loads((self.models_dir / "registro_modelos.json").read_text(encoding="utf-8"))
        if self.config["features"] != FEATURES:
            raise ValueError("Las variables de config_despliegue.json no coinciden con src/config.py")
        self.modelo_defecto = self.config["mejor_modelo"]
        self.umbrales = {m: float(v["negocio"]) for m, v in self.config["umbrales"].items()}
        self._modelos = {}
        self._preprocesador = None
        self._explicador = None

    # ------------------------------------------------------------------
    @property
    def modelos_disponibles(self) -> list[str]:
        return [m for m in MODELOS if m in self.registro and m in self.umbrales]

    def _validar_modelo(self, nombre: str | None) -> str:
        nombre = nombre or self.modelo_defecto
        if nombre not in self.modelos_disponibles:
            raise ValueError(f"Modelo desconocido: {nombre}. Disponibles: {self.modelos_disponibles}")
        return nombre

    def _cargar(self, nombre: str):
        if nombre not in self._modelos:
            info = self.registro[nombre]
            ruta = self.models_dir / info["archivo"]
            if info["tipo"] in ("dnn", "autoencoder"):
                import keras   # import diferido: TensorFlow tarda en cargar
                self._modelos[nombre] = keras.models.load_model(ruta)
            else:
                self._modelos[nombre] = joblib.load(ruta)
        return self._modelos[nombre]

    @property
    def preprocesador(self):
        if self._preprocesador is None:
            self._preprocesador = joblib.load(self.models_dir / "preprocesador.joblib")
        return self._preprocesador

    # ------------------------------------------------------------------
    def puntaje(self, X: pd.DataFrame, nombre: str) -> np.ndarray:
        """
        Puntaje de riesgo para variables ya derivadas (FEATURES, sin escalar):
        probabilidad de fraude, o error de reconstrucción en el autoencoder.
        """
        tipo = self.registro[nombre]["tipo"]
        modelo = self._cargar(nombre)
        if tipo == "pipeline":                       # incluye su propio escalador
            return modelo.predict_proba(X[FEATURES])[:, 1]
        X_s = np.asarray(self.preprocesador.transform(X[FEATURES]), dtype="float32")
        # Para lotes pequeños (tiempo real), llamar al modelo directamente es más rápido que predict()
        salida = modelo(X_s, training=False).numpy() if len(X_s) <= 2048 else \
            modelo.predict(X_s, batch_size=8192, verbose=0)
        if tipo == "dnn":
            return salida.ravel()
        return np.mean(np.square(X_s - salida), axis=1)   # autoencoder

    def predecir(self, df_crudo: pd.DataFrame, modelo: str | None = None,
                 umbral: float | None = None) -> pd.DataFrame:
        """
        Clasifica transacciones en formato original (Time, V1..V28, Amount).
        Devuelve puntaje, umbral, alerta, nivel de riesgo y acción sugerida.
        """
        nombre = self._validar_modelo(modelo)
        umbral = self.umbrales[nombre] if umbral is None else float(umbral)
        X = preparar_para_prediccion(df_crudo)
        p = self.puntaje(X, nombre)
        salida = pd.DataFrame(index=df_crudo.index)
        salida["Time"] = df_crudo["Time"].values
        salida["Hora"] = hora_del_ciclo(df_crudo["Time"]).astype(int).values
        salida["Amount"] = df_crudo["Amount"].values
        salida["puntaje"] = p
        salida["umbral"] = umbral
        salida["alerta"] = p >= umbral
        salida["riesgo"] = nivel_riesgo(p, umbral)
        salida["accion"] = salida["riesgo"].map(ACCIONES)
        salida["modelo"] = nombre
        return salida

    def predecir_todos(self, df_crudo: pd.DataFrame, modelos: list[str] | None = None) -> pd.DataFrame:
        """Puntaje y alerta de varios modelos a la vez, más qué modelos marcaron cada transacción."""
        modelos = modelos or self.modelos_disponibles
        X = preparar_para_prediccion(df_crudo)
        salida = pd.DataFrame(index=df_crudo.index)
        for m in modelos:
            salida[m] = self.puntaje(X, m)
        return marcar_por_modelos(salida, self.umbrales, modelos)

    # ------------------------------------------------------------------
    def explicar(self, df_crudo: pd.DataFrame, top: int = 8) -> pd.DataFrame:
        """
        Contribuciones SHAP del XGBoost para UNA transacción (log-odds):
        valores positivos empujan hacia fraude y negativos hacia legítima.
        """
        import shap

        if len(df_crudo) != 1:
            raise ValueError("explicar() recibe una sola transacción")
        pipeline = self._cargar("XGBoost")
        if self._explicador is None:
            self._explicador = shap.TreeExplainer(pipeline.named_steps["modelo"])
        X = preparar_para_prediccion(df_crudo)
        valores = self._explicador.shap_values(pipeline.named_steps["prep"].transform(X))[0]
        tabla = pd.DataFrame({"Variable": FEATURES, "Valor": X.iloc[0].values, "Contribución": valores})
        tabla = tabla.reindex(tabla["Contribución"].abs().sort_values(ascending=False).index).head(top)
        return tabla.reset_index(drop=True)


def marcar_por_modelos(puntajes: pd.DataFrame, umbrales: dict, modelos: list[str]) -> pd.DataFrame:
    """Agrega, para cada transacción, la lista y el número de modelos que la marcan como alerta."""
    salida = puntajes.copy()
    alertas = pd.DataFrame({m: salida[m] >= umbrales[m] for m in modelos}, index=salida.index)
    salida["Marcada por"] = alertas.apply(lambda fila: ", ".join(m for m in modelos if fila[m]), axis=1)
    salida["N.º modelos"] = alertas.sum(axis=1).astype(int)
    return salida


def _main() -> None:
    parser = argparse.ArgumentParser(description="Clasifica transacciones de un CSV")
    parser.add_argument("csv", type=Path, help="CSV con columnas Time, V1..V28, Amount")
    parser.add_argument("--modelo", default=None, help="Modelo a usar (por defecto, el mejor)")
    parser.add_argument("--salida", type=Path, default=None, help="CSV de salida")
    args = parser.parse_args()

    datos = pd.read_csv(args.csv)
    faltan = set(COLUMNAS_CRUDAS) - set(datos.columns)
    if faltan:
        raise SystemExit(f"Faltan columnas: {sorted(faltan)}")
    resultado = DetectorFraude().predecir(datos, args.modelo)
    print(resultado[["Amount", "puntaje", "alerta", "riesgo"]].to_string())
    print(f"\nAlertas: {int(resultado['alerta'].sum())} de {len(resultado)}")
    if args.salida:
        resultado.to_csv(args.salida, index=False)
        print("Guardado en", args.salida)


if __name__ == "__main__":
    _main()
