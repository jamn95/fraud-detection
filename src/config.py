"""
Configuración central del proyecto.

Todas las rutas se calculan a partir de la ubicación de este archivo, así los
notebooks, los scripts de src/ y la app de Streamlit encuentran los datos y
modelos sin importar desde qué carpeta se ejecuten.
"""
import os
import random
import warnings
from pathlib import Path

import numpy as np

# En Windows 11, joblib/loky intenta contar los núcleos físicos con `wmic` (ya no
# viene instalado); al fallar imprime un aviso y un traceback, y termina usando los
# núcleos lógicos. Se precarga ese mismo valor en su caché para evitar el intento
# fallido: el comportamiento es idéntico, pero sin ruido en la salida.
try:
    from joblib.externals.loky.backend import context as _loky_context
    if getattr(_loky_context, "physical_cores_cache", None) is None:
        _loky_context.physical_cores_cache = os.cpu_count() or 1
except ImportError:
    pass
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw" / "creditcard.csv"
DATA_PROCESSED = RAIZ / "data" / "processed"
DATA_DEMO = RAIZ / "data" / "demo" / "transacciones_demo.parquet"   # muestra liviana para la app
MODELS_DIR = RAIZ / "models"
FIG_DIR = RAIZ / "reports" / "figures"

for _carpeta in (DATA_PROCESSED, MODELS_DIR, FIG_DIR):
    _carpeta.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Esquema del dataset
# --------------------------------------------------------------------------
COLUMNAS_V = [f"V{i}" for i in range(1, 29)]
COLUMNAS_CRUDAS = ["Time", *COLUMNAS_V, "Amount"]   # lo que llega en producción
TARGET = "Class"

# Variables que entran a los modelos (decididas en la Fase 2):
#   - V1..V28 tal cual (luego se estandarizan)
#   - Amount_log en lugar de Amount (reduce la asimetría)
#   - Hour_sin / Hour_cos en lugar de Time (Time crudo es un contador que no generaliza)
VARIABLES_DERIVADAS = ["Amount_log", "Hour_sin", "Hour_cos"]
FEATURES = [*COLUMNAS_V, *VARIABLES_DERIVADAS]

# Proporciones de la división estratificada (entrenamiento / validación / prueba = 60 / 20 / 20)
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# --------------------------------------------------------------------------
# Colores de las clases (paleta validada para daltonismo: azul / naranja).
# Se usan en todas las figuras para que cada clase tenga siempre el mismo color.
# --------------------------------------------------------------------------
COLOR_LEGITIMA = "#2a78d6"
COLOR_FRAUDE = "#eb6834"
COLOR_SINTETICO = "#1baf7a"   # fraudes creados por SMOTE (siempre con marcador propio y leyenda)
PALETA_CLASES = {0: COLOR_LEGITIMA, 1: COLOR_FRAUDE}
NOMBRES_CLASES = {0: "Legítima", 1: "Fraude"}

# Colores por modelo y por estrategia de balanceo: orden categórico fijo y validado
# para daltonismo. Cada modelo conserva su color en todas las figuras.
_PALETA_CATEGORICA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MODELOS = ["Regresión Logística", "Random Forest", "XGBoost", "Stacking (RF + XGB)",
           "DNN", "Autoencoder"]
COLORES_MODELOS = dict(zip(MODELOS, _PALETA_CATEGORICA))
BALANCEOS = ["Sin balanceo", "Pesos de clase", "SMOTE"]
COLORES_BALANCEO = dict(zip(BALANCEOS, _PALETA_CATEGORICA))


def fijar_semillas(semilla: int = RANDOM_STATE, incluir_tensorflow: bool = False) -> None:
    """Fija todas las fuentes de aleatoriedad para que los resultados se repitan."""
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)
    if incluir_tensorflow:
        # Import diferido: TensorFlow tarda en cargar y no todos los módulos lo necesitan
        import keras
        keras.utils.set_random_seed(semilla)   # fija Python, NumPy y TensorFlow a la vez
