"""
Utilidades compartidas por las páginas de la app: carga de modelos y datos
(con caché de Streamlit), formato de números y estilo común de los gráficos.
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (  # noqa: E402  (se reexportan para las páginas)
    COLOR_FRAUDE, COLOR_LEGITIMA, COLORES_MODELOS, COLUMNAS_CRUDAS, COLUMNAS_V, DATA_DEMO,
    FIG_DIR, MODELOS, MODELS_DIR, NOMBRES_CLASES, TARGET,
)
from src.data_preprocessing import hora_del_ciclo  # noqa: E402
from src.evaluate import COSTO_REVISION, evaluar_en_umbral  # noqa: E402
from src.predict import ACCIONES, DetectorFraude, marcar_por_modelos, nivel_riesgo  # noqa: E402

# El dataset no trae fechas reales: Time son segundos desde la primera transacción.
# Para mostrar una "fecha y hora" se usa este inicio ilustrativo (septiembre de 2013).
INICIO_REGISTRO = pd.Timestamp("2013-09-01 00:00:00")
COLOR_TEXTO = "#52514e"
# Colores de estado: reservados para el nivel de riesgo y siempre con ícono + texto
ETIQUETA_RIESGO = {"ALTO": "🔴 ALTO", "MEDIO": "🟠 MEDIO", "BAJO": "🟢 BAJO"}


# ----------------------------------------------------------------------
# Carga con caché
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos…")
def detector() -> DetectorFraude:
    """Un único detector para toda la sesión (los modelos se cargan una sola vez)."""
    return DetectorFraude()


@st.cache_data(show_spinner=False)
def datos_demo() -> pd.DataFrame:
    """Muestra del conjunto de prueba: 95 fraudes + 4 905 legítimas, en formato original."""
    return pd.read_parquet(DATA_DEMO)


@st.cache_data(show_spinner=False)
def puntajes_prueba() -> pd.DataFrame:
    """Puntajes precalculados de todos los modelos sobre el conjunto de prueba completo."""
    return pd.read_parquet(MODELS_DIR / "puntajes" / "puntajes_test.parquet")


# ----------------------------------------------------------------------
# Formato
# ----------------------------------------------------------------------
def numero(x: float, decimales: int = 0) -> str:
    """Formato español: 12.345,67."""
    texto = f"{x:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def eur(x: float, decimales: int = 2) -> str:
    return f"{numero(x, decimales)} €"


def pct(x: float, decimales: int = 1) -> str:
    return f"{numero(100 * x, decimales)} %"


def momento(time_segundos) -> pd.Series | pd.Timestamp:
    """Fecha y hora ilustrativas a partir de Time."""
    return INICIO_REGISTRO + pd.to_timedelta(time_segundos, unit="s")


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def estilo(fig: go.Figure, alto: int = 380, titulo: str | None = None) -> go.Figure:
    """Estilo común: fondo limpio, rejilla tenue, leyenda arriba y tooltip claro."""
    if titulo:   # sin título no se pasa la clave: Plotly mostraría "undefined"
        fig.update_layout(title=dict(text=titulo, x=0, xanchor="left", font=dict(size=15)))
    fig.update_layout(
        template="plotly_white",
        height=alto,
        margin=dict(l=10, r=10, t=60 if titulo else 30, b=10),
        font=dict(size=13, color="#0b0b0b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    return fig


def selector_modelo(clave: str, etiqueta: str = "Modelo", contenedor=st) -> str:
    """Selector de modelo con el mejor modelo preseleccionado."""
    det = detector()
    opciones = det.modelos_disponibles
    return contenedor.selectbox(
        etiqueta, opciones, index=opciones.index(det.modelo_defecto), key=clave,
        format_func=lambda m: f"{m}  ★" if m == det.modelo_defecto else m,
        help="★ = modelo seleccionado en la evaluación (Fase 5)",
    )


def control_umbral(modelo: str, puntajes: pd.Series, contenedor=st) -> float:
    """Control del umbral de decisión, iniciado en el umbral de negocio del modelo."""
    det = detector()
    base = det.umbrales[modelo]
    if modelo == "Autoencoder":     # el puntaje es un error de reconstrucción, no una probabilidad
        maximo = float(max(np.quantile(puntajes, 0.9995), base * 2))
        return contenedor.slider("Umbral (error de reconstrucción)", 0.0, maximo, float(base),
                                 step=maximo / 500, format="%.2f", key=f"umbral_{modelo}")
    return contenedor.slider("Umbral de probabilidad", 0.001, 0.999, float(round(base, 3)), step=0.001,
                             format="%.3f", key=f"umbral_{modelo}",
                             help=f"Umbral de negocio elegido en validación: {base:.4f}")


def pie_de_pagina() -> None:
    st.divider()
    st.caption("Proyecto Final de Minería de Datos · Metodología CRISP-DM · Dataset: Credit Card Fraud "
               "Detection (ULB, Kaggle). Las fechas son ilustrativas: el dataset solo contiene segundos "
               "desde la primera transacción.")
