"""
Pruebas del despliegue: predictor, API y páginas de la app.
Ejecutar desde la raíz del proyecto:  python -m pytest -q
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNAS_CRUDAS, DATA_DEMO, TARGET
from src.predict import DetectorFraude, nivel_riesgo


@pytest.fixture(scope="module")
def detector():
    return DetectorFraude()


@pytest.fixture(scope="module")
def demo():
    return pd.read_parquet(DATA_DEMO)


# ----------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------
def test_nivel_riesgo():
    assert list(nivel_riesgo([0.9, 0.04, 0.01], 0.06)) == ["ALTO", "MEDIO", "BAJO"]


def test_reproduce_puntajes_de_evaluacion(detector):
    """La inferencia en producción debe dar los mismos puntajes que en la evaluación."""
    from src.data_preprocessing import cargar_splits
    from src.train_models import cargar_puntajes

    prueba = cargar_splits()["test"].head(2000)
    referencia = cargar_puntajes()["test"].loc[prueba.index]
    for modelo in detector.modelos_disponibles:
        obtenido = detector.predecir(prueba[COLUMNAS_CRUDAS], modelo)["puntaje"].to_numpy()
        np.testing.assert_allclose(obtenido, referencia[modelo].to_numpy(), rtol=1e-4, atol=1e-5,
                                   err_msg=modelo)


def test_detecta_fraudes_de_demo(detector, demo):
    fraudes = demo[demo[TARGET] == 1]
    resultado = detector.predecir(fraudes[COLUMNAS_CRUDAS])
    assert resultado["alerta"].mean() > 0.6
    assert set(resultado["riesgo"]) <= {"BAJO", "MEDIO", "ALTO"}


def test_columnas_faltantes(detector, demo):
    with pytest.raises(ValueError, match="Faltan columnas"):
        detector.predecir(demo[COLUMNAS_CRUDAS].drop(columns="V3").head(1))


def test_modelo_desconocido(detector, demo):
    with pytest.raises(ValueError, match="Modelo desconocido"):
        detector.predecir(demo[COLUMNAS_CRUDAS].head(1), "Modelo inexistente")


def test_explicar_shap_suma_coherente(detector, demo):
    tabla = detector.explicar(demo[COLUMNAS_CRUDAS].head(1), top=31)
    assert len(tabla) == 31 and tabla["Contribución"].abs().is_monotonic_decreasing


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def cliente():
    from fastapi.testclient import TestClient
    from app.api import app
    return TestClient(app)


def test_api_health(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200 and respuesta.json()["estado"] == "ok"


def test_api_predict_una_y_varias(cliente, demo):
    filas = demo[COLUMNAS_CRUDAS].head(3).to_dict("records")
    una = cliente.post("/predict", json=filas[0])
    varias = cliente.post("/predict", json=filas, params={"modelo": "Random Forest"})
    assert una.status_code == 200 and len(una.json()) == 1
    assert varias.status_code == 200 and len(varias.json()) == 3
    assert varias.json()[0]["modelo"] == "Random Forest"


def test_api_valida_entrada(cliente, demo):
    fila = demo[COLUMNAS_CRUDAS].iloc[0].to_dict()
    fila.pop("V1")
    assert cliente.post("/predict", json=fila).status_code == 422
    assert cliente.post("/predict", json=demo[COLUMNAS_CRUDAS].iloc[0].to_dict(),
                        params={"modelo": "X"}).status_code == 400


# ----------------------------------------------------------------------
# App de Streamlit (cada página se ejecuta sin navegador)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pagina", ["inicio", "dashboard", "monitoreo", "sospechosas", "prediccion"])
def test_paginas_sin_errores(pagina):
    from streamlit.testing.v1 import AppTest

    prueba = AppTest.from_file(str(RAIZ / "app" / "paginas" / f"{pagina}.py"), default_timeout=120).run()
    assert not prueba.exception, prueba.exception


def test_monitoreo_procesa_y_alerta():
    from streamlit.testing.v1 import AppTest

    prueba = AppTest.from_file(str(RAIZ / "app" / "paginas" / "monitoreo.py"), default_timeout=120).run()
    prueba.session_state["mon_corriendo"] = True
    for _ in range(4):           # cuatro ciclos de 50 transacciones
        prueba.select_slider[0].set_value(50).run()
    assert not prueba.exception, prueba.exception
    assert prueba.session_state["mon_pos"] >= 150
    assert len(prueba.session_state["mon_historial"]) == prueba.session_state["mon_pos"]


def test_prediccion_formulario():
    from streamlit.testing.v1 import AppTest

    prueba = AppTest.from_file(str(RAIZ / "app" / "paginas" / "prediccion.py"), default_timeout=120).run()
    prueba.button[0].click().run()                        # cargar ejemplo de fraude
    prueba.button(key="FormSubmitter:formulario-Evaluar transacción").click().run()
    assert not prueba.exception, prueba.exception
    assert prueba.metric, "debe mostrar la probabilidad"
