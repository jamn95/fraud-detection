"""
API REST del detector de fraude (FastAPI).

Ejecutar desde la raíz del proyecto:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
Documentación interactiva: http://localhost:8000/docs
"""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, create_model

from src.config import COLUMNAS_CRUDAS, COLUMNAS_V
from src.predict import DetectorFraude

detector = DetectorFraude()

app = FastAPI(
    title="API de Detección de Fraude",
    description="Clasifica transacciones con tarjeta de crédito (dataset ULB) y devuelve la probabilidad de "
                "fraude, la alerta y el nivel de riesgo.",
    version="1.0.0",
)

# Esquema de entrada: Time, V1..V28 y Amount (se genera dinámicamente para no repetir 28 campos)
Transaccion = create_model(
    "Transaccion",
    Time=(float, Field(..., ge=0, description="Segundos desde la primera transacción del registro")),
    **{v: (float, Field(..., description=f"Componente PCA {v}")) for v in COLUMNAS_V},
    Amount=(float, Field(..., ge=0, description="Monto en euros")),
)


class Resultado(BaseModel):
    puntaje: float = Field(description="Probabilidad de fraude (o error de reconstrucción en el autoencoder)")
    umbral: float
    alerta: bool
    riesgo: str = Field(description="BAJO, MEDIO o ALTO")
    accion: str
    modelo: str


@app.get("/health")
def health() -> dict:
    return {"estado": "ok", "modelo_por_defecto": detector.modelo_defecto}


@app.get("/modelos")
def modelos() -> dict:
    return {"por_defecto": detector.modelo_defecto,
            "modelos": {m: {"umbral": detector.umbrales[m],
                            "metricas_prueba": detector.config["metricas_prueba"].get(m)}
                        for m in detector.modelos_disponibles}}


@app.post("/predict", response_model=list[Resultado])
def predict(transacciones: Transaccion | list[Transaccion],
            modelo: str | None = Query(None, description="Por defecto, el modelo seleccionado en la evaluación")):
    """Recibe una transacción o una lista y devuelve un resultado por transacción."""
    lista = transacciones if isinstance(transacciones, list) else [transacciones]
    if not lista:
        raise HTTPException(status_code=422, detail="La lista de transacciones está vacía")
    datos = pd.DataFrame([t.model_dump() for t in lista])[COLUMNAS_CRUDAS]
    try:
        salida = detector.predecir(datos, modelo)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return salida[["puntaje", "umbral", "alerta", "riesgo", "accion", "modelo"]].to_dict("records")
