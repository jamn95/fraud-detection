# Despliegue — Detección de Fraude Financiero

## 1. Ejecución local

Desde la raíz del proyecto, con el entorno `.venv` activo:

```powershell
streamlit run app/app.py --server.address localhost   # app web -> http://localhost:8501
uvicorn app.api:app --port 8000               # API REST  -> http://localhost:8000/docs
python -m pytest                              # pruebas (predictor, API y páginas)
```

## 2. Docker

```bash
docker compose up --build                     # app en :8501 y API en :8000
```

O solo la imagen:

```bash
docker build -t deteccion-fraude .
docker run -p 8501:8501 deteccion-fraude
docker run -p 8000:8000 deteccion-fraude uvicorn app.api:app --host 0.0.0.0 --port 8000
```

La imagen (`python:3.11-slim`) contiene solo lo necesario para servir: `src/`, `app/`, `models/`, `data/demo/`, `reports/figures/` y `.streamlit/`. Corre con un usuario sin privilegios e incluye un chequeo de salud.

## 3. Streamlit Community Cloud

1. Subir el proyecto a un repositorio de GitHub. Deben incluirse `models/` (≈ 24 MB con los puntajes) y `data/demo/transacciones_demo.parquet` (1,4 MB). `data/raw/` y `data/processed/` quedan fuera por `.gitignore`.
2. En [share.streamlit.io](https://share.streamlit.io): **New app** → repositorio y rama → *Main file path* = `app/app.py`.
3. En **Advanced settings**, elegir **Python 3.11** (TensorFlow 2.18 no admite versiones más nuevas).
4. Cloud instala `app/requirements.txt` (lo busca primero junto al script principal).

## 4. Arquitectura

```
                 ┌──────────────────────── models/ ────────────────────────┐
                 │ config_despliegue.json  registro_modelos.json            │
                 │ preprocesador.joblib    *.joblib (pipelines)  *.keras    │
                 │ puntajes/puntajes_test.parquet                           │
                 └───────────────▲──────────────────────────▲──────────────┘
                                 │                          │
 transacción cruda      ┌────────┴─────────┐                │
 (Time, V1..V28, ──────►│ src/predict.py   │                │
  Amount)               │ DetectorFraude   │                │
                        │ 1. crear_variables (Amount_log, Hour_sin/cos)
                        │ 2. escalado (dentro del pipeline o preprocesador)
                        │ 3. puntaje del modelo                │
                        │ 4. umbral de negocio → alerta, riesgo, acción
                        └───▲──────────▲───┘                │
                            │          │                    │
              ┌─────────────┴──┐   ┌───┴──────────────────────┴───┐
              │ app/api.py     │   │ app/app.py (Streamlit)        │
              │ FastAPI        │   │  Inicio · Dashboard · Alertas │
              │ POST /predict  │   │  en tiempo real · Sospechosas │
              │ GET /modelos   │   │  · Predicción (form + CSV)    │
              └────────────────┘   └───────────────────────────────┘
```

- **Un único punto de inferencia** (`DetectorFraude`): la app y la API aplican exactamente el mismo preprocesamiento que el entrenamiento. Una prueba automática verifica que los puntajes coinciden con los de la evaluación (diferencia < 10⁻⁵).
- **Configuración separada del código**: el modelo por defecto y los umbrales se leen de `models/config_despliegue.json`, generado en la Fase 5. Cambiar de modelo o de umbral no requiere tocar la app.
- **Carga perezosa y caché**: cada modelo se carga la primera vez que se usa (`st.cache_resource` mantiene un solo detector por servidor). Latencia medida tras la primera carga: 10–60 ms por transacción.

## 5. Monitoreo en tiempo real

La página **Alertas en tiempo real** simula un flujo de transacciones:

1. Toma las transacciones de `data/demo/transacciones_demo.parquet` (los 95 fraudes del conjunto de prueba y 4 905 legítimas, en orden aleatorio). El fraude se enriquece a ≈ 1,9 % para que la demostración muestre alertas; en la realidad es ≈ 0,17 %.
2. Un fragmento de Streamlit con `run_every = 1 s` se vuelve a ejecutar cada segundo mientras la simulación está activa. En cada ciclo clasifica un lote (1 a 50 transacciones por segundo, según la velocidad elegida), **sin conocer su etiqueta**.
3. Si el puntaje supera el umbral del modelo, la transacción genera una **alerta**: aviso emergente (`st.toast`), banner rojo, punto naranja en el gráfico y una fila en la tabla de alertas, con su nivel de riesgo y la acción recomendada.
4. El estado (posición, historial y alertas) vive en `st.session_state`; *Pausar* detiene el refresco y *Reiniciar* vacía el historial.

Niveles de riesgo: **ALTO** si el puntaje ≥ umbral (bloquear y contactar al titular), **MEDIO** si ≥ 50 % del umbral (revisión manual o 3-D Secure), **BAJO** en otro caso (aprobar).

En producción, el mismo `DetectorFraude.predecir()` se conectaría a la fuente real de transacciones (una cola como Kafka, o la API `POST /predict` llamada por el sistema de autorización de pagos).

## 6. Plan de mantenimiento

| Riesgo | Señal a vigilar | Acción |
|---|---|---|
| *Concept drift* (cambian los patrones de fraude) | Recall y Precision semanales con las etiquetas confirmadas por el equipo de revisión y los contracargos | Reentrenar con datos recientes si el Recall cae por debajo de 0,80 o la Precision por debajo de 0,70 |
| *Data drift* (cambia la distribución de las variables) | PSI (*Population Stability Index*) o prueba KS de cada variable frente al entrenamiento | Investigar si PSI > 0,2 en variables importantes (V14, V4, V12, V10) |
| Volumen de alertas | Alertas por día frente a la capacidad del equipo | Ajustar el umbral con `src/evaluate.py` (curva de costo) |
| Cambio del costo de revisión | Costo operativo real por alerta | Recalcular el umbral de negocio; la sensibilidad de la Fase 5 muestra que es estable entre 1 € y 10 € |
| Fraudes nuevos no etiquetados | Transacciones con error alto en el autoencoder pero baja probabilidad supervisada | Enviarlas a revisión y añadirlas al próximo entrenamiento |

**Procedimiento de reentrenamiento** (reproduce todo el pipeline):

```powershell
python -m src.data_preprocessing   # limpieza, variables, división, escalado
python -m src.train_models         # modelos y puntajes
python -m src.evaluate             # umbrales, selección y config_despliegue.json
python -m pytest                   # verifica el despliegue antes de publicar
```

Se recomienda versionar cada conjunto de modelos con la fecha (`config_despliegue.json` ya la registra) y conservar el anterior para poder volver atrás.

## 7. API REST

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y modelo por defecto |
| GET | `/modelos` | Modelos disponibles, umbrales y métricas en prueba |
| POST | `/predict?modelo=XGBoost` | Una transacción (objeto) o varias (lista) |

Ejemplo con un fraude real del conjunto de prueba (transacción de 0,68 €):

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json"   -d '{"Time": 29531, "V1": -1.06, "V2": 2.61, "V3": -2.97, "V4": 4.36, "V5": 3.74, "V6": -2.73,
       "V7": 1.99, "V8": -0.36, "V9": -2.76, "V10": -2.34, "V11": 2.11, "V12": -2.59, "V13": -1.95,
       "V14": -7.31, "V15": 0.08, "V16": 2.29, "V17": 6.51, "V18": 2.57, "V19": -3.29, "V20": -0.09,
       "V21": -0.06, "V22": -0.21, "V23": -0.18, "V24": -0.1, "V25": 0.9, "V26": 0.41, "V27": -0.13,
       "V28": 0.19, "Amount": 0.68}'
```

```json
[{"puntaje": 0.333, "umbral": 0.0589, "alerta": true, "riesgo": "ALTO",
  "accion": "Bloquear la transacción y contactar al titular", "modelo": "XGBoost"}]
```

La respuesta es una lista con un resultado por transacción: `puntaje`, `umbral`, `alerta`, `riesgo`, `accion`, `modelo`. Si falta un campo responde 422; si el modelo no existe, 400.

## 8. Seguridad y limitaciones

- La API no tiene autenticación: está pensada para una red interna. Antes de exponerla a Internet, agregar autenticación (por ejemplo, una clave en cabecera o OAuth2), HTTPS y límite de peticiones.
- En local, ejecutar la app con `--server.address localhost` para que solo sea accesible desde el propio equipo; en Docker se abre en `0.0.0.0` para poder publicar el puerto.
- Las fechas que muestra la app son ilustrativas: el dataset solo contiene segundos desde la primera transacción.
- La columna "clase real" de las páginas de demostración existe porque se usan datos de prueba etiquetados; en producción no se conoce al momento de clasificar.
