# Resumen CRISP-DM — Fase 6. Despliegue

Fuente: `src/predict.py`, `app/` (Streamlit y FastAPI), `Dockerfile`, `docker-compose.yml`, `tests/`. Detalle técnico en `docs/DESPLIEGUE.md`.

**Componentes.**

| Componente | Tecnología | Función |
|---|---|---|
| Motor de inferencia | `src/predict.py` (`DetectorFraude`) | Aplica el mismo preprocesamiento del entrenamiento, calcula el puntaje, compara con el umbral de negocio y asigna nivel de riesgo y acción |
| Aplicación web | Streamlit 1.40 (5 páginas) | Inicio, dashboard de fraude, alertas en tiempo real, transacciones sospechosas y predicción |
| API REST | FastAPI | `POST /predict` (una o varias transacciones), `GET /modelos`, `GET /health` |
| Contenedor | Docker (`python:3.11-slim`) | Imagen única para la app y la API, con usuario sin privilegios y chequeo de salud |
| Configuración | `models/config_despliegue.json` | Modelo por defecto (XGBoost), umbrales y métricas, generados en la Fase 5 |

**Requerimientos funcionales.**

1. **Dashboard de fraude.** KPIs (transacciones, fraudes, alertas, Recall, Precision, falsas alarmas, fraude evitado y ahorro), distribución de montos por clase, fraudes reales y detectados por hora, comparación de los seis modelos (métricas y curvas Precision-Recall), matriz de confusión e interpretabilidad SHAP. Un control de umbral permite ver en vivo el compromiso entre fraudes detectados y falsas alarmas.
2. **Alertas en tiempo real.** Un fragmento de Streamlit con refresco cada segundo procesa transacciones del conjunto de prueba (de 1 a 50 por segundo) y las clasifica sin conocer su etiqueta. Si superan el umbral, se genera una alerta: aviso emergente, banner, marca en el gráfico y fila en la tabla de alertas.
3. **Transacciones sospechosas.** Tabla de las transacciones marcadas por al menos un modelo, con probabilidad, monto, fecha y hora, nivel de riesgo y **qué modelos la marcaron**. Se filtra por modelo, número mínimo de modelos, probabilidad, monto, hora y riesgo, y se descarga en CSV.
4. **Predicción.** Formulario individual (con ejemplos reales, la opinión de los seis modelos y la explicación SHAP de la decisión) y carga de archivos CSV con descarga de resultados.

**Verificación.** 16 pruebas automáticas (`python -m pytest`) comprueban que:
- la inferencia reproduce los puntajes de la evaluación en los seis modelos (diferencia < 10⁻⁵);
- la API responde y valida la entrada (422 si falta un campo, 400 si el modelo no existe);
- las cinco páginas se ejecutan sin errores, el monitoreo procesa lotes y el formulario devuelve una predicción.

La latencia de inferencia, tras la primera carga de cada modelo, es de 10 a 60 ms por transacción.

**Mantenimiento.** Vigilar Recall y Precision con las etiquetas confirmadas (reentrenar si bajan de 0,80 / 0,70), la deriva de las variables más importantes (PSI de V14, V4, V12 y V10) y el volumen de alertas frente a la capacidad del equipo. El pipeline completo se reproduce con cuatro órdenes (`src.data_preprocessing`, `src.train_models`, `src.evaluate`, `pytest`).

**Limitaciones.** La API no incluye autenticación (pensada para red interna). El flujo de la demostración enriquece el fraude a 1,9 % para que se vean alertas. Las fechas son ilustrativas, porque el dataset solo registra segundos relativos.
