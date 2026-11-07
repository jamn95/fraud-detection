# Resumen CRISP-DM — Fase 3. Preparación de los datos

Fuente: `notebooks/02_preparacion_datos.ipynb` y `src/data_preprocessing.py`. Figuras: `09_transformacion_amount.png`, `10_smote_proyeccion.png`.

**Procedimiento.** (1) Eliminación de duplicados exactos antes de dividir. (2) *Feature engineering*: `Amount_log = log(1 + Amount)` y la hora del día en forma cíclica (`Hour_sin`, `Hour_cos`); `Time` crudo se excluye porque es un contador relativo que no generaliza. (3) División estratificada 60/20/20 (entrenamiento, validación y prueba). (4) Escalado ajustado solo con entrenamiento: `StandardScaler` para `V1`–`V28` y `RobustScaler` para `Amount_log`. (5) SMOTE (k = 5) aplicado solo sobre entrenamiento, dentro de un `imblearn.Pipeline` para que en la validación cruzada actúe únicamente sobre los pliegues de entrenamiento.

| Paso | Resultado |
|---|---|
| Limpieza | 1 081 duplicados eliminados (19 fraudes). Quedan **283 726** transacciones y **473** fraudes |
| Variables | 31 variables: `V1`–`V28`, `Amount_log`, `Hour_sin`, `Hour_cos`. La asimetría del monto baja de 16,98 a 0,16 |
| División | Entrenamiento 170 235 (284 fraudes) · Validación 56 745 (94) · Prueba 56 746 (95). Proporción de fraude ≈ 0,167 % en los tres. Ninguna transacción se repite entre conjuntos |
| Escalado | Ajustado solo con entrenamiento. Validación queda con desviación ≈ 1 sin haber participado en el ajuste |
| SMOTE | Probado con proporciones de 0,01 a 1,0. Con 0,1 se crean 16 711 fraudes sintéticos (59 por cada fraude real); con 1,0 serían 169 667 (597 por fraude real). La proporción final se elige en la Fase 4 |

**Experimento de fuga de datos.** Un mismo XGBoost obtiene PR-AUC = **0,815** y Recall = 0,768 cuando SMOTE se aplica correctamente (solo en entrenamiento), y PR-AUC = **1,000** y Recall = 1,000 cuando SMOTE se aplica antes de dividir. Las métricas "perfectas" son falsas: la prueba contaminada tiene 33,3 % de fraudes (frente al 0,167 % real) y casi todos son sintéticos derivados de fraudes que el modelo vio al entrenar. Este resultado justifica aplicar SMOTE dentro de un `imblearn.Pipeline`.

**Artefactos para la Fase 4.**
- `data/processed/train.parquet`, `val.parquet`, `test.parquet` (sin escalar, con todas las columnas).
- `models/preprocesador.joblib` (escalador ajustado con entrenamiento).
- `data/processed/metadata_preparacion.json`.
- `pipeline_con_balanceo(modelo, sampling_strategy)` en `src/data_preprocessing.py`: escalador → SMOTE (opcional) → modelo.
