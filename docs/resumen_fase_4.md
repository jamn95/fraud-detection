# Resumen CRISP-DM — Fase 4. Modelado

Fuente: `notebooks/03_modelado.ipynb` y `src/train_models.py`. Figuras: `11_comparacion_balanceo.png`, `12_entrenamiento_dnn.png`, `13_autoencoder_error.png`.

**Procedimiento.** (1) Cada modelo supervisado se entrenó sin balanceo, con pesos de clase y con SMOTE (proporción 0,1), con hiperparámetros por defecto, y se comparó el PR-AUC en validación. (2) Con la mejor estrategia, los hiperparámetros se ajustaron con `RandomizedSearchCV` (Regresión Logística: 10 combinaciones; Random Forest: 8; XGBoost: 20) y validación cruzada estratificada de 3 pliegues sobre entrenamiento, maximizando el PR-AUC. El escalador y SMOTE forman parte del pipeline y se reajustan en cada pliegue. (3) La DNN se ajustó con una búsqueda aleatoria de 6 configuraciones y *early stopping* sobre el PR-AUC de validación. (4) El autoencoder se entrenó solo con transacciones legítimas y se compararon 4 arquitecturas. (5) El conjunto de prueba no intervino en ninguna decisión. Tiempo total de entrenamiento: 28 minutos (CPU de 12 núcleos lógicos).

### Paso 1 · Estrategia de balanceo (PR-AUC en validación, hiperparámetros por defecto)

| Modelo | Sin balanceo | Pesos de clase | SMOTE | Elegida |
|---|---|---|---|---|
| Regresión Logística | 0,801 | 0,790 | **0,803** | SMOTE |
| Random Forest | 0,874 | 0,875 | **0,875** | SMOTE |
| XGBoost | **0,887** | 0,883 | 0,869 | Sin balanceo |
| DNN | **0,870** | 0,729 | 0,831 | Sin balanceo |

**Hallazgo.** El balanceo casi no cambia el PR-AUC de los modelos de árboles (diferencias menores a 0,02) e incluso lo empeora en XGBoost y, sobre todo, en la DNN con pesos de clase. El PR-AUC mide qué tan bien el modelo **ordena** las transacciones por riesgo; SMOTE y los pesos de clase desplazan sobre todo las probabilidades hacia arriba, lo que equivale a mover el umbral de decisión. Por eso, en la Fase 5, el umbral se optimiza para cada modelo en lugar de usar 0,5. En la DNN, los pesos de clase (≈ 300 veces más peso para el fraude) hacen el entrenamiento inestable: el *early stopping* se detuvo en la época 15.

### Paso 2 · Ajuste de hiperparámetros (`RandomizedSearchCV`, CV estratificada de 3 pliegues)

| Modelo | Mejores hiperparámetros | PR-AUC CV | PR-AUC val |
|---|---|---|---|
| Regresión Logística + SMOTE | `C` = 0,031; `sampling_strategy` = 0,05 | 0,734 ± 0,043 | 0,802 |
| Random Forest + SMOTE | 299 árboles, sin límite de profundidad, `max_features` = 0,3, `min_samples_leaf` = 3, `max_samples` = 0,5; `sampling_strategy` = 0,05 | 0,835 ± 0,062 | 0,880 |
| XGBoost | 543 árboles, `max_depth` = 7, `learning_rate` = 0,083, `subsample` = 0,68, `colsample_bytree` = 0,93, `gamma` = 1,2, `reg_lambda` = 4,6 | 0,837 ± 0,072 | **0,889** |

En ambos modelos con SMOTE la búsqueda eligió la proporción más baja (0,05): pocos sintéticos funcionan mejor que muchos. La desviación entre pliegues (0,04 – 0,07) es alta porque cada pliegue de validación tiene solo unos 95 fraudes; diferencias menores a ~0,02 entre modelos no son concluyentes.

### Modelos adicionales

- **Stacking (RF + XGB):** PR-AUC val = 0,889, igual que XGBoost solo. El meta-modelo reparte el peso de forma parecida entre ambos (coeficientes 6,9 para RF y 5,7 para XGB). La combinación no aporta mejora en ranking, pero cuesta el doble de cómputo.
- **DNN:** la mejor configuración fue la más pequeña, 32-16 con *dropout* 0,4 y `lr` = 10⁻³ (PR-AUC val = 0,873, 1 633 parámetros entrenables). Las redes más grandes no mejoraron, lo que es coherente con tener solo 284 fraudes de entrenamiento. El PR-AUC de entrenamiento queda por debajo del de validación porque durante el entrenamiento el *dropout* está activo.
- **Autoencoder:** la mejor arquitectura fue 31-28-14-7-14-28-31 (PR-AUC val = 0,583, ROC-AUC = 0,943). La mediana del error de reconstrucción de los fraudes es **37 veces mayor** que la de las legítimas, pero una parte de los fraudes se reconstruye tan bien como una transacción normal y queda mezclada con ellas. Es esperable: el autoencoder nunca ve un fraude y no puede aprender qué lo distingue. Su valor es complementario: puede detectar patrones de fraude nuevos que los modelos supervisados no conocen. Las cuatro arquitecturas usaron las 80 épocas, así que con más épocas podría mejorar algo.

### Resumen (validación)

| Modelo | Balanceo | PR-AUC (val) | ROC-AUC (val) |
|---|---|---|---|
| XGBoost | Sin balanceo | **0,889** | 0,977 |
| Stacking (RF + XGB) | Hereda de RF y XGB | **0,889** | 0,971 |
| Random Forest | SMOTE (0,05) | 0,880 | 0,965 |
| DNN | Sin balanceo | 0,873 | 0,965 |
| Regresión Logística | SMOTE (0,05) | 0,802 | 0,980 |
| Autoencoder | Solo legítimas | 0,583 | 0,943 |

La Regresión Logística tiene el ROC-AUC más alto y el PR-AUC más bajo de los supervisados: un ejemplo de por qué el ROC-AUC solo es engañoso con este desbalance.

**Artefactos:** 6 modelos en `models/` (`logistica.joblib`, `random_forest.joblib`, `xgboost.joblib`, `stacking.joblib`, `dnn.keras`, `autoencoder.keras`), `models/registro_modelos.json` con hiperparámetros y métricas, y los puntajes de 18 modelos y variantes en `models/puntajes/` para la Fase 5.
