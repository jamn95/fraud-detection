# Resumen CRISP-DM — Fase 5. Evaluación

Fuente: `notebooks/04_evaluacion.ipynb` y `src/evaluate.py`. Figuras: `14_comparacion_metricas.png` a `23_shap_fraude_no_detectado.png`.

**Protocolo.** Los umbrales de decisión y el modelo final se eligieron con el conjunto de validación (56 745 transacciones, 94 fraudes). El conjunto de prueba (56 746 transacciones, 95 fraudes) se usó una sola vez para reportar el desempeño, con intervalos de confianza al 95 % obtenidos por *bootstrap* (1 000 remuestreos). Criterio de selección fijado de antemano: mayor PR-AUC en validación; ante empate (< 0,005), el modelo más simple.

### Política de umbral
Cada alerta cuesta 5 € de revisión y cada fraude no detectado cuesta su monto. El umbral de cada modelo se eligió en validación, minimizando el costo con la condición Precision ≥ 0,70. Para XGBoost el umbral resultó **0,059**, muy por debajo de 0,5: el modelo asigna probabilidades bajas incluso a muchos fraudes, y bajar el umbral recupera casos a un costo de revisión pequeño.

### Resultados en prueba (95 fraudes; umbral de negocio)

| Modelo | Precision | Recall | F1 | PR-AUC | ROC-AUC | Costo (€) | Ahorro |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0,830** | **0,821** | **0,825** | **0,829** | **0,972** | **4 248** | **71,2 %** |
| Stacking (RF + XGB) | 0,750 | 0,821 | 0,784 | 0,817 | 0,971 | 4 302 | 70,9 % |
| Random Forest | 0,745 | 0,800 | 0,772 | 0,813 | 0,967 | 4 297 | 70,9 % |
| DNN | 0,811 | 0,768 | 0,789 | 0,780 | 0,944 | 4 502 | 69,5 % |
| Regresión Logística | 0,787 | 0,779 | 0,783 | 0,695 | 0,956 | 4 278 | 71,0 % |
| Autoencoder | 0,735 | 0,263 | 0,388 | 0,493 | 0,946 | 13 393 | 9,3 % |

Sin modelo, el fraude de prueba cuesta 14 766 €.

### Modelo seleccionado: XGBoost
Elegido con el criterio fijado de antemano: empata con el Stacking en PR-AUC de validación (0,889) y es más simple. En prueba detecta **78 de 95 fraudes** con **16 falsas alarmas** entre 56 651 transacciones legítimas (0,028 %).

| Criterio de éxito (Fase 1) | Resultado en prueba | IC 95 % (bootstrap) | ¿Cumple? |
|---|---|---|---|
| Recall ≥ 0,80 | 0,821 | [0,742; 0,897] | ✔ |
| Precision ≥ 0,70 | 0,830 | [0,750; 0,899] | ✔ |
| PR-AUC ≥ 0,80 | 0,829 | [0,755; 0,901] | ✔ |
| Costo menor que sin modelo | 4 248 € frente a 14 766 € | — | ✔ |

Los intervalos son amplios porque la prueba tiene solo 95 fraudes: el límite inferior del Recall y del PR-AUC queda por debajo de 0,80, así que el cumplimiento debe confirmarse con más datos.

### Hallazgos de la evaluación
1. **El umbral importa más que el balanceo.** Con el umbral de 0,5, XGBoost tiene Recall 0,737 (no cumple el criterio); con el umbral de negocio sube a 0,821, a cambio de 22 alertas más. El umbral que maximiza el F1 favorece la precisión (0,972) pero deja escapar más fraudes (Recall 0,726).
2. **El umbral elegido en validación no es exactamente el óptimo en prueba.** En prueba, el umbral 0,5 da un costo algo menor (4 185 € frente a 4 248 €), una diferencia de 1,5 % debida a la variación entre muestras pequeñas. Se mantiene el umbral de validación para no ajustar con la prueba.
3. **Con y sin SMOTE (prueba).** XGBoost sin balanceo obtiene el mayor PR-AUC (0,833) frente a 0,820 con SMOTE y 0,819 con pesos. En Random Forest, SMOTE mejora el PR-AUC (0,817 frente a 0,790) pero baja la Precision a 0,70. En la DNN, los pesos de clase y SMOTE empeoran el PR-AUC (0,669 y 0,719 frente a 0,800). Se confirma lo observado en validación: SMOTE no es necesario cuando el umbral se optimiza.
4. **ROC-AUC engañoso.** Todos los supervisados superan 0,94 de ROC-AUC, incluso la Regresión Logística, cuyo PR-AUC (0,695) es claramente inferior. La curva PR separa los modelos mucho mejor.
5. **Autoencoder.** Con la restricción Precision ≥ 0,70 su umbral resulta muy alto y solo detecta 26 % de los fraudes. Sin etiquetas no alcanza a los supervisados, pero su ROC-AUC (0,946) muestra que ordena bien las transacciones; es útil como señal complementaria de anomalía.
6. **Sensibilidad al costo de revisión.** El umbral de XGBoost no cambia entre 1 € y 10 € por alerta (ahorro entre 68 % y 74 %). Solo con 20 € o más conviene el umbral más exigente (0,549), con Recall 0,726. La decisión es robusta frente al supuesto de costo.
7. **Fraudes no detectados.** Los 17 fraudes que XGBoost no detecta suman 3 778 € y tienen un monto mediano de 1,79 €. Son en su mayoría transacciones pequeñas (compatibles con *card testing*) cuyas variables se parecen a las legítimas.

### Interpretabilidad (SHAP sobre XGBoost)
Las variables con mayor impacto medio son **V14** (1,16), **V4** (1,11), V8 (0,46), V12 (0,43), V11 (0,33) y V10 (0,30). Coinciden con las más separables del análisis exploratorio (KS de la Fase 2: V14, V10, V12, V4, V11). Valores **bajos** de V14, V12 y V10 y valores **altos** de V4 y V11 empujan la predicción hacia fraude. `Amount_log` y la hora tienen un aporte pequeño: el monto, por sí solo, distingue poco. En el fraude no detectado analizado, un V4 alto empuja hacia fraude, pero V14 (−0,43, un valor normal) lo compensa: el fraude "no se parece" a los fraudes típicos en su variable más importante.

**Artefacto para el despliegue:** `models/config_despliegue.json` (modelo por defecto, umbrales de negocio y F1 de los seis modelos, métricas en prueba, costo de revisión y precisión mínima).
