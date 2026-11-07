<!--
Contenido del artículo para volcar en el Google Doc "Copia de Articulo_Cientifico"
(https://docs.google.com/document/d/1nMGP6eQwzKsiuYcLDC9Loc3Ly1y5LJBeInhZnQjyOVw).
Se conserva la portada del documento (estudiantes, institución, fecha). Desde "Resumen"
este texto reemplaza el existente. Las figuras indicadas están en reports/figures/.
Citas en APA 7.
-->

# Detección de fraude financiero con técnicas de *machine learning* y aprendizaje profundo bajo la metodología CRISP-DM

**Resumen:** El fraude con tarjetas de crédito genera pérdidas millonarias cada año al comercio y al sector bancario, por lo que es necesario detectar las transacciones fraudulentas casi en tiempo real y bloquearlas sin perjudicar la experiencia de los clientes legítimos. Este trabajo desarrolla, bajo la metodología CRISP-DM, un sistema inteligente que estima la probabilidad de fraude de cada transacción a partir del *Credit Card Fraud Detection Dataset* (284 807 transacciones, 0,173 % de fraude). Tras eliminar duplicados, se crearon variables derivadas (monto logarítmico y hora del día en forma cíclica) y se dividieron los datos de forma estratificada en entrenamiento, validación y prueba; el balanceo con SMOTE se aplicó solo sobre el entrenamiento, dentro de un *pipeline*, para evitar la fuga de información. Se compararon seis modelos: Regresión Logística, Random Forest, XGBoost, *Stacking* (Random Forest + XGBoost), una red neuronal profunda (DNN) y un *autoencoder* para detección de anomalías. El umbral de decisión de cada modelo se eligió en validación minimizando el costo para el negocio. XGBoost fue el modelo seleccionado: en el conjunto de prueba detectó el 82,1 % de los fraudes con una precisión del 83,0 % (PR-AUC = 0,829) y redujo el costo del fraude en un 71,2 % frente a no usar ningún modelo. El análisis SHAP identificó a las componentes V14 y V4 como las variables más influyentes, mientras que el monto aportó poco a la predicción. Además, se observó que optimizar el umbral de decisión mejora más los resultados que aplicar SMOTE. La solución se desplegó como una aplicación web con un *dashboard* de fraude, alertas en tiempo real, visualización de transacciones sospechosas y una API REST.

**Palabras clave:** Detección de fraude, Aprendizaje automático, CRISP-DM, XGBoost, SMOTE, Desbalance de clases, SHAP.

## 1. Introducción

A lo largo de los años, la inteligencia artificial ha evolucionado como herramienta de optimización y automatización de procesos, con metodologías aplicadas en áreas muy diversas. En este contexto, las técnicas de *machine learning* han experimentado un gran crecimiento, tanto en el ámbito académico como en el empresarial, y se han convertido en un factor de transformación importante (Sadgali et al., 2019).

Las instituciones financieras procesan volúmenes de transacciones tan grandes que la revisión manual de cada operación es inviable. Por ello necesitan sistemas automáticos capaces de analizar cada transacción, detectar las sospechosas y prevenir el fraude antes de que se complete el pago. El problema presenta dos dificultades. La primera es el **desbalance extremo de clases**: el fraude es un evento raro, de modo que un modelo que nunca lo detecte puede tener una exactitud (*accuracy*) cercana al 100 % y, aun así, no servir de nada. La segunda es la **asimetría de costos**: dejar pasar un fraude cuesta el monto de la transacción, mientras que una falsa alarma cuesta una revisión manual y una molestia para el cliente (Dal Pozzolo et al., 2018; Elkan, 2001).

Esta investigación evalúa la efectividad de modelos de *bagging* (Random Forest), *boosting* (XGBoost), redes neuronales profundas y *autoencoders* para la detección de fraude, y los compara con una Regresión Logística como línea base. Para ello se emplean las fases de la metodología CRISP-DM (Chapman et al., 2000), que garantizan un proceso ordenado, reproducible y orientado al negocio, desde la comprensión del problema hasta el despliegue de la solución.

### 1.1 Objetivos

**Objetivo general**

Desarrollar un sistema inteligente para detectar transacciones fraudulentas utilizando técnicas avanzadas de *machine learning* y aprendizaje profundo.

**Objetivos específicos**

1. Analizar el *dataset* de transacciones financieras mediante técnicas de comprensión del negocio y preparación de datos, para identificar patrones asociados al fraude.
2. Desarrollar modelos de *machine learning* y aprendizaje profundo (Random Forest, XGBoost, redes neuronales y *autoencoders*) para detectar transacciones fraudulentas, comparando estrategias de balanceo de clases.
3. Evaluar y comparar el rendimiento de los modelos mediante Precision, Recall, F1, ROC-AUC y PR-AUC, con el propósito de identificar el modelo más eficiente para la detección de fraude.
4. Desplegar el modelo seleccionado en una aplicación web con *dashboard*, alertas en tiempo real y visualización de transacciones sospechosas.

## 2. Metodología

La investigación siguió la metodología CRISP-DM (*Cross-Industry Standard Process for Data Mining*), que organiza un proyecto de minería de datos en seis fases: comprensión del negocio, comprensión de los datos, preparación de los datos, modelado, evaluación y despliegue (Chapman et al., 2000; Wirth y Hipp, 2000). Todo el proceso se implementó en Python 3.11 con scikit-learn (Pedregosa et al., 2011), imbalanced-learn (Lemaître et al., 2017), XGBoost (Chen y Guestrin, 2016), TensorFlow/Keras y SHAP (Lundberg y Lee, 2017). Todas las fuentes de aleatoriedad se fijaron con la semilla 42 para garantizar la reproducibilidad.

### 2.1 Comprensión del negocio

El objetivo de negocio es reducir la pérdida económica por fraude manteniendo un volumen de alertas que el equipo de revisión pueda atender. El objetivo de minería de datos es construir un clasificador binario que estime, para cada transacción, la probabilidad de que sea fraude. Se definió el siguiente modelo de costo: cada alerta (verdadera o falsa) implica una revisión manual de 5 €, y cada fraude no detectado cuesta su monto. Se fijaron como criterios de éxito, medidos en el conjunto de prueba: Recall ≥ 0,80, Precision ≥ 0,70, PR-AUC ≥ 0,80 y un costo total menor que el de operar sin modelo. El *accuracy* se descartó como criterio por el desbalance de clases.

### 2.2 Conjunto de datos y comprensión de los datos

Se utilizó el *Credit Card Fraud Detection Dataset* (Dal Pozzolo et al., 2015), que contiene 284 807 transacciones realizadas por titulares europeos durante dos días de septiembre de 2013. Incluye 28 variables numéricas (V1–V28) obtenidas mediante Análisis de Componentes Principales (PCA) para proteger la confidencialidad de los clientes, además del tiempo transcurrido desde la primera transacción (`Time`, en segundos), el monto (`Amount`, en euros) y la clase (`Class`: 1 = fraude). Solo 492 transacciones son fraudulentas (0,173 %), es decir, una por cada 578 legítimas (Fig. 1).

El análisis exploratorio mostró que:

- No hay valores nulos, pero sí 1 081 filas duplicadas (19 de ellas fraudes), sin etiquetas contradictorias.
- El monto es muy asimétrico (asimetría de 16,98). La mediana del fraude (9,25 €) es menor que la de las legítimas (22,00 €), y el 25 % de los fraudes es de 1 € o menos, un patrón compatible con pruebas de tarjetas robadas (*card testing*). La diferencia entre las distribuciones es significativa (prueba de Mann-Whitney, p = 8,6 × 10⁻⁶).
- La tasa de fraude se dispara de madrugada: 1,71 % a las 2 h, unas diez veces la tasa global (Fig. 2).
- Las componentes V1–V28 no están correlacionadas entre sí, y su correlación lineal con la clase es débil (máximo 0,33 en V17). Sin embargo, el estadístico de Kolmogórov-Smirnov revela una separación fuerte y no lineal en V14 (0,84), V10 (0,80), V12 (0,78), V4 (0,77) y V11 (0,76) (Fig. 3).
- El 97,0 % de los fraudes tiene al menos una variable atípica (criterio de 1,5·IQR), frente al 48,5 % de las legítimas. Por eso los valores atípicos no se eliminaron: son la señal del fraude.

**Figura 1.** Desbalance de clases (escala logarítmica). *Archivo: `01_desbalance_clases.png`.*

**Figura 2.** Volumen de transacciones y tasa de fraude por hora del día. *Archivo: `04_patron_horario.png`.*

**Figura 3.** Densidad de las seis variables con mayor estadístico KS, por clase. *Archivo: `07_variables_mas_separables.png`.*

### 2.3 Preparación de los datos

1. **Limpieza:** se eliminaron los 1 081 duplicados antes de dividir los datos, para que una misma transacción no quedara a la vez en entrenamiento y en prueba. Quedaron 283 726 transacciones y 473 fraudes.
2. **Variables derivadas:** `Amount_log = log(1 + Amount)`, que reduce la asimetría del monto de 16,98 a 0,16, y la hora del día en forma cíclica (`Hour_sin`, `Hour_cos`). `Time` no se usó directamente, porque es un contador relativo que en producción tomaría valores nunca vistos.
3. **División estratificada 60/20/20:** entrenamiento (170 235 transacciones, 284 fraudes), validación (56 745; 94) y prueba (56 746; 95). La validación se usó para el *early stopping* y para elegir los umbrales; la prueba, solo para la evaluación final.
4. **Escalado ajustado solo con entrenamiento:** `StandardScaler` para V1–V28 y `RobustScaler` para `Amount_log`.
5. **Balanceo:** SMOTE (Chawla et al., 2002) genera fraudes sintéticos interpolando entre fraudes reales cercanos. Se aplicó solo sobre el entrenamiento, dentro de un `imblearn.Pipeline`, de modo que en la validación cruzada actúa únicamente sobre los pliegues de entrenamiento (Fig. 4).

Para dimensionar el riesgo de fuga de información (Kaufman et al., 2012) se entrenó un mismo XGBoost de dos formas. Aplicando SMOTE correctamente se obtuvo un PR-AUC de 0,815; aplicándolo antes de dividir los datos, el PR-AUC fue de 1,000. Ese resultado "perfecto" es falso: la prueba contaminada contenía un 33 % de fraudes, casi todos sintéticos y derivados de fraudes que el modelo ya había visto.

**Figura 4.** Fraudes reales y sintéticos generados por SMOTE en el plano V14–V10. *Archivo: `10_smote_proyeccion.png`.*

### 2.4 Modelado

Se entrenaron seis modelos: Regresión Logística (línea base), Random Forest (Breiman, 2001), XGBoost (Chen y Guestrin, 2016), *Stacking* de Random Forest y XGBoost con un meta-modelo logístico (Wolpert, 1992), una red neuronal profunda (DNN) con capas ReLU, *Batch Normalization* y *Dropout*, y un *autoencoder* entrenado solo con transacciones legítimas, cuyo error de reconstrucción se usa como puntaje de anomalía (Pumsirirat y Yan, 2018).

El modelado siguió dos pasos:

1. **Estrategia de balanceo:** cada modelo supervisado se entrenó sin balanceo, con pesos de clase (`class_weight` o `scale_pos_weight`) y con SMOTE, y se comparó el PR-AUC en validación.
2. **Ajuste de hiperparámetros:** con la mejor estrategia de cada modelo se ejecutó `RandomizedSearchCV` con validación cruzada estratificada de tres pliegues sobre el entrenamiento, maximizando el PR-AUC. Para la DNN se probaron seis configuraciones de capas, *dropout* y tasa de aprendizaje, y para el *autoencoder*, cuatro arquitecturas.

Se eligió el PR-AUC (*Average Precision*) como métrica de selección porque, con clases muy desbalanceadas, la curva ROC puede parecer excelente aunque el modelo genere muchas falsas alarmas (Saito y Rehmsmeier, 2015).

### 2.5 Evaluación

El umbral de decisión de cada modelo se eligió en validación como el que minimiza el costo total con la restricción Precision ≥ 0,70. El mejor modelo se seleccionó con un criterio fijado de antemano: mayor PR-AUC en validación y, ante un empate (diferencia menor a 0,005), el modelo más simple. Las métricas finales se midieron una sola vez en el conjunto de prueba, con intervalos de confianza del 95 % obtenidos por *bootstrap* (1 000 remuestreos). La interpretabilidad se analizó con SHAP sobre el XGBoost.

### 2.6 Despliegue

El modelo se desplegó en una aplicación web desarrollada con Streamlit, con cinco secciones: inicio, *dashboard* de fraude, alertas en tiempo real, transacciones sospechosas y predicción (individual o por archivo CSV). Además, se publicó una API REST con FastAPI (`POST /predict`). La aplicación y la API comparten un mismo motor de inferencia, que aplica exactamente el preprocesamiento del entrenamiento; 16 pruebas automáticas verifican que los puntajes en producción coinciden con los de la evaluación. La solución se empaquetó en Docker y se publicó en Streamlit Community Cloud.

## 3. Resultados

### 3.1 Estrategias de balanceo

La Tabla 1 muestra el PR-AUC en validación de cada modelo con las tres estrategias de balanceo y los hiperparámetros por defecto (Fig. 5).

**Tabla 1.** PR-AUC en validación según la estrategia de balanceo.

| Modelo | Sin balanceo | Pesos de clase | SMOTE |
|---|---|---|---|
| Regresión Logística | 0,801 | 0,790 | **0,803** |
| Random Forest | 0,874 | 0,875 | **0,875** |
| XGBoost | **0,887** | 0,883 | 0,869 |
| DNN | **0,870** | 0,729 | 0,831 |

En los modelos de árboles, las diferencias entre estrategias son menores a 0,02. En la DNN, los pesos de clase redujeron el PR-AUC de 0,870 a 0,729. Cuando se usó SMOTE, la búsqueda de hiperparámetros eligió la proporción más baja probada (0,05 fraudes sintéticos por legítima) tanto en Random Forest como en la Regresión Logística.

**Figura 5.** Efecto de la estrategia de balanceo en el PR-AUC de validación. *Archivo: `11_comparacion_balanceo.png`.*

### 3.2 Modelos ajustados

Los mejores hiperparámetros fueron los siguientes:

- **XGBoost:** sin balanceo, 543 árboles, profundidad máxima 7 y tasa de aprendizaje 0,083.
- **Random Forest:** SMOTE con proporción 0,05, 299 árboles y `max_features` = 0,3.
- **DNN:** arquitectura 32-16 con *dropout* 0,4.
- **Autoencoder:** arquitectura 31-28-14-7-14-28-31.

En validación, XGBoost y el *Stacking* empataron en el mejor PR-AUC (0,889). Les siguieron Random Forest (0,880), la DNN (0,873), la Regresión Logística (0,802) y el *autoencoder* (0,583).

### 3.3 Desempeño en el conjunto de prueba

La Tabla 2 resume el desempeño de cada modelo en el conjunto de prueba (95 fraudes), con el umbral de negocio elegido en validación. Sin modelo, el fraude de prueba cuesta 14 766 €.

**Tabla 2.** Desempeño en el conjunto de prueba.

| Modelo | Precision | Recall | F1 | PR-AUC | ROC-AUC | Costo (€) | Ahorro |
|---|---|---|---|---|---|---|---|
| **XGBoost** | **0,830** | **0,821** | **0,825** | **0,829** | **0,972** | **4 248** | **71,2 %** |
| Stacking (RF + XGB) | 0,750 | 0,821 | 0,784 | 0,817 | 0,971 | 4 302 | 70,9 % |
| Random Forest | 0,745 | 0,800 | 0,772 | 0,813 | 0,967 | 4 297 | 70,9 % |
| DNN | 0,811 | 0,768 | 0,789 | 0,780 | 0,944 | 4 502 | 69,5 % |
| Regresión Logística | 0,787 | 0,779 | 0,783 | 0,695 | 0,956 | 4 278 | 71,0 % |
| Autoencoder | 0,735 | 0,263 | 0,388 | 0,493 | 0,946 | 13 393 | 9,3 % |

Por el criterio de selección, el modelo elegido fue **XGBoost**: empató con el *Stacking* en validación y es más simple. Con un umbral de 0,059, en prueba detectó 78 de los 95 fraudes y generó 16 falsas alarmas entre 56 651 transacciones legítimas (0,028 %). Cumplió los cuatro criterios de éxito (Tabla 3). Las curvas Precision-Recall y las matrices de confusión se muestran en las Figuras 6 y 7.

**Tabla 3.** Criterios de éxito del modelo seleccionado (XGBoost) en prueba.

| Criterio | Resultado | IC 95 % (bootstrap) | ¿Cumple? |
|---|---|---|---|
| Recall ≥ 0,80 | 0,821 | [0,742; 0,897] | Sí |
| Precision ≥ 0,70 | 0,830 | [0,750; 0,899] | Sí |
| PR-AUC ≥ 0,80 | 0,829 | [0,755; 0,901] | Sí |
| Costo menor que sin modelo | 4 248 € frente a 14 766 € | — | Sí |

**Figura 6.** Curvas Precision-Recall en prueba; el punto marca el umbral de negocio de cada modelo. *Archivo: `16_curvas_pr.png`.*

**Figura 7.** Matrices de confusión en prueba con el umbral de negocio. *Archivo: `17_matrices_confusion.png`.*

### 3.4 Efecto del umbral y sensibilidad al costo

Con el umbral convencional de 0,5, XGBoost obtuvo un Recall de 0,737. El umbral que maximiza el F1 dio una Precision de 0,972 y un Recall de 0,726. Con el umbral de negocio, el Recall subió a 0,821, a cambio de 22 alertas más (Fig. 8). El umbral de negocio no cambió para costos de revisión entre 1 € y 10 € por alerta, con ahorros entre el 68,1 % y el 73,8 %. Solo desde 20 € por alerta resultó óptimo un umbral más exigente (0,549).

En la comparación de estrategias de balanceo sobre la prueba, XGBoost sin balanceo obtuvo el mayor PR-AUC (0,833), frente a 0,820 con SMOTE y 0,819 con pesos de clase.

**Figura 8.** Costo total, Precision y Recall en validación según el umbral de XGBoost. *Archivo: `19_costo_umbral.png`.*

### 3.5 Interpretabilidad

Según el valor SHAP medio absoluto, las variables más influyentes fueron V14 (1,16), V4 (1,11), V8 (0,46), V12 (0,43), V11 (0,33) y V10 (0,30) (Fig. 9). Los valores bajos de V14, V12 y V10 y los valores altos de V4 y V11 empujan la predicción hacia fraude. `Amount_log` y las variables horarias no figuraron entre las quince más influyentes. Los 17 fraudes que XGBoost no detectó suman 3 778 € y tienen un monto mediano de 1,79 €.

**Figura 9.** Resumen SHAP del XGBoost: impacto de cada variable en la predicción. *Archivo: `20_shap_resumen.png`.*

## 4. Discusión

Los resultados confirman que, en detección de fraude, la elección de la métrica es tan importante como la del modelo. Todos los modelos supervisados superaron un ROC-AUC de 0,94, incluida la Regresión Logística. Sin embargo, el PR-AUC de esta última (0,695) quedó claramente por debajo del de XGBoost (0,829). Esto coincide con Saito y Rehmsmeier (2015): con clases muy desbalanceadas, la curva ROC se infla por el gran número de verdaderos negativos, mientras que la curva Precision-Recall refleja el costo real de las falsas alarmas.

Un hallazgo relevante es que **SMOTE no fue necesario**. En los modelos de árboles, el balanceo cambió el PR-AUC en menos de 0,02, y en XGBoost el mejor resultado se obtuvo sin balanceo. SMOTE y los pesos de clase desplazan sobre todo las probabilidades hacia la clase minoritaria, lo que equivale a mover el umbral de decisión. Por eso, una vez optimizado el umbral según el costo, su aporte desaparece. Este resultado es coherente con la literatura de aprendizaje sensible al costo, según la cual el umbral óptimo depende de la relación entre los costos de los errores (Elkan, 2001). En cambio, en la DNN, los pesos de clase (unas 300 veces más peso para el fraude) desestabilizaron el entrenamiento.

El **umbral pesó más que la técnica de balanceo**. Con el umbral convencional de 0,5, XGBoost no habría alcanzado el Recall exigido; con el umbral de negocio sí lo alcanzó, a cambio de revisar 22 transacciones más. El umbral elegido en validación no fue exactamente el óptimo en prueba: el umbral 0,5 habría costado un 1,5 % menos. Se mantuvo, sin embargo, el umbral de validación para no ajustar decisiones con los datos de prueba. Esa diferencia se explica por la variación entre muestras con pocos fraudes.

El experimento de fuga de información muestra por qué el orden de la preparación es crítico. Aplicar SMOTE antes de dividir los datos produjo métricas "perfectas" (PR-AUC = 1,000) que no se sostendrían en producción, un error metodológico que Kaufman et al. (2012) describen como fuga de datos.

El *Stacking* no superó a XGBoost solo y duplicó el costo de entrenamiento, por lo que no se justifica su complejidad adicional. El *autoencoder* fue el modelo más débil como detector principal (Recall de 0,263 con Precision ≥ 0,70), porque nunca ve un fraude durante el entrenamiento. Aun así, su ROC-AUC de 0,946 muestra que ordena bien las transacciones, por lo que tiene valor como señal complementaria para detectar patrones de fraude nuevos.

El análisis SHAP coincidió con el análisis exploratorio: las variables con mayor impacto (V14, V4, V12, V10 y V11) son las mismas que mostraron mayor separabilidad según el estadístico KS. El monto aportó poco a la predicción, y los fraudes no detectados fueron sobre todo de montos muy pequeños, compatibles con pruebas de tarjetas robadas.

**Limitaciones.** El *dataset* abarca solo dos días de 2013, por lo que no permite evaluar el *concept drift*: los patrones de fraude cambian con el tiempo, y en producción el modelo debe reentrenarse periódicamente (Dal Pozzolo et al., 2018). Las variables V1–V28 son anónimas, lo que impide interpretar su significado de negocio. El conjunto de prueba tiene solo 95 fraudes, de modo que los límites inferiores de los intervalos de confianza del Recall (0,742) y del PR-AUC (0,755) quedan por debajo de 0,80; el cumplimiento de estos criterios debe confirmarse con más datos. Por último, el costo de revisión de 5 € por alerta es un supuesto, aunque el análisis de sensibilidad muestra que la decisión es estable entre 1 € y 10 €.

**Trabajo futuro.** Se propone validar el modelo con datos posteriores al entrenamiento (validación temporal), combinar el puntaje del *autoencoder* con el de XGBoost en un sistema híbrido, incorporar variables de comportamiento del cliente (frecuencia de uso, comercios habituales) y monitorear en producción la deriva de las variables más importantes.

## 5. Conclusiones

Se desarrolló, bajo la metodología CRISP-DM, un sistema inteligente de detección de fraude que cumple el objetivo general del proyecto. XGBoost fue el modelo seleccionado: en un conjunto de prueba nunca usado durante el desarrollo, detectó el 82,1 % de los fraudes con una precisión del 83,0 % (PR-AUC = 0,829), generó falsas alarmas en solo el 0,028 % de las transacciones legítimas y redujo el costo del fraude en un 71,2 % frente a no usar ningún modelo. Así, se cumplieron los cuatro criterios de éxito definidos en la comprensión del negocio.

Respecto de los objetivos específicos: el análisis de los datos reveló un desbalance de 1 fraude por cada 578 transacciones, una concentración del fraude en la madrugada y en montos pequeños, y una fuerte separabilidad no lineal en las componentes V14, V10, V12 y V4. De los seis modelos comparados, los de árboles (XGBoost, *Stacking* y Random Forest) obtuvieron el mejor desempeño, seguidos de la DNN. El *autoencoder* resultó útil solo como señal complementaria. La evaluación mostró que el PR-AUC discrimina mejor entre modelos que el ROC-AUC, y que optimizar el umbral según el costo de negocio aporta más que aplicar SMOTE.

Finalmente, la solución se desplegó como una aplicación web con *dashboard* de fraude, alertas en tiempo real, visualización de transacciones sospechosas, predicción explicada con SHAP y una API REST, lista para integrarse en un proceso de monitoreo de transacciones.

## Referencias

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chapman, P., Clinton, J., Kerber, R., Khabaza, T., Reinartz, T., Shearer, C., y Wirth, R. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS Inc.

Chawla, N. V., Bowyer, K. W., Hall, L. O., y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., y Bontempi, G. (2018). Credit card fraud detection: A realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning Systems, 29*(8), 3784–3797. https://doi.org/10.1109/TNNLS.2017.2736643

Dal Pozzolo, A., Caelen, O., Johnson, R. A., y Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. En *2015 IEEE Symposium Series on Computational Intelligence* (pp. 159–166). IEEE. https://doi.org/10.1109/SSCI.2015.33

Elkan, C. (2001). The foundations of cost-sensitive learning. En *Proceedings of the 17th International Joint Conference on Artificial Intelligence* (pp. 973–978). Morgan Kaufmann.

Kaufman, S., Rosset, S., Perlich, C., y Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), Artículo 15. https://doi.org/10.1145/2382577.2382579

Lemaître, G., Nogueira, F., y Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research, 18*(17), 1–5.

Lundberg, S. M., y Lee, S.-I. (2017). A unified approach to interpreting model predictions. En *Advances in Neural Information Processing Systems 30* (pp. 4765–4774). Curran Associates.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Pumsirirat, A., y Yan, L. (2018). Credit card fraud detection using deep learning based on auto-encoder and restricted Boltzmann machine. *International Journal of Advanced Computer Science and Applications, 9*(1), 18–25. https://doi.org/10.14569/IJACSA.2018.090103

Sadgali, I., Sael, N., y Benabbou, F. (2019). Performance of machine learning techniques in the detection of financial frauds. *Procedia Computer Science, 148*, 45–54. https://doi.org/10.1016/j.procs.2019.01.007

Saito, T., y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Wirth, R., y Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. En *Proceedings of the 4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining* (pp. 29–39).

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259. https://doi.org/10.1016/S0893-6080(05)80023-1

## Anexo. Entregables

| Entregable | Ubicación |
|---|---|
| Código fuente completo y diseño web | https://github.com/jamn95/fraud-detection |
| Aplicación desplegada | https://deteccion-fraude-mineria.streamlit.app/ |
| Dataset utilizado | *Credit Card Fraud Detection* (Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. El archivo `creditcard.csv` (150 MB) supera el límite de GitHub; en el repositorio se incluye la muestra de demostración `data/demo/transacciones_demo.parquet` |
| Notebooks por fase CRISP-DM | `notebooks/01` a `04` del repositorio |
