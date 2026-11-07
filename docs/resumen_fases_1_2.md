# Resumen CRISP-DM — Fases 1 y 2

Fuente: `notebooks/01_comprension_negocio_y_datos.ipynb`. Figuras en `reports/figures/`.

## Fase 1. Comprensión del negocio

**Problema.** El fraude con tarjetas de crédito genera pérdidas directas (monto defraudado y contracargos) y pérdidas indirectas (costos operativos y deterioro de la confianza del cliente). Según *The Nilson Report*, las pérdidas globales por fraude con tarjetas de pago superaron los 33 000 millones de dólares en 2023 *[verificar la edición exacta antes de citar]*. La entidad necesita detectar las transacciones fraudulentas en tiempo casi real, sin bloquear a los clientes legítimos.

**Objetivo de negocio.** Reducir la pérdida económica por fraude manteniendo un volumen de alertas que el equipo de revisión pueda atender.

**Objetivo de minería de datos.** Construir un clasificador binario que estime la probabilidad de fraude de cada transacción (`Class = 1`), complementado con un detector de anomalías no supervisado (autoencoder).

**Costo de los errores.** Un falso negativo (fraude no detectado) implica perder el monto de la transacción. Un falso positivo (transacción legítima bloqueada) implica un costo de revisión y molestia al cliente. El primero es mucho más costoso, por lo que se prioriza el Recall sin descuidar la Precision.

**Criterios de éxito (conjunto de prueba).**

| Criterio | Meta |
|---|---|
| Recall (fraude) | ≥ 0,80 |
| Precision (fraude) | ≥ 0,70 |
| PR-AUC (Average Precision) | ≥ 0,80 |
| Costo total | Menor que operar sin modelo |

El *accuracy* se descarta como criterio por el desbalance de clases.

**Supuestos y riesgos.** Las variables `V1`–`V28` son componentes PCA anonimizados, lo que limita la interpretación de negocio. Los datos son de 2013 y abarcan solo dos días, por lo que existe riesgo de *concept drift* y el modelo debe reentrenarse periódicamente.

## Fase 2. Comprensión de los datos

**Descripción.** 284 807 transacciones de titulares europeos (septiembre de 2013) con 31 columnas numéricas: `Time` (segundos desde la primera transacción), `V1`–`V28` (componentes PCA), `Amount` (euros) y `Class` (1 = fraude).

**Calidad.** Sin valores nulos. 1 081 filas duplicadas exactas (0,38 %), 19 de ellas fraudes, sin etiquetas contradictorias. Tras eliminarlas quedan 473 fraudes únicos.

**Desbalance (Fig. 01).** 492 fraudes (0,173 %), es decir, 1 fraude por cada 578 transacciones legítimas. Un clasificador trivial que siempre predice "legítima" alcanza un *accuracy* de 99,83 % sin detectar ningún fraude.

**Monto (Fig. 02).** Distribución con asimetría de 16,98, que se reduce a 0,16 con `log(1 + Amount)`. La mediana del fraude es 9,25 € frente a 22,00 € en las legítimas, pero su media es mayor (122,21 € frente a 88,29 €). El 25 % de los fraudes es de 1,00 € o menos, patrón compatible con *card testing*. La diferencia de distribuciones es significativa (Mann-Whitney U, p = 8,6 × 10⁻⁶).

**Tiempo (Figs. 03 y 04).** El registro abarca 48 horas con un ciclo diario. El volumen cae de madrugada y la tasa de fraude sube: 1,71 % a las 2 h (≈ 10 veces la tasa global de 0,173 %) y 1,04 % a las 4 h.

**Correlaciones (Figs. 05 y 06).** Las componentes `V1`–`V28` son ortogonales entre sí (|r| máxima ≈ 0), por lo que no hay multicolinealidad. `Amount` se correlaciona con `V2` (0,53), `V7` (0,40) y `V5` (0,39); `Time` con `V3` (0,42). Las correlaciones lineales con `Class` son débiles: `V17` (0,33), `V14` (0,30), `V12` (0,26), `V10` (0,22).

**Separabilidad (Fig. 07).** El estadístico de Kolmogórov-Smirnov indica una separación fuerte pero no lineal: `V14` (0,84), `V10` (0,80), `V12` (0,78), `V4` (0,77), `V11` (0,76), `V17` (0,75).

**Valores atípicos (Fig. 08).** Con el criterio de 1,5·IQR, el 97,0 % de los fraudes presenta al menos una variable atípica, frente al 48,5 % de las legítimas. En `V14`, el 87,4 % de los fraudes es atípico frente al 4,8 % de las legítimas. Los atípicos constituyen la señal del fraude y no se eliminan.

**Decisiones derivadas para la Fase 3.**

| Hallazgo | Decisión |
|---|---|
| Duplicados exactos | Eliminar antes de la división |
| Desbalance 1:578 | División estratificada; SMOTE solo en entrenamiento; comparación con pesos de clase |
| *Accuracy* engañoso | Precision, Recall, F1, ROC-AUC, PR-AUC y matriz de confusión |
| `Amount` asimétrico | `Amount_log` escalado con `RobustScaler` |
| Fraude concentrado de madrugada | Variables cíclicas `Hour_sin`, `Hour_cos` |
| `Time` es un contador relativo | Excluir `Time` crudo; su información útil queda en la hora cíclica |
| Varianzas distintas en `V1`–`V28` | `StandardScaler` (necesario para las redes neuronales) |
| Atípicos = fraude | Conservarlos; refuerza el enfoque de detección de anomalías |

**Figuras generadas.**

| Archivo | Contenido |
|---|---|
| `01_desbalance_clases.png` | Conteo por clase (escala log) |
| `02_monto_por_clase.png` | Distribución y diagrama de caja del monto por clase |
| `03_time_por_clase.png` | Distribución temporal por clase (48 h) |
| `04_patron_horario.png` | Volumen y tasa de fraude por hora del día |
| `05_correlacion_con_clase.png` | Correlación de cada variable con `Class` |
| `06_matriz_correlacion.png` | Matriz de correlación completa |
| `07_variables_mas_separables.png` | Densidades de las 6 variables con mayor KS |
| `08_outliers_por_clase.png` | Diagramas de caja con atípicos por clase |
