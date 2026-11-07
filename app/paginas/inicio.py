"""Página de inicio: qué hace el sistema y resultados del modelo seleccionado."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # carpeta app/ (para utils)

import pandas as pd
import streamlit as st

from utils import detector, eur, numero, pct, pie_de_pagina

det = detector()
cfg = det.config
mejor = det.modelo_defecto
m = cfg["metricas_prueba"][mejor]

st.title("🛡️ Sistema de Detección de Fraude Financiero")
st.markdown(
    "Sistema basado en *machine learning* y aprendizaje profundo que estima, para cada transacción con "
    "tarjeta de crédito, la probabilidad de que sea **fraude**, y genera alertas cuando supera un umbral "
    "elegido según el costo para el negocio."
)

st.subheader(f"Modelo en producción: {mejor}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Fraudes detectados (Recall)", pct(m["Recall"]),
          help="Proporción de fraudes reales del conjunto de prueba que el modelo alerta")
c2.metric("Alertas correctas (Precision)", pct(m["Precision"]),
          help="Proporción de alertas que son fraudes reales")
c3.metric("PR-AUC", numero(m["PR-AUC"], 3), help="Área bajo la curva Precision-Recall")
c4.metric("Ahorro frente a no usar modelo", pct(m["Ahorro vs. sin modelo (%)"] / 100))
st.caption(
    f"Conjunto de prueba: 56.746 transacciones no usadas en el entrenamiento (95 fraudes). "
    f"Umbral de decisión = {m['Umbral']:.4f}, elegido en validación para minimizar el costo "
    f"(revisión de {eur(cfg['costo_revision'], 0)} por alerta + monto de cada fraude no detectado) "
    f"con Precision ≥ {numero(cfg['precision_minima'], 2)}."
)

st.subheader("Secciones")
st.markdown(
    """
| Sección | Qué permite hacer |
|---|---|
| 📊 **Dashboard de fraude** | KPIs, distribución de montos, fraudes por hora, comparación de modelos y efecto del umbral |
| 🚨 **Alertas en tiempo real** | Simular un flujo de transacciones que se clasifican al llegar y disparan alertas |
| 🔎 **Transacciones sospechosas** | Tabla filtrable con probabilidad, monto, hora y modelos que marcaron cada transacción |
| 🧮 **Predicción** | Evaluar una transacción (con explicación SHAP) o un archivo CSV |

Usa el menú de la izquierda para navegar.
"""
)

with st.expander("Modelos comparados"):
    tabla = pd.DataFrame(cfg["metricas_prueba"]).T[["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC",
                                                     "Costo total (€)"]]
    st.dataframe(tabla.style.format("{:.3f}", subset=["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"])
                 .format("{:,.0f}", subset=["Costo total (€)"]), use_container_width=True)

with st.expander("Metodología CRISP-DM"):
    st.markdown(
        """
1. **Comprensión del negocio:** el costo de un fraude no detectado es mucho mayor que el de una falsa alarma.
2. **Comprensión de los datos:** 284.807 transacciones europeas (septiembre de 2013); 0,173 % fraude.
3. **Preparación:** eliminación de duplicados, variables derivadas (monto logarítmico y hora cíclica),
   división estratificada 60/20/20, escalado y SMOTE **solo** sobre entrenamiento.
4. **Modelado:** Regresión Logística, Random Forest, XGBoost, Stacking, red neuronal profunda y autoencoder,
   con y sin balanceo de clases y con ajuste de hiperparámetros por validación cruzada.
5. **Evaluación:** Precision, Recall, F1, ROC-AUC, PR-AUC, matrices de confusión, costo e interpretabilidad SHAP.
6. **Despliegue:** esta aplicación y una API REST (`POST /predict`).
"""
    )

pie_de_pagina()
