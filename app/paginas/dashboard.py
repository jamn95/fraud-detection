"""Dashboard de fraude: KPIs, montos, patrón horario, comparación de modelos y efecto del umbral."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from utils import (
    COLOR_FRAUDE, COLOR_LEGITIMA, COLOR_TEXTO, COLORES_MODELOS, COSTO_REVISION, FIG_DIR, TARGET,
    control_umbral, detector, estilo, eur, evaluar_en_umbral, hora_del_ciclo, numero, pct,
    pie_de_pagina, puntajes_prueba, selector_modelo,
)

det = detector()
datos = puntajes_prueba()
y, montos = datos[TARGET], datos["Amount"]

st.title("📊 Dashboard de fraude")
st.caption(f"Conjunto de prueba: {numero(len(datos))} transacciones que ningún modelo vio al entrenar.")

with st.sidebar:
    st.header("Configuración")
    modelo = selector_modelo("dash_modelo")
    umbral = control_umbral(modelo, datos[modelo])
    costo_revision = st.number_input("Costo de revisión por alerta (€)", 0.0, 100.0, COSTO_REVISION, 1.0)

r = evaluar_en_umbral(y, datos[modelo], montos, umbral, costo_revision)
fraudes_reales = int(y.sum())
monto_fraude = float(montos[y == 1].sum())
ahorro = 1 - r["Costo total (€)"] / monto_fraude

# ----------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------
c = st.columns(4)
c[0].metric("Transacciones analizadas", numero(len(datos)))
c[1].metric("Fraudes reales", numero(fraudes_reales), f"{eur(monto_fraude, 0)} en juego", delta_color="off")
c[2].metric("Alertas generadas", numero(r["Alertas"]), f"{pct(r['Alertas'] / len(datos), 2)} del total",
            delta_color="off")
c[3].metric("Ahorro frente a no usar modelo", pct(ahorro), f"costo {eur(r['Costo total (€)'], 0)}",
            delta_color="off")
c = st.columns(4)
c[0].metric("Fraudes detectados (Recall)", pct(r["Recall"]), f"{r['TP']} de {fraudes_reales}", delta_color="off")
c[1].metric("Alertas correctas (Precision)", pct(r["Precision"]))
c[2].metric("Falsas alarmas", numero(r["FP"]), f"{pct(r['FP'] / (r['FP'] + r['TN']), 3)} de las legítimas",
            delta_color="off")
c[3].metric("Fraude evitado", eur(r["Fraude evitado (€)"], 0), f"perdido: {eur(r['Fraude perdido (€)'], 0)}",
            delta_color="off")

st.divider()

# ----------------------------------------------------------------------
# Montos y patrón horario
# ----------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    bordes = np.logspace(-1, np.log10(montos.max() + 1), 45)
    fig = go.Figure()
    for clase, nombre, color in ((0, "Legítima", COLOR_LEGITIMA), (1, "Fraude", COLOR_FRAUDE)):
        valores = montos[y == clase].clip(lower=0.1)
        densidad, _ = np.histogram(valores, bins=bordes, density=False)
        proporcion = densidad / densidad.sum()
        centros = np.sqrt(bordes[:-1] * bordes[1:])
        fig.add_trace(go.Scatter(
            x=centros, y=proporcion, name=nombre, mode="lines", line=dict(color=color, width=2.5, shape="hvh"),
            customdata=np.c_[bordes[:-1], bordes[1:]],
            hovertemplate=f"<b>{nombre}</b><br>%{{customdata[0]:.2f}} – %{{customdata[1]:.2f}} €"
                          "<br>%{y:.1%} de las transacciones<extra></extra>",
        ))
    fig.update_xaxes(type="log", title="Monto (€, escala logarítmica)")
    fig.update_yaxes(title="Proporción dentro de cada clase", tickformat=".0%")
    st.plotly_chart(estilo(fig, titulo="Distribución de montos por clase"), use_container_width=True)
    st.caption("Los montos cero se muestran como 0,10 €. Una cuarta parte de los fraudes es de 1 € o menos.")

with col2:
    pred = datos[modelo] >= umbral
    por_hora = (pd.DataFrame({"Hora": hora_del_ciclo(datos["Time"]).astype(int), "Fraude": y == 1,
                              "Detectado": (y == 1) & pred, "Falsa alarma": (y == 0) & pred})
                .groupby("Hora").sum().reindex(range(24), fill_value=0))
    fig = go.Figure()
    fig.add_bar(x=por_hora.index, y=por_hora["Fraude"], name="Fraudes reales", marker_color=COLOR_FRAUDE,
                hovertemplate="Hora %{x}:00<br>%{y} fraudes reales<extra></extra>")
    fig.add_bar(x=por_hora.index, y=por_hora["Detectado"], name=f"Detectados por {modelo}",
                marker_color=COLORES_MODELOS[modelo],
                hovertemplate="Hora %{x}:00<br>%{y} detectados<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(title="Hora del día", dtick=2)
    fig.update_yaxes(title="Fraudes")
    st.plotly_chart(estilo(fig, titulo="Fraudes por hora del día"), use_container_width=True)
    pico = por_hora["Fraude"].idxmax()
    st.caption(f"Hora con más fraudes en prueba: {pico}:00 ({por_hora.loc[pico, 'Fraude']} casos). "
               "La hora se deriva de `Time` asumiendo que el registro empieza a medianoche.")

st.divider()

# ----------------------------------------------------------------------
# Comparación de modelos
# ----------------------------------------------------------------------
st.subheader("Comparación de modelos")
st.caption("Cada modelo con su umbral de negocio (elegido en validación). Prueba con la distribución real.")
tabla = pd.DataFrame(det.config["metricas_prueba"]).T
col1, col2 = st.columns([1.1, 1])
with col1:
    fig = go.Figure()
    metricas = ["Precision", "Recall", "F1", "PR-AUC"]
    for m in det.modelos_disponibles:
        fig.add_bar(x=metricas, y=tabla.loc[m, metricas].astype(float), name=m, marker_color=COLORES_MODELOS[m],
                    hovertemplate=f"<b>{m}</b><br>%{{x}} = %{{y:.3f}}<extra></extra>")
    fig.add_hline(y=0.80, line_dash="dot", line_color=COLOR_TEXTO, line_width=1,
                  annotation_text="meta 0,80 (Recall, PR-AUC)", annotation_position="top left")
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_yaxes(range=[0, 1.05], title="Valor en prueba")
    st.plotly_chart(estilo(fig, alto=420, titulo="Métricas por modelo"), use_container_width=True)
with col2:
    fig = go.Figure()
    for m in det.modelos_disponibles:
        p, rc, _ = precision_recall_curve(y, datos[m])
        paso = max(1, len(p) // 400)   # se submuestrea la curva para que el gráfico sea liviano
        fig.add_scatter(x=rc[::paso], y=p[::paso], mode="lines", name=m,
                        line=dict(color=COLORES_MODELOS[m], width=3 if m == modelo else 1.8),
                        hovertemplate=f"<b>{m}</b><br>Recall %{{x:.2f}}<br>Precision %{{y:.2f}}<extra></extra>")
    fig.add_scatter(x=[r["Recall"]], y=[r["Precision"]], mode="markers", name="Umbral actual",
                    marker=dict(size=12, color=COLORES_MODELOS[modelo], line=dict(color="white", width=2)),
                    hovertemplate="Umbral actual<br>Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>")
    fig.update_xaxes(title="Recall", range=[0, 1.02])
    fig.update_yaxes(title="Precision", range=[0, 1.02])
    st.plotly_chart(estilo(fig, alto=420, titulo="Curvas Precision-Recall"), use_container_width=True)

vista = tabla[["Umbral", "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "TP", "FP", "FN",
               "Costo total (€)", "Ahorro vs. sin modelo (%)"]].astype(float)
st.dataframe(
    vista.style.format("{:.3f}", subset=["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"])
    .format("{:.4f}", subset=["Umbral"]).format("{:.0f}", subset=["TP", "FP", "FN"])
    .format("{:,.0f}", subset=["Costo total (€)"]).format("{:.1f}", subset=["Ahorro vs. sin modelo (%)"])
    .highlight_max(subset=["Precision", "Recall", "F1", "PR-AUC"], color="#cde2fb"),
    use_container_width=True,
)

st.divider()

# ----------------------------------------------------------------------
# Matriz de confusión con el umbral actual
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])
with col1:
    matriz = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    texto = [[f"{numero(v)}<br>{etq}" for v, etq in zip(fila, etqs)]
             for fila, etqs in zip(matriz, [["Verdaderos negativos", "Falsas alarmas"],
                                            ["Fraudes no detectados", "Fraudes detectados"]])]
    fig = go.Figure(go.Heatmap(
        z=np.log10(matriz + 1), x=["Legítima", "Fraude"], y=["Legítima", "Fraude"], text=texto,
        texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{text}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicción", side="top")
    fig.update_yaxes(title="Clase real", autorange="reversed")
    st.plotly_chart(estilo(fig, alto=360, titulo=f"Matriz de confusión – {modelo}"), use_container_width=True)
with col2:
    st.markdown(f"**Umbral actual:** {umbral:.4f} · umbral de negocio: {det.umbrales[modelo]:.4f}")
    st.markdown(
        "- **Bajar el umbral** detecta más fraudes (sube el Recall) a cambio de más falsas alarmas.\n"
        "- **Subir el umbral** reduce las falsas alarmas, pero deja pasar más fraudes.\n"
        f"- Con el umbral actual se revisan **{numero(r['Alertas'])}** transacciones para detectar "
        f"**{r['TP']}** fraudes y se pierden **{eur(r['Fraude perdido (€)'], 0)}** en fraudes no detectados."
    )

with st.expander("Interpretabilidad del modelo (SHAP sobre XGBoost)"):
    c1, c2 = st.columns(2)
    for col, archivo, pie in ((c1, "20_shap_resumen.png", "Cada punto es una transacción; a la derecha empuja hacia fraude."),
                              (c2, "21_shap_importancia.png", "Importancia media de cada variable.")):
        ruta = FIG_DIR / archivo
        if ruta.exists():
            col.image(str(ruta), caption=pie, use_container_width=True)
    st.caption("Valores bajos de V14, V12 y V10 y valores altos de V4 y V11 son los principales indicadores de fraude.")

pie_de_pagina()
