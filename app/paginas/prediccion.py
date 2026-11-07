"""Predicción individual (formulario con explicación SHAP) y por lotes (archivo CSV)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_score, recall_score

from utils import (
    ACCIONES, COLOR_FRAUDE, COLOR_LEGITIMA, COLUMNAS_CRUDAS, COLUMNAS_V, ETIQUETA_RIESGO, TARGET,
    datos_demo, detector, estilo, eur, numero, pct, pie_de_pagina, selector_modelo,
)

det = detector()
demo = datos_demo()

st.title("🧮 Predicción de fraude")
tab_individual, tab_lote = st.tabs(["Transacción individual", "Archivo CSV"])

# ======================================================================
# Transacción individual
# ======================================================================
CAMPOS = ["Time", "Amount", *COLUMNAS_V]
for campo in CAMPOS:
    st.session_state.setdefault(f"in_{campo}", 0.0)
st.session_state.setdefault("in_real", None)


def cargar_ejemplo(clase: int) -> None:
    """Rellena el formulario con una transacción real del conjunto de prueba."""
    fila = demo[demo[TARGET] == clase].sample(1).iloc[0]
    for campo in CAMPOS:
        st.session_state[f"in_{campo}"] = float(fila[campo])
    st.session_state.in_real = "Fraude" if clase == 1 else "Legítima"


with tab_individual:
    st.markdown("Introduce los datos de la transacción o carga un ejemplo real del conjunto de prueba.")
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.button("Cargar ejemplo de fraude", on_click=cargar_ejemplo, args=(1,), use_container_width=True)
    c2.button("Cargar ejemplo legítimo", on_click=cargar_ejemplo, args=(0,), use_container_width=True)
    if st.session_state.in_real:
        c3.caption(f"Ejemplo cargado · clase real: **{st.session_state.in_real}** (el modelo no la ve)")

    with st.form("formulario"):
        a, b, c = st.columns(3)
        a.number_input("Time (segundos desde la primera transacción)", min_value=0.0, max_value=200_000.0,
                       step=1.0, key="in_Time")
        b.number_input("Amount (€)", min_value=0.0, step=1.0, format="%.2f", key="in_Amount")
        modelo = c.selectbox("Modelo", det.modelos_disponibles, index=det.modelos_disponibles.index(det.modelo_defecto))
        with st.expander("Variables V1–V28 (componentes PCA)", expanded=False):
            columnas = st.columns(4)
            for i, v in enumerate(COLUMNAS_V):
                columnas[i % 4].number_input(v, format="%.6f", step=0.1, key=f"in_{v}")
        enviado = st.form_submit_button("Evaluar transacción", type="primary")

    if enviado:
        transaccion = pd.DataFrame([{campo: st.session_state[f"in_{campo}"] for campo in COLUMNAS_CRUDAS}])
        r = det.predecir(transaccion, modelo).iloc[0]
        es_prob = modelo != "Autoencoder"

        col1, col2 = st.columns([1, 1.4])
        with col1:
            etiqueta = "Probabilidad de fraude" if es_prob else "Error de reconstrucción"
            st.metric(etiqueta, pct(r["puntaje"]) if es_prob else numero(r["puntaje"], 3),
                      f"umbral {r['umbral']:.4f}", delta_color="off")
            if r["alerta"]:
                st.error(f"🚨 **ALERTA** · Riesgo {ETIQUETA_RIESGO[r['riesgo']]}")
            elif r["riesgo"] == "MEDIO":
                st.warning(f"Riesgo {ETIQUETA_RIESGO[r['riesgo']]}")
            else:
                st.success(f"Riesgo {ETIQUETA_RIESGO[r['riesgo']]}")
            st.markdown(f"**Acción recomendada:** {ACCIONES[r['riesgo']]}")
        with col2:
            todos = det.predecir_todos(transaccion).iloc[0]
            comparacion = pd.DataFrame({
                "Modelo": det.modelos_disponibles,
                "Puntaje": [todos[m] for m in det.modelos_disponibles],
                "Umbral": [det.umbrales[m] for m in det.modelos_disponibles],
            })
            comparacion["Alerta"] = np.where(comparacion["Puntaje"] >= comparacion["Umbral"], "🚨 Sí", "No")
            st.markdown(f"**Opinión de todos los modelos** · {todos['N.º modelos']} de "
                        f"{len(det.modelos_disponibles)} la marcan")
            st.dataframe(comparacion, hide_index=True, use_container_width=True,
                         column_config={"Puntaje": st.column_config.NumberColumn(format="%.4f"),
                                        "Umbral": st.column_config.NumberColumn(format="%.4f")})

        st.subheader("¿Por qué? Explicación SHAP (XGBoost)")
        explicacion = det.explicar(transaccion, top=10).iloc[::-1]
        colores = [COLOR_FRAUDE if v > 0 else COLOR_LEGITIMA for v in explicacion["Contribución"]]
        fig = go.Figure(go.Bar(
            x=explicacion["Contribución"], orientation="h", marker_color=colores,
            y=[f"{v} = {x:.3f}" for v, x in zip(explicacion["Variable"], explicacion["Valor"])],
            hovertemplate="%{y}<br>Contribución %{x:+.3f} (log-odds)<extra></extra>",
        ))
        fig.add_vline(x=0, line_color="#52514e", line_width=1)
        fig.update_xaxes(title="Contribución a la predicción (log-odds) · ← legítima | fraude →")
        st.plotly_chart(estilo(fig, alto=380), use_container_width=True)
        st.caption("Barras naranjas empujan la predicción hacia fraude; azules, hacia legítima. "
                   "La explicación usa siempre XGBoost, el modelo seleccionado.")

# ======================================================================
# Archivo CSV
# ======================================================================
with tab_lote:
    st.markdown(
        "Sube un CSV con las columnas `Time`, `V1` … `V28` y `Amount` (una fila por transacción). "
        "Si incluye la columna `Class`, también se calculan Precision y Recall."
    )
    plantilla = demo.sample(20, random_state=1)[COLUMNAS_CRUDAS]
    st.download_button("⬇ Descargar plantilla de ejemplo (20 transacciones)",
                       plantilla.to_csv(index=False).encode("utf-8"), "plantilla_transacciones.csv", "text/csv")

    archivo = st.file_uploader("Archivo CSV", type="csv")
    modelo_lote = selector_modelo("lote_modelo", "Modelo para el archivo")
    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
        except Exception as error:  # archivo corrupto o con otro separador
            st.error(f"No se pudo leer el archivo: {error}")
            st.stop()
        faltan = sorted(set(COLUMNAS_CRUDAS) - set(datos.columns))
        if faltan:
            st.error(f"Faltan columnas: {', '.join(faltan)}")
            st.stop()
        validos = datos[COLUMNAS_CRUDAS].notna().all(axis=1)
        if not validos.all():
            st.warning(f"Se omiten {int((~validos).sum())} filas con valores vacíos.")
            datos = datos[validos]

        resultado = det.predecir(datos, modelo_lote)
        salida = pd.concat([datos.reset_index(drop=True), resultado[["puntaje", "alerta", "riesgo", "accion"]]
                            .reset_index(drop=True)], axis=1)

        c = st.columns(4)
        c[0].metric("Transacciones", numero(len(salida)))
        c[1].metric("Alertas", numero(int(salida["alerta"].sum())), pct(salida["alerta"].mean(), 2), delta_color="off")
        c[2].metric("Monto en alertas", eur(salida.loc[salida["alerta"], "Amount"].sum(), 0))
        if TARGET in salida.columns:
            y = salida[TARGET].astype(int)
            c[3].metric("Precision / Recall", f"{precision_score(y, salida['alerta'], zero_division=0):.2f} / "
                                              f"{recall_score(y, salida['alerta'], zero_division=0):.2f}")

        st.dataframe(
            salida.assign(riesgo=salida["riesgo"].map(ETIQUETA_RIESGO))
            .sort_values("puntaje", ascending=False)[["Time", "Amount", "puntaje", "alerta", "riesgo", "accion"]
                                                     + ([TARGET] if TARGET in salida.columns else [])],
            use_container_width=True, height=400,
            column_config={"Amount": st.column_config.NumberColumn(format="%.2f €"),
                           "puntaje": st.column_config.NumberColumn("Puntaje", format="%.4f"),
                           "alerta": "Alerta", "riesgo": "Riesgo", "accion": "Acción recomendada"},
        )
        st.download_button("⬇ Descargar resultados", salida.to_csv(index=False).encode("utf-8"),
                           "resultados_fraude.csv", "text/csv")

pie_de_pagina()
