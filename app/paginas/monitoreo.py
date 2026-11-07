"""
Alertas en tiempo real: reproduce transacciones del conjunto de prueba como si
llegaran una a una, las clasifica al llegar y dispara una alerta visual cuando
el puntaje supera el umbral del modelo.

Implementación: un fragmento de Streamlit con `run_every` se vuelve a ejecutar
cada segundo mientras la simulación está activa; en cada ciclo procesa un lote
de transacciones (según la velocidad elegida). El estado vive en session_state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLOR_FRAUDE, COLOR_LEGITIMA, COLOR_TEXTO, ETIQUETA_RIESGO, TARGET, datos_demo, detector, estilo,
    eur, momento, numero, pct, pie_de_pagina, selector_modelo,
)

INTERVALO_S = 1.0          # cada cuánto se actualiza el panel
VENTANA_GRAFICO = 300      # transacciones visibles en el gráfico

det = detector()
# Orden cronológico (por Time), como llegarían las transacciones en la realidad
flujo = datos_demo().sort_values("Time", kind="stable")

st.title("🚨 Alertas en tiempo real")
st.caption(
    f"Simulación con {numero(len(flujo))} transacciones del conjunto de prueba ({int(flujo[TARGET].sum())} fraudes, "
    f"{pct(flujo[TARGET].mean())} del flujo; en la realidad es ≈ 0,17 %, se enriquece para que la demostración "
    "muestre alertas). Las transacciones se clasifican **al llegar**, sin conocer su etiqueta real."
)


def reiniciar() -> None:
    st.session_state.mon_pos = 0
    st.session_state.mon_corriendo = False
    st.session_state.mon_historial = []
    st.session_state.mon_nuevas = 0


for clave, valor in (("mon_pos", 0), ("mon_corriendo", False), ("mon_historial", []), ("mon_nuevas", 0)):
    st.session_state.setdefault(clave, valor)

with st.sidebar:
    st.header("Simulación")
    modelo = selector_modelo("mon_modelo")
    if st.session_state.get("mon_modelo_activo") != modelo:      # cambiar de modelo reinicia el flujo
        reiniciar()
        st.session_state.mon_modelo_activo = modelo
    velocidad = st.select_slider("Transacciones por segundo", options=[1, 2, 5, 10, 25, 50], value=5)
    umbral = det.umbrales[modelo]
    st.caption(f"Umbral de alerta de {modelo}: **{umbral:.4f}**")

c1, c2, c3, _ = st.columns([1, 1, 1, 3])
if c1.button("▶ Iniciar", type="primary", disabled=st.session_state.mon_corriendo, use_container_width=True):
    if st.session_state.mon_pos >= len(flujo):
        reiniciar()
    st.session_state.mon_corriendo = True
    st.rerun()
if c2.button("⏸ Pausar", disabled=not st.session_state.mon_corriendo, use_container_width=True):
    st.session_state.mon_corriendo = False
    st.rerun()
if c3.button("↺ Reiniciar", use_container_width=True):
    reiniciar()
    st.rerun()


def procesar_lote() -> None:
    """Clasifica las siguientes transacciones del flujo y guarda el resultado."""
    inicio = st.session_state.mon_pos
    lote = flujo.iloc[inicio:inicio + velocidad]
    if lote.empty:
        st.session_state.mon_corriendo = False
        st.rerun(scope="app")      # detiene el refresco automático
    resultado = det.predecir(lote, modelo)
    registros = pd.DataFrame({
        "id": lote.index, "Fecha y hora": momento(lote["Time"]).values, "Monto": lote["Amount"].values,
        "Puntaje": resultado["puntaje"].values, "Alerta": resultado["alerta"].values,
        "Riesgo": resultado["riesgo"].values, "Real": lote[TARGET].map({0: "Legítima", 1: "Fraude"}).values,
    })
    st.session_state.mon_historial.extend(registros.to_dict("records"))
    st.session_state.mon_pos = inicio + len(lote)
    nuevas = registros[registros["Alerta"]]
    st.session_state.mon_nuevas = len(nuevas)
    for _, fila in nuevas.head(3).iterrows():     # como máximo 3 avisos emergentes por ciclo
        st.toast(f"Transacción #{fila['id']} · {eur(fila['Monto'])} · puntaje {fila['Puntaje']:.3f}", icon="🚨")


def dibujar_panel() -> None:
    historial = pd.DataFrame(st.session_state.mon_historial)
    total = len(flujo)
    st.progress(st.session_state.mon_pos / total,
                text=f"Procesadas {numero(st.session_state.mon_pos)} de {numero(total)} transacciones"
                     + (" · en curso" if st.session_state.mon_corriendo else " · en pausa"))
    if historial.empty:
        st.info("Pulsa **▶ Iniciar** para comenzar a recibir transacciones.")
        return

    alertas = historial[historial["Alerta"]]
    fraudes = historial[historial["Real"] == "Fraude"]
    detectados = int((alertas["Real"] == "Fraude").sum())
    c = st.columns(5)
    c[0].metric("Procesadas", numero(len(historial)))
    c[1].metric("Alertas", numero(len(alertas)), f"+{st.session_state.mon_nuevas} en el último ciclo",
                delta_color="inverse" if st.session_state.mon_nuevas else "off")
    c[2].metric("Fraudes reales recibidos", numero(len(fraudes)))
    c[3].metric("Fraudes detectados", numero(detectados),
                pct(detectados / len(fraudes)) if len(fraudes) else "—", delta_color="off")
    c[4].metric("Falsas alarmas", numero(len(alertas) - detectados))

    if st.session_state.mon_nuevas:
        st.error(f"🚨 **{st.session_state.mon_nuevas} alerta(s) nueva(s)** en el último ciclo. "
                 "Revisa la tabla de alertas.")
    else:
        st.success("✅ Sin alertas nuevas en el último ciclo.")

    reciente = historial.tail(VENTANA_GRAFICO)
    fig = go.Figure()
    for alerta, nombre, color in ((False, "Sin alerta", COLOR_LEGITIMA), (True, "Alerta", COLOR_FRAUDE)):
        parte = reciente[reciente["Alerta"] == alerta]
        fig.add_scatter(
            x=parte["Fecha y hora"], y=parte["Puntaje"].clip(lower=1e-6), mode="markers", name=nombre,
            marker=dict(color=color, size=10 if alerta else 7, line=dict(color="white", width=1),
                        symbol=["diamond" if r == "Fraude" else "circle" for r in parte["Real"]]),
            customdata=parte[["id", "Monto", "Real"]].values,
            hovertemplate="#%{customdata[0]}<br>%{x|%d/%m %H:%M:%S}<br>Monto %{customdata[1]:.2f} €"
                          "<br>Puntaje %{y:.4f}<br>Real: %{customdata[2]}<extra></extra>",
        )
    fig.add_hline(y=umbral, line_dash="dash", line_color=COLOR_TEXTO, line_width=1,
                  annotation_text=f"umbral {umbral:.4f}", annotation_position="top left")
    fig.update_yaxes(type="log", title="Puntaje de riesgo (escala log)")
    fig.update_xaxes(title="Hora de la transacción (simulada)")
    st.plotly_chart(estilo(fig, alto=360, titulo=f"Últimas {len(reciente)} transacciones – "
                                                    "◆ = fraude real, ● = legítima"),
                    use_container_width=True)

    st.subheader("Alertas generadas")
    if alertas.empty:
        st.caption("Todavía no hay alertas.")
        return
    tabla = alertas.iloc[::-1].head(50).assign(Riesgo=lambda d: d["Riesgo"].map(ETIQUETA_RIESGO),
                                               Resultado=lambda d: d["Real"].map({"Fraude": "✔ Fraude real",
                                                                                  "Legítima": "✘ Falsa alarma"}))
    st.dataframe(
        tabla[["id", "Fecha y hora", "Monto", "Puntaje", "Riesgo", "Resultado"]], hide_index=True,
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn("Transacción", format="%d"),
            "Fecha y hora": st.column_config.DatetimeColumn(format="DD/MM HH:mm:ss"),
            "Monto": st.column_config.NumberColumn(format="%.2f €"),
            "Puntaje": st.column_config.NumberColumn(format="%.4f"),
            "Resultado": st.column_config.TextColumn("Etiqueta real (solo para evaluar la demo)"),
        },
    )


@st.fragment(run_every=INTERVALO_S if st.session_state.mon_corriendo else None)
def panel_en_vivo() -> None:
    if st.session_state.mon_corriendo:
        procesar_lote()
    dibujar_panel()


panel_en_vivo()
pie_de_pagina()
