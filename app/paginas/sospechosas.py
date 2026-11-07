"""Transacciones sospechosas: tabla filtrable con probabilidad, monto, hora y modelos que las marcaron."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import (
    COLOR_FRAUDE, COLOR_LEGITIMA, ETIQUETA_RIESGO, TARGET, detector, estilo, eur, hora_del_ciclo,
    marcar_por_modelos, momento, nivel_riesgo, numero, pct, pie_de_pagina, puntajes_prueba,
)

det = detector()
datos = puntajes_prueba()
modelos = det.modelos_disponibles
mejor = det.modelo_defecto

st.title("🔎 Transacciones sospechosas")
st.caption(
    f"Transacciones del conjunto de prueba ({numero(len(datos))}) que al menos un modelo marca como alerta, "
    f"cada uno con su umbral de negocio. La probabilidad y el riesgo corresponden a {mejor}."
)


@st.cache_data(show_spinner=False)
def tabla_marcadas():
    base = marcar_por_modelos(datos[modelos], det.umbrales, modelos)
    base["Fecha y hora"] = momento(datos["Time"]).values
    base["Hora"] = hora_del_ciclo(datos["Time"]).astype(int).values
    base["Monto"] = datos["Amount"].values
    base["Probabilidad"] = datos[mejor].values
    base["Riesgo"] = nivel_riesgo(datos[mejor], det.umbrales[mejor])
    base["Clase real"] = datos[TARGET].map({0: "Legítima", 1: "Fraude"}).values
    return base[base["N.º modelos"] > 0]


marcadas = tabla_marcadas()

with st.sidebar:
    st.header("Filtros")
    elegidos = st.multiselect("Marcada por alguno de estos modelos", modelos, default=modelos)
    minimo_modelos = st.slider("Mínimo de modelos que la marcan", 1, len(modelos), 1,
                               help="Una transacción marcada por varios modelos es una alerta más confiable")
    prob_min = st.slider(f"Probabilidad mínima ({mejor})", 0.0, 1.0, 0.0, 0.01)
    monto_max = float(np.ceil(marcadas["Monto"].max()))
    rango_monto = st.slider("Monto (€)", 0.0, monto_max, (0.0, monto_max), 1.0)
    rango_hora = st.slider("Hora del día", 0, 23, (0, 23))
    riesgos = st.multiselect("Nivel de riesgo", ["ALTO", "MEDIO", "BAJO"], default=["ALTO", "MEDIO", "BAJO"],
                             format_func=ETIQUETA_RIESGO.get)
    mostrar_real = st.checkbox("Mostrar clase real", value=True,
                               help="Solo disponible porque son datos de prueba etiquetados; en producción no se conoce")

filtro = (
    marcadas[elegidos].ge(np.array([det.umbrales[m] for m in elegidos])).any(axis=1) if elegidos
    else np.zeros(len(marcadas), dtype=bool)
)
vista = marcadas[
    filtro
    & (marcadas["N.º modelos"] >= minimo_modelos)
    & (marcadas["Probabilidad"] >= prob_min)
    & marcadas["Monto"].between(*rango_monto)
    & marcadas["Hora"].between(*rango_hora)
    & marcadas["Riesgo"].isin(riesgos)
].sort_values(["N.º modelos", "Probabilidad"], ascending=False)

c = st.columns(4)
c[0].metric("Transacciones sospechosas", numero(len(vista)))
c[1].metric("Monto total", eur(vista["Monto"].sum(), 0))
c[2].metric("Marcadas por todos los modelos", numero(int((vista["N.º modelos"] == len(modelos)).sum())))
if mostrar_real and len(vista):
    c[3].metric("Son fraude real", pct((vista["Clase real"] == "Fraude").mean()),
                f"{int((vista['Clase real'] == 'Fraude').sum())} de {len(vista)}", delta_color="off")

if vista.empty:
    st.warning("Ninguna transacción cumple los filtros.")
else:
    columnas = ["Fecha y hora", "Hora", "Monto", "Probabilidad", "Riesgo", "Marcada por", "N.º modelos"]
    if mostrar_real:
        columnas.append("Clase real")
    st.dataframe(
        vista[columnas].assign(Riesgo=lambda d: d["Riesgo"].map(ETIQUETA_RIESGO)),
        use_container_width=True, height=440,
        column_config={
            "_index": st.column_config.NumberColumn("Transacción", format="%d"),
            "Fecha y hora": st.column_config.DatetimeColumn(format="DD/MM/YYYY HH:mm:ss"),
            "Monto": st.column_config.NumberColumn(format="%.2f €"),
            "Probabilidad": st.column_config.ProgressColumn(f"P(fraude) · {mejor}", format="%.3f",
                                                            min_value=0.0, max_value=1.0),
            "N.º modelos": st.column_config.NumberColumn(help=f"De {len(modelos)} modelos"),
        },
    )
    st.download_button("⬇ Descargar CSV", vista[columnas].to_csv(index_label="id_transaccion").encode("utf-8"),
                       "transacciones_sospechosas.csv", "text/csv")

    fig = go.Figure()
    grupos = ((("Fraude", COLOR_FRAUDE, "diamond"), ("Legítima", COLOR_LEGITIMA, "circle")) if mostrar_real
              else ((None, COLOR_FRAUDE, "circle"),))
    for clase, color, simbolo in grupos:
        parte = vista if clase is None else vista[vista["Clase real"] == clase]
        fig.add_scatter(
            x=parte["Probabilidad"], y=parte["Monto"].clip(lower=0.1), mode="markers",
            name=clase or "Sospechosa", marker=dict(color=color, symbol=simbolo, size=9, opacity=0.85,
                                                    line=dict(color="white", width=1)),
            customdata=np.c_[parte.index, parte["N.º modelos"], parte["Hora"]],
            hovertemplate="#%{customdata[0]}<br>P(fraude) %{x:.3f}<br>Monto %{y:.2f} €<br>"
                          "Hora %{customdata[2]}:00<br>Marcada por %{customdata[1]} modelos<extra></extra>",
        )
    fig.update_xaxes(title=f"Probabilidad de fraude ({mejor})", range=[-0.02, 1.02])
    fig.update_yaxes(title="Monto (€, escala log)", type="log")
    st.plotly_chart(estilo(fig, alto=380, titulo="Monto frente a probabilidad"), use_container_width=True)

pie_de_pagina()
