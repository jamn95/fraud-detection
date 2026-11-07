"""
Aplicación web del sistema de detección de fraude (CRISP-DM fase 6).

Ejecutar desde la raíz del proyecto:
    streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="Detección de Fraude", page_icon="🛡️", layout="wide")

paginas = [
    st.Page("paginas/inicio.py", title="Inicio", icon="🏠", default=True),
    st.Page("paginas/dashboard.py", title="Dashboard de fraude", icon="📊"),
    st.Page("paginas/monitoreo.py", title="Alertas en tiempo real", icon="🚨"),
    st.Page("paginas/sospechosas.py", title="Transacciones sospechosas", icon="🔎"),
    st.Page("paginas/prediccion.py", title="Predicción", icon="🧮"),
]
st.navigation(paginas).run()
