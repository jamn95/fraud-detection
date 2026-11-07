# Imagen de la app web (Streamlit) y la API (FastAPI) del detector de fraude.
#   docker build -t deteccion-fraude .
#   docker run -p 8501:8501 deteccion-fraude                     -> app en http://localhost:8501
#   docker run -p 8000:8000 deteccion-fraude \
#       uvicorn app.api:app --host 0.0.0.0 --port 8000           -> API en http://localhost:8000/docs
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# libgomp1: OpenMP, necesario para XGBoost en imágenes slim; curl: chequeo de salud
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /proyecto

# Primero las dependencias (capa que se reutiliza si solo cambia el código)
COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

# Solo lo necesario para servir: código, modelos entrenados y datos de demostración
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/demo/ data/demo/
COPY reports/figures/ reports/figures/
COPY .streamlit/ .streamlit/

# Usuario sin privilegios
RUN useradd --create-home appuser && chown -R appuser /proyecto
USER appuser

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8501/_stcore/health || curl -fs http://localhost:8000/health || exit 1

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
