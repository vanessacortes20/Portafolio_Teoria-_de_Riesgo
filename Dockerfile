# =============================================================================
# RiskLab USTA — Imagen Docker multi-stage
# =============================================================================
# Stage 1 (builder): instala dependencias compilando lo que haga falta
#                    (cvxpy, scikit-learn, arch usan extensiones nativas).
# Stage 2 (runtime): solo Python + el venv ya construido. Sin compiladores,
#                    sin cache de pip y corre como usuario no-root.
# =============================================================================

# ---------- Stage 1: builder ----------
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build

# Toolchain mínimo para compilar wheels nativos (numpy/scipy/cvxpy/arch)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Aislamos las dependencias en un venv y lo copiamos al runtime
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt


# ---------- Stage 2: runtime ----------
FROM python:3.11.9-slim-bookworm

WORKDIR /app

# Variables de entorno productivas
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

# Usuario no-root para correr la app
RUN groupadd --system app \
    && useradd --system --gid app --no-create-home --home-dir /app app

# Trae el venv ya armado desde el builder (sin compiladores, sin pip cache)
COPY --from=builder /opt/venv /opt/venv

# Copia codigo de la aplicacion con ownership correcto
COPY --chown=app:app api/         ./api/
COPY --chown=app:app frontend/    ./frontend/

# Directorio de datos (SQLite). En produccion (Render) se monta un disco
# persistente sobre este path para que la BD sobreviva redeploys.
RUN mkdir -p /app/data && chown -R app:app /app/data

USER app

EXPOSE 8001

# Shell-form para resolver ${PORT} cuando el PaaS lo asigna; cae a 8001 local.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
