# ── Stage 1: builder (con compiladores y cache de pip) ────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Compiladores solo necesarios para wheels que no tengan binarios precompilados
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ── Stage 2: runtime (sin compiladores ni cache de pip) ───────────────────
FROM python:3.11-slim

WORKDIR /app

# Trae solo las dependencias instaladas, no compiladores
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copia solo lo estrictamente necesario para correr el backend
COPY backend/         ./backend/
COPY generate_data.py ./generate_data.py

# data/ se crea automáticamente al importar backend.app.database
RUN mkdir -p /app/data

EXPOSE 8001

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
