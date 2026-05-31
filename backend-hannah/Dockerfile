FROM python:3.11-slim

# Dependencias de sistema para audio y compilacion
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python (cacheado si requirements no cambia)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo fuente
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
