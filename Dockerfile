FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# torch CPU-only: evita di scaricare i pacchetti CUDA (~1GB+), inutili senza GPU
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ src/

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
