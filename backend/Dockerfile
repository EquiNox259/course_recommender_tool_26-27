FROM python:3.12-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   ENV PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
   CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --worker-class gthread --threads 2 --timeout 300