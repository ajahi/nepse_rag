# NEPSE Chat — FastAPI app
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; wheels cover faiss/psycopg2/cohere.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (nepse_kb/ index is copied in so RAG works without a rebuild).
COPY . .

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
