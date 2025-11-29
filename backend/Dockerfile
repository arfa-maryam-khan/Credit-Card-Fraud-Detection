FROM python:3.9-slim

WORKDIR /app

# Copy requirements from backend directory
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Copy model artifacts
COPY models/artifacts/ ./models/artifacts/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD uvicorn main:app --host 0.0.0.0 --port $PORT