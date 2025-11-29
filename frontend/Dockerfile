FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PORT=8501

EXPOSE 8501

RUN mkdir -p .streamlit

CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0