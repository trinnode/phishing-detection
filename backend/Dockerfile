FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

RUN mkdir -p models/saved results cache/structural

EXPOSE 5000

CMD ["python", "start.py"]
