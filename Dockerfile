FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "-m", "waitress", "--listen=0.0.0.0:8001", "--threads=8", "backend_api.wsgi:application"]
