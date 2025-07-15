FROM python:3.10-slim

WORKDIR /app

COPY requirements-code-exec.txt .

RUN pip install --no-cache-dir -r requirements-code-exec.txt

CMD ["python", "runner.py"]