FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY requirements-code-exec.txt .
RUN apt update && apt install -y git
RUN apt install ffmpeg -y
RUN pip install --no-cache-dir -U -r requirements.txt
RUN pip install --no-cache-dir -r requirements-code-exec.txt

COPY bot/ .

CMD ["python", "main.py"]
CMD ["python", "runner.py"]