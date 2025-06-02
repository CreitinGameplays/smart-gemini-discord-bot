FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update && apt install -y git
RUN apt install ffmpeg -y
RUN pip install --no-cache-dir -U -r requirements.txt

COPY bot/ .

CMD ["python", "main.py"]