FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update
RUN apt install ffmpeg
RUN pip install --no-cache-dir -U -r requirements.txt

COPY bot/ .

CMD ["python", "v4.0.py"]
