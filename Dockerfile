FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update && apt install -y git
RUN apt install ffmpeg -y
RUN pip install --no-cache-dir -r requirements.txt
RUN git describe --tags --always --dirty > .version

COPY bot/ .

CMD ["python", "main.py"]