FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt update && apt install -y git
RUN apt install ffmpeg -y
RUN pip install --no-cache-dir -U -r requirements.txt
# Run version update script during build
COPY update-version.sh /app/update-version.sh
RUN chmod +x /app/update-version.sh && /app/update-version.sh

COPY bot/ .

CMD ["python", "main.py"]