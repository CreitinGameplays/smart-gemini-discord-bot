FROM python:3.10-slim

WORKDIR /app

ARG GIT_REPO_URL="https://github.com/CreitinGameplays/smart-gemini-discord-bot"

COPY requirements.txt .
RUN apt update && apt install -y git
RUN apt install ffmpeg -y
RUN pip install --no-cache-dir -r requirements.txt

RUN git -c 'versionsort.suffix=-' ls-remote --exit-code --refs --sort='version:refname' --tags ${GIT_REPO_URL} '*.*.*' \
    | tail --lines=1 \
    | cut --delimiter='/' --fields=3 \
    > .version || echo "null" > .version

COPY bot/ .

CMD ["python", "main.py"]