# --- Builder Stage ---
FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG RAILWAY_GIT_COMMIT_SHA
RUN git clone https://github.com/CreitinGameplays/smart-gemini-discord-bot .
RUN if [ -n "$RAILWAY_GIT_COMMIT_SHA" ]; then git checkout $RAILWAY_GIT_COMMIT_SHA; fi
RUN git describe --tags --always --dirty > .version

RUN [ -s .version ] || (echo "ERROR: .version file is empty or not created." && exit 1)

# --- Final Stage --- 
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /app/bot/ .
COPY --from=builder /app/.version .

CMD ["python", "main.py"]