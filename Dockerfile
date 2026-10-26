FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    ISL_FAILURE_LOG_PATH=/tmp/failure_events.jsonl

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runtime.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install --no-cache-dir -r requirements-runtime.txt

COPY . .

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/reports \
    && chown app:app /app/reports

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os; from urllib.request import urlopen; urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/api/health', timeout=3)"

CMD ["python", "web/server.py"]
