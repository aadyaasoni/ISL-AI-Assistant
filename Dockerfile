FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    ISL_FAILURE_LOG_PATH=reports/failure_events.jsonl

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-preprocessing.txt requirements-training.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && python -m pip install --no-cache-dir -r requirements-preprocessing.txt -r requirements-training.txt

COPY . .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/reports \
    && chown app:app /app/reports

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "web/server.py"]
