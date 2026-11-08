FROM docker.io/library/python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_TELEMETRY=true

WORKDIR /app
COPY requirements.lock.txt ./
RUN python -m pip install pip==26.2.2 setuptools==78.1.0 wheel==0.45.1 \
    && python -m pip install -r requirements.lock.txt
COPY pyproject.toml ./
COPY mibici_equipo14 ./mibici_equipo14
RUN python -m pip install --no-deps --no-build-isolation . \
    && useradd --create-home --uid 10001 appuser
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
ENTRYPOINT ["python", "-m", "mibici_equipo14.serving"]
CMD ["--host", "0.0.0.0", "--model-version", "1"]
