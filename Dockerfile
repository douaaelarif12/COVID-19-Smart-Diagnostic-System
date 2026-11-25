FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8501 8000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
