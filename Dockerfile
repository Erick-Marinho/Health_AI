FROM python:3.12-slim as builder

WORKDIR /app_build

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION}
ENV POETRY_NO_INTERACTION=1

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-dev --no-root

FROM python:3.12-slim as runtime

WORKDIR /code

ENV PORT="8000"
ENV WORKERS="4"
ENV LOG_LEVEL="info"
ENV GUNICORN_TIMEOUT="120"
ENV APP_MODULE="app.main:app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --shell /bin/bash --create-home appuser

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY ./app /code/app

USER appuser

EXPOSE ${PORT}

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "${APP_MODULE}", "--bind", "0.0.0.0:${PORT}", "--workers", "${WORKERS}", "--log-level", "${LOG_LEVEL}", "--timeout", "${GUNICORN_TIMEOUT}"]