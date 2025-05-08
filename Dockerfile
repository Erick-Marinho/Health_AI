FROM python:3.12-slim

WORKDIR /code

ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock* /code/

RUN poetry install --no-interaction --no-ansi --only main

COPY ./app /code/app

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]