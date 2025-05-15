FROM python:3.12-slim

WORKDIR /code

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar uma versão específica do Poetry (1.7.1 é conhecida por ser estável)
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV PATH="$POETRY_HOME/bin:$PATH"

# Instalar Poetry usando curl (método mais confiável)
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -

# Copiar arquivos de configuração
COPY pyproject.toml poetry.lock .env /code/

# Gerar um novo arquivo lock e instalar dependências
RUN poetry lock && \
    poetry install --no-interaction --no-ansi

COPY ./app /code/app

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]