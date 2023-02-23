FROM python:3.9.16-buster

ARG YOUR_ENV

ENV YOUR_ENV=${YOUR_ENV} \
    # python:
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=1 \
    # pip:
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # poetry:
    POETRY_VERSION=1.3.2 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'

SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
ENV POLUS_LOG="INFO"
ENV DATA_DIR="/data"

RUN apt-get update && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
    bash \
    brotli \
    build-essential \
    curl \
    gettext \
    git \
    libpq-dev \
    # Installing `poetry` package manager:
    # https://github.com/python-poetry/poetry
    && curl -sSL 'https://install.python-poetry.org' | python - \
    && poetry --version \
    # Cleaning cache:
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

WORKDIR "ome-converter-plugin"
RUN mkdir "ome-converter-plugin"

# Copy only requirements, to cache them in docker layer
COPY ./poetry.lock ./pyproject.toml ./README.md /ome-converter-plugin/
RUN ls -la

COPY src/* src/polus/.
RUN ls -la src/polus/plugins/formats/ome_converter/**

RUN poetry config virtualenvs.create false && \
    poetry run pip install -U pip && \
    poetry install \
    $(if [ "$YOUR_ENV" = 'production' ]; then echo '--only main'; fi) \
    --no-interaction --no-ansi

# Running as non-root user:
USER 1000

ENTRYPOINT ["python", "-m", "polus.plugins.formats.ome_converter"]
