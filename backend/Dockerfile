FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 libpq5 libxcb1 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1000 app && adduser --disabled-password --gecos "" --uid 1000 --gid 1000 app

WORKDIR /app

FROM base AS dependencies
RUN python -m venv /opt/phosca
ENV PATH="/opt/phosca/bin:$PATH"

ARG TORCH_WHEEL_INDEX=""
COPY backend/requirements.txt /app/requirements.txt
RUN --mount=type=secret,id=registry_ca \
    if [ -f /run/secrets/registry_ca ]; then export PIP_CERT=/run/secrets/registry_ca; fi \
    && pip install --upgrade pip \
    && if [ -n "$TORCH_WHEEL_INDEX" ]; then \
        pip install --index-url "$TORCH_WHEEL_INDEX" torch torchvision \
        && pip install cython wheel \
        && pip install --no-build-isolation --no-deps basicsr==1.4.2; \
    fi \
    && pip install -r /app/requirements.txt

FROM base
ENV PATH="/opt/phosca/bin:$PATH"
COPY backend/app /app/app
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic
COPY scripts/create_api_key.py /app/scripts/create_api_key.py
COPY --from=dependencies /opt/phosca /opt/phosca

USER app
