FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 libpq5 libxcb1 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --gid 1000 app && adduser --disabled-password --gecos "" --uid 1000 --gid 1000 app

WORKDIR /app

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY backend/app /app/app
COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic

USER app
