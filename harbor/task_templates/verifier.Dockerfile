FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash coreutils procps \
    && rm -rf /var/lib/apt/lists/*

COPY . /tests/
RUN chmod 755 /tests/test.sh /tests/verify.py
WORKDIR /app

