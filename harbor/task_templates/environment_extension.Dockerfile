FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash coreutils curl git procps tmux asciinema build-essential \
    && uv tool install mini-swe-agent==2.4.6 --with 'litellm[proxy]' \
    && rm -rf /var/lib/apt/lists/* /root/.cache/uv

ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /app
COPY workspace/ /app/
