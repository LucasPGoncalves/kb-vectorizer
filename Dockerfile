# === Builder ===
FROM python:3.12-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY uv.lock ./

RUN uv sync --extra dev --frozen --no-install-project

COPY src ./src
RUN uv sync --frozen

# === Runtime ===
FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
ENV PATH="/app/.venv/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y bash

ENTRYPOINT ["kb-vectorizer"]
CMD ["--help"]
