# Build stage
FROM registry.access.redhat.com/hi/python:3.14-builder AS builder

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN pip install --user uv && \
    export PATH="$HOME/.local/bin:$PATH" && \
    uv sync --no-dev --no-install-project

COPY src/ ./src/
COPY tests/ ./tests/

# Runtime stage
FROM registry.access.redhat.com/hi/python:3.14

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "store_listing.orchestration.tasks"]
