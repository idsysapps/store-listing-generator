FROM registry.access.redhat.com/ubi9/python:3.13

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN pip install uv && uv sync --no-dev --no-install-project

COPY src/ ./src/
COPY tests/ ./tests/

ENV PYTHONPATH=/app

CMD ["python", "-m", "store_listing.orchestration.tasks"]
