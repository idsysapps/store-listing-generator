# Store Listing Generator

[![CI](https://github.com/idsysapps/store-listing-generator/actions/workflows/ci.yml/badge.svg)](https://github.com/idsysapps/store-listing-generator/actions)
[![Python Version](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)

Automated E-Commerce Apparel Trend Mining & Publishing System

## Overview

An automated pipeline that monitors real-time search and social media trends (Google Trends, TikTok, Pinterest, Etsy autocomplete), processes data through a local DGX LLM node to extract non-infringing slogan mechanics, filters output through an automated USPTO Class 025 (Clothing) trademark gate, and auto-publishes approved listings directly to Amazon (SP-API), Etsy, and Shopify stores.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TREND DATA INGESTION                     │
├─────────────────────────┬───────────────────────────────────┤
│ Google Trends / SerpAPI │ Search Volume & Intent Spikes    │
│ TikTok & Pinterest APIs │ Micro-Trends & Viral Phrases     │
│ Amazon Bestsellers API  │ BSR Velocity & Rank Tracking    │
└────────────┬────────────┴───────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  LOCAL DGX LLM NODE                          │
├─────────────────────────────────────────────────────────────┤
│ Host: vLLM / Ollama                                          │
│ Models: Qwen 2.5 32B / Llama 3.1 70B                        │
│ Function: Pattern Extraction & Safe Slogan Synthesis        │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│               AUTOMATED IP CLEARANCE GATE                   │
├─────────────────────────────────────────────────────────────┤
│ USPTO TSDR API Check (Class 025 - Apparel)                  │
│ Fuzzy Matching (RapidFuzz) & Manual One-Click Review         │
└────────────┬────────────────────────────────────────────────┘
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  PUBLISHING & RENDERING                     │
├─────────────────────────────────────────────────────────────┤
│ Dynamic Graphic Generator (PIL / ImageMagick 300DPI)         │
│ Multi-Channel Sync: Amazon SP-API / Shopify / Etsy          │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

| Requirement | Version | Description |
|------------|---------|-------------|
| Python | 3.11+ (3.13 recommended) | Language runtime |
| podman | 4.0+ | Container runtime |
| PostgreSQL | 16+ | Database |
| Redis | 7+ | Celery broker |
| GitHub CLI | Latest | For contributing |

## Quick Start

```bash
# Clone the repository
git clone https://github.com/idsysapps/store-listing-generator.git
cd store-listing-generator

# Start database and Redis services
podman-compose up -d postgres redis

# Setup database schema
psql -h localhost -U postgres -d store_listing -f src/store_listing/db/schema.sql

# Install dependencies
pip install uv
uv sync --dev

# Run tests
uv run pytest

# Run linter
uv run ruff check .
```

## Local Development

### Starting Services

```bash
# Start all services in background
podman-compose up -d

# View logs
podman-compose logs -f

# Stop services
podman-compose down

# Rebuild and start
podman-compose up -d --build
```

### Running Celery Tasks

```bash
# Start Celery worker (in one terminal)
celery -A store_listing.orchestration.tasks worker --loglevel=info

# Start Celery beat scheduler (in another terminal)
celery -A store_listing.orchestration.tasks beat --loglevel=info

# Trigger manual trend harvest
celery -A store_listing.orchestration.tasks call fetch_trends_manual --args='[["hoodie", "funny t-shirt", "gift"]]'

# Trigger scheduled daily harvest (runs at 6 AM UTC)
# This is handled automatically by the beat scheduler
```

### Database Schema

The database schema is located at `src/store_listing/db/schema.sql`.

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d store_listing

# Run schema
\i src/store_listing/db/schema.sql
```

**Tables:**

| Table | Description |
|-------|-------------|
| `trend_queries` | Seed keywords and generated search queries |
| `trend_scores` | Trend scores, deltas, and regional data |

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Run specific test file
uv run pytest tests/unit/test_google_trends.py

# Run tests matching a pattern
uv run pytest -k "test_fetch"
```

## Helm Deployment

The application can be deployed to Kubernetes/OpenShift using the Helm chart in `helm/store-listing/`.

### Installing from GitHub Pages

```bash
# Add the Helm repository (master/development builds)
helm repo add store-listing-generator https://idsysapps.github.io/store-listing-generator/master

# Update dependencies
helm repo update

# Install from master branch (latest development build)
helm install store-listing store-listing-generator/store-listing

# Install a specific release version
helm install store-listing https://idsysapps.github.io/store-listing-generator/store-listing-<version>.tgz
```

### Helm Chart Repository URLs

| Branch | URL | Description |
|--------|-----|-------------|
| `master` | `https://idsysapps.github.io/store-listing-generator/master` | Development builds with latest changes |
| Releases | `https://idsysapps.github.io/store-listing-generator` | Official releases via GitHub Releases |

### Development Builds (master)

Master builds publish charts with version `0.1.0-master+sha<sha>` and use Docker digest references for images, ensuring `helm upgrade` always triggers pod redeployment.

### Release Builds

Release builds (triggered by git tags like `v0.1.8`) publish to the root URL with semantic versions and use immutable Docker tag references.

## CI/CD

All CI workflows are located in `.github/workflows/`.

### Pipeline Jobs

| Job | Trigger | Purpose |
|-----|---------|---------|
| `lint-and-test` | All pushes and PRs | Runs ruff, pyright, and pytest |
| `build-and-push` | Push to master or tag | Builds and pushes Docker images to Quay.io |
| `helm-chart-publish` | After build-and-push | Publishes Helm chart to GitHub Pages |

### Running CI Locally

Before pushing, run pre-commit checks:

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit checks
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --files src/
```

## Project Structure

```
store-listing-generator/
├── src/store_listing/
│   ├── db/
│   │   └── schema.sql           # PostgreSQL schema
│   ├── ingest/
│   │   └── trends/
│   │       ├── google_trends.py # Google Trends client
│   │       ├── schemas.py       # Pydantic models
│   │       └── __init__.py
│   └── orchestration/
│       ├── tasks.py             # Celery tasks
│       └── __init__.py
├── tests/
│   └── unit/
│       └── test_google_trends.py
├── docker-compose.yml            # Local development services
├── Dockerfile                   # Container image definition
├── pyproject.toml              # Project configuration
├── AGENTS.md                   # Development standards
└── README.md                   # This file
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_HOST` | localhost | PostgreSQL host |
| `DATABASE_PORT` | 5432 | PostgreSQL port |
| `DATABASE_NAME` | store_listing | Database name |
| `DATABASE_USER` | postgres | Database user |
| `DATABASE_PASSWORD` | (empty) | Database password |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection URL |

## Contributing

All contributions must follow the standards defined in [AGENTS.md](./AGENTS.md).

### Git Workflow

1. Create a GitHub issue for all work
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes following TDD (write failing tests first)
4. Run lint and tests locally before committing
5. Commit with conventional commit format: `feat: description (closes #123)`
6. Push and create PR via GitHub CLI

### Code Standards

- **Type hints** required on all public function signatures
- **Pydantic models** for API/request/response validation
- **pytest** for testing with async/await support
- **ruff** for linting and formatting
- **pyright** for static type checking

## License

MIT
