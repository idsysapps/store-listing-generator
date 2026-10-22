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

## Design Image Pipeline

Design briefs are rendered on the DGX with ComfyUI + Flux Schnell, and the
headline and the tagline are drawn on top of the artwork with a font - the
model cannot spell, so it is never asked to.

| Layout | What is drawn |
|--------|---------------|
| `full_bleed` | The illustration only, no text |
| `text_top` | The illustration with a headline at the top |
| `text_top_bottom` | The illustration with a headline at the top and a tagline at the bottom |
| `text_only` | The text alone on a blank canvas - the image model is never called |

Three gates stand between a render and the print file:

- **The artwork gate** reads the render back and rejects it when it shows the
  product instead of the design (a bag, a hoodie, a mug). A brief whose scene
  is unusable is rendered as a typography-only design when it has text, and
  left without an image when it has none. The `concept` is never used as the
  visual prompt: a concept names the design idea, and a concept that names
  the product draws the product.
- **The lettering gate** rejects the render when the artwork carries words of
  its own. The image model cannot spell, and the text of the design is drawn
  on the artwork afterwards with a font, so a letter the model drew is a
  letter that is wrong. The scene is stripped of the words it asks the model
  to spell, so the lettering does not happen in the first place.
- **The text gate** reads the composited text back and rejects the image when
  the headline cannot be verified. The words have to be there: a word that is
  misspelled fails even when the characters look close enough. A missing
  tagline is reported but does not reject the image.

The gates can be tuned with `ARTWORK_VALIDATION_ENABLED`,
`ARTWORK_TEXT_VALIDATION_ENABLED`, `MAX_ARTWORK_REGENERATION_ATTEMPTS`,
`MAX_LETTERING_REGENERATION_ATTEMPTS`, `MAX_TEXT_REGENERATION_ATTEMPTS` and
`REQUIRE_TAGLINE_TEXT` in the Helm chart.

## The seed pool and the brief mix

A batch of briefs is only as good as the pool it is made of, and the pool used
to be nothing but a calendar: every holiday injected its whole seed list every
day, and the briefs that came out were all seasonal. The pool now keeps three
kinds of seeds apart, and a seed says which kind it is.

| Kind | Where it comes from | How it behaves |
|------|--------------------|----------------|
| `calendar` | A holiday of the calendar | A few seeds a day, rotating through the list of the event; it leaves the pool when its day has passed |
| `evergreen` | A standing theme (pets, careers, hobbies, food, home, sports) | Injected every day; never archived below the evergreen reserve |
| `trend` | A harvested trend | Ranked by its promotion score |
| `discovery` | A trend found without a seed (the trending list, a theme of the marketplace) | Ranked higher than the same signal harvested from a seed, and never used to harvest again |

The brief task is given a mix of those seeds instead of the highest scoring
ones, and a batch is only allowed a few briefs per event. A batch that misses
the evergreen minimum is asked for once more before it is accepted.

The mix of the last days can be read at any time:

```bash
python -m store_listing.orchestration.brief_mix_report --days 14
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
| `APIFY_API_TOKEN` | (empty) | Apify API token; enables the Apify TikTok/Pinterest actors when set (falls back to free scraping when empty) |
| `APIFY_ACTOR_TIKTOK_HASHTAG` | `clockworks/tiktok-hashtag-scraper` | Apify actor ID for TikTok hashtag scraping |
| `APIFY_ACTOR_PINTEREST` | `automation-lab/pinterest-scraper` | Apify actor ID for Pinterest search scraping |
| `ETSY_SUGGEST_URL` | `https://www.etsy.com/autosuggest` | Etsy search autocomplete endpoint |
| `APIFY_ACTOR_X` | `xquik/x-tweet-scraper` | Apify actor ID for X (Twitter) search scraping |
| `TIKTOK_RESULTS_PER_PAGE` | `20` | Caps videos returned per TikTok hashtag (Apify bills per video; keeps spend low) |
| `TIKTOK_EXTRA_HASHTAGS` | (empty) | Comma-separated curated hashtags to add to the active seed set |
| `SOURCE_HEALTH_ENABLED` | `true` | Track per-source harvest health and auto-file GitHub issues on repeated failures |
| `HEALTH_FAILURE_THRESHOLD` | `2` | Consecutive failures before an `[auto-health]` issue is opened |
| `HEALTH_EMPTY_THRESHOLD` | `3` | Consecutive empty harvests before an `[auto-health]` issue is opened |
| `GITHUB_TOKEN` | (empty) | Token used to open auto-health issues (requires `GITHUB_REPO` too) |
| `GITHUB_REPO` | (empty) | `owner/repo` to file auto-health issues against |

### LLM models

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint (the DGX proxy in production) |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | The model that writes the design briefs |
| `LLM_CURATION_MODEL` | (empty, follows `LLM_MODEL`) | The model that curates the seeds |
| `VISION_LLM_MODEL` | `qwen_vl_ocr` | The vision model that verifies the rendered text |

**The model names come from the DGX proxy.** The proxy routes a model name to
the vLLM instance behind it, and the names it routes are declared in
[`deploy/dgx/openai-proxy.py`](deploy/dgx/openai-proxy.py) (`MODEL_ROUTES`).
A model the proxy does not route is refused, and every brief and curation
call fails - which is what happened when the cluster pointed at
`qwen2.5:32b-instruct-q4_K_M` (issue #194).

Check the served models before a rollout:

```bash
curl -s http://<proxy>:9000/v1/models | jq -r '.data[].id'
```

The app does the same check: before an LLM task it asks the endpoint which
models it serves and logs an error naming the served ones when the configured
model is not among them. A refused call reports the same thing instead of a
generic LLM failure.

The model that answered is stored with the brief (`design_briefs.llm_model`)
and with the curation decision (`seed_curation_log.llm_model`). To check that
the column is filled in production:

```sql
SELECT llm_model, count(*), max(created_at) FROM design_briefs GROUP BY 1;
```

### Product types in production

The store focuses on one production method at a time: the DTF apparel batches
are the ones that sell, and the sublimation ones wait.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLED_PRODUCT_TYPES` | (empty, every known type) | Comma-separated product types the pipeline produces, e.g. `dtf_apparel` |

The scope comes from the database when a row is there, and from the
environment otherwise, so it can be changed without a redeploy:

```graphql
query {
  enabledProductTypes
}

mutation {
  setEnabledProductTypes(types: ["dtf_apparel"])
}
```

The setting is stored in the `app_settings` table:

```sql
SELECT key, value, updated_at FROM app_settings;
```

A name the pipeline does not know is ignored with a warning, and a scope that
names nothing usable falls back to the environment.

### The materials of a brief

A design brief says which substrate it should be printed on, and the client
can read that recommendation and the catalogue it comes from:

```graphql
query {
  designBriefs(productType: "dtf_apparel") {
    items {
      concept
      productType
      recommendedMaterials {
        externalId
        name
        color
        substrateType
        unitCost
      }
    }
  }
}

query {
  fabricationMaterials(productType: "dtf_apparel") {
    externalId
    name
    brand
    substrateType
    color
    sizes
    productTypes
    unitCost
  }
}
```

A material can serve several product types, so `productTypes` is a list. A
recommendation the model invented, or one that does not serve the product type
of the brief, is dropped when the brief is read.

### Seed pool and brief mix
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_CALENDAR_SEEDS_PER_EVENT_PER_DAY` | 3 | Calendar seeds one holiday may inject per day |
| `LLM_EVERGREEN_SEEDS_PER_DAY` | 4 | Standing theme seeds injected per day |
| `LLM_EVERGREEN_RESERVE` | 10 | Evergreen seeds the pool always keeps |
| `LLM_BRIEF_SEED_SEASONAL_QUOTA` | 6 | Seasonal seeds the brief task may see in one batch |
| `LLM_BRIEF_SEED_EVERGREEN_QUOTA` | 8 | Evergreen seeds the brief task may see in one batch |
| `LLM_BRIEF_SEED_TREND_QUOTA` | 6 | Trend seeds the brief task may see in one batch |
| `LLM_MAX_BRIEFS_PER_THEME` | 2 | Briefs of one batch that may be tied to the same event |
| `LLM_MIN_EVERGREEN_BRIEFS` | 4 | Briefs of one batch that may not be tied to an event |
| `LLM_BRIEF_MIX_REGENERATE_ONCE` | true | Ask for a second batch when the first misses the evergreen minimum |
| `LLM_DISCOVERY_BONUS` | 1000 | What a discovery is worth on top of the score of its source |
| `LLM_ECHO_FACTOR` | 0.5 | What is left of the score of a candidate that only repeats its seed |
| `DISCOVERY_CATEGORIES_PER_RUN` | 4 | Standing themes one discovery run asks for |

### Image generation (ComfyUI on the DGX)

Design brief images are rendered by **ComfyUI + Flux Schnell** on the DGX Spark and
verified with a vision model before they are uploaded to S3. Leonardo.ai, the
Ideogram text routing and the AWS Bedrock image calls have been removed.

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_BASE_URL` | `http://localhost:8188` | ComfyUI on the DGX |
| `COMFYUI_TIMEOUT_SECONDS` | `300` | Per-workflow wall-clock limit |
| `COMFYUI_POLL_INTERVAL_SECONDS` | `1.0` | How often the workflow history is polled |
| `COMFYUI_UNET_NAME` | `flux1-schnell.safetensors` | Flux diffusion model |
| `COMFYUI_CLIP_NAME1` | `t5xxl_fp8_e4m3fn.safetensors` | T5-XXL text encoder |
| `COMFYUI_CLIP_NAME2` | `clip_l.safetensors` | CLIP-L text encoder |
| `COMFYUI_VAE_NAME` | `ae.safetensors` | Flux VAE |
| `COMFYUI_BG_MODEL_NAME` | (auto-discovered) | BiRefNet background removal model; discovered from the server when unset |
| `COMFYUI_BG_ALPHA_BLACK` | `24` | Alpha black point for the cut-out; clears the residual background haze |
| `COMFYUI_BG_ALPHA_WHITE` | `240` | Alpha white point for the cut-out |
| `COMFYUI_BG_ERODE_PIXELS` | `1` | Pixels the cut-out is eroded by, to drop the dark halo around the subject |
| `VISION_LLM_BASE_URL` | `LLM_BASE_URL` | OpenAI-compatible base URL for the vision model (the DGX proxy) |
| `VISION_LLM_MODEL` | `qwen_vl_ocr` | Vision model name |
| `VISION_LLM_API_KEY` | `LLM_API_KEY` | Usually a dummy for a local proxy |
| `TEXT_MATCH_THRESHOLD` | `0.85` | Similarity threshold for the rendered text to count as a match |
| `MAX_TEXT_REGENERATION_ATTEMPTS` | `2` | Retries after a text mismatch (3 generations max) |
| `ARTWORK_TEXT_VALIDATION_ENABLED` | `true` | Reject an artwork that carries letters of its own |
| `MAX_LETTERING_REGENERATION_ATTEMPTS` | `1` | Retries after the artwork came back with lettering in it |

The OCR gate **fails closed**: when the text rendered into the image cannot be
verified within the retry budget, the image is not uploaded and the brief is
counted as an error.

`dtf_apparel` designs are rendered on a flat saturated magenta background that is
keyed out to produce the transparent PNG, so white artwork and white text stay
visible. The `sublimation` and `sticker_vinyl` designs are printed full-bleed on
white paper. The DGX side is documented in
[`deploy/dgx/COMFYUI_SETUP.md`](deploy/dgx/COMFYUI_SETUP.md).

## Source Health & Free-Path-First Harvesting

Each scheduled harvest task records an outcome (`success` / `failure` / `empty`)
for its source (google, tiktok, pinterest, marketplace, x) into the
`source_health` table. After `HEALTH_FAILURE_THRESHOLD` consecutive failures (or
`HEALTH_EMPTY_THRESHOLD` consecutive empty runs) the tracker opens a labeled
`[auto-health] <source> harvester is failing consistently` GitHub issue with a
runbook, de-duplicating while the issue is open. Closing the issue resets the
tracker so a later regression re-opens it after the next `HEALTH_FAILURE_THRESHOLD`
failures.

Sources prefer the free web-gateway path first; the paid Apify actors are only
used when `APIFY_API_TOKEN` is set. If a source is blocked (for example the
free TikTok tag pages), its issue becomes the work item for the fix loop.
Harvest health is tracked even without `GITHUB_TOKEN`; issue creation is how
the `[auto-health]` tickets reach GitHub. See
`src/store_listing/orchestration/source_health.py`.

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
