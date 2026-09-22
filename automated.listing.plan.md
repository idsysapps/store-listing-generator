# Project Plan: Automated E-Commerce Apparel Trend Mining & Publishing System

## System Overview
An automated pipeline that monitors real-time search and social media trends (Google Trends, TikTok, Pinterest, Etsy autocomplete), processes data through a local DGX LLM node to extract non-infringing slogan mechanics, filters output through an automated USPTO Class 025 (Clothing) trademark gate, and auto-publishes approved listings directly to Amazon (SP-API), Etsy, and Shopify stores.

---

## Architecture Diagram
┌─────────────────────────────────────────────────────────┐
│                    TREND DATA INGESTION                 │
├─────────────────────────┬───────────────────────────────┤
│ Google Trends / SerpAPI │ Search Volume & Intent Spikes │
│ TikTok & Pinterest APIs │ Micro-Trends & Viral Phrases  │
│ Amazon Bestsellers API  │ BSR Velocity & Rank Tracking  │
└────────────┬────────────┴──────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│                  LOCAL DGX LLM NODE                     │
├─────────────────────────────────────────────────────────┤
│ Host: vLLM / Ollama                                     │
│ Models: Qwen 2.5 32B / Llama 3.1 70B                    │
│ Function: Pattern Extraction & Safe Slogan Synthesis    │
└────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│               AUTOMATED IP CLEARANCE GATE               │
├─────────────────────────────────────────────────────────┤
│ USPTO TSDR API Check (Class 025 - Apparel)              │
│ Fuzzy Matching (RapidFuzz) & Manual One-Click Review    │
└────────────┬────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│                  PUBLISHING & RENDERING                 │
├─────────────────────────────────────────────────────────┤
│ Dynamic Graphic Generator (PIL / ImageMagick 300DPI)    │
│ Multi-Channel Sync: Amazon SP-API / Shopify / Etsy      │
└─────────────────────────────────────────────────────────┘


---

## Milestones

1. **Milestone 1: Data Ingestion & Trend Scraping Engine** (Target: Weeks 1–2)
2. **Milestone 2: Local DGX LLM Processing & Prompt Pipeline** (Target: Weeks 3–4)
3. **Milestone 3: Automated IP Clearance & Trademark Gate** (Target: Weeks 5–6)
4. **Milestone 4: Automated Mockup Rendering & Asset Generation** (Target: Week 7)
5. **Milestone 5: Amazon SP-API & Multi-Channel Listing Engine** (Target: Weeks 8–9)
6. **Milestone 6: Orchestration, Dashboard & Monitoring** (Target: Week 10)

---

## Issues & Task Breakdown

### Milestone 1: Data Ingestion & Trend Scraping Engine

#### Issue 1.1: Build Google Trends & Search Intent Scraper
* **Type:** Feature
* **Description:** Implement scraper pipeline targeting high-intent search query spikes.
* **Tasks:**
  - [ ] Integrate `pytrends` or SerpAPI / DataForSEO.
  - [ ] Query rising terms combining niche seeds with keywords (`"funny t-shirt"`, `"hoodie"`, `"gift"`).
  - [ ] Store trend scores and daily search delta in local PostgreSQL database.

#### Issue 1.2: Build TikTok & Pinterest Micro-Trend Harvester
* **Type:** Feature
* **Description:** Collect viral audio, text overlays, and visual meme trends before they hit Amazon.
* **Tasks:**
  - [ ] Set up Apify TikTok/Pinterest Scraper actors.
  - [ ] Scrape text overlay data from videos under target seeds (`#shirttok`, `#momhumor`, `#pickleball`, `#gymhumor`).
  - [ ] Parse Etsy & Amazon search autocomplete APIs for real-time buyer queries.

#### Issue 1.3: Build Amazon Best Seller Rank (BSR) Scraper
* **Type:** Feature
* **Description:** Collect top-performing apparel listings from Amazon to measure BSR velocity.
* **Tasks:**
  - [ ] Set up proxy rotation via ScraperAPI/Apify to pull `Clothing, Shoes & Jewelry > Novelty & More`.
  - [ ] Extract ASIN, title, bullets, price, and BSR rank.
  - [ ] Identify items jumping >50% in BSR rank within 24 hours.

---

### Milestone 2: Local DGX LLM Processing & Prompt Pipeline

#### Issue 2.1: Deploy vLLM Server on Local DGX Hardware
* **Type:** Infrastructure
* **Description:** Set up vLLM server hosting Qwen 2.5 or Llama 3.1 on local DGX workstation.
* **Tasks:**
  - [ ] Install vLLM / Ollama serving environment with OpenAI-compatible API endpoints.
  - [ ] Benchmark token generation latency and context window processing.
  - [ ] Implement system service daemon (`systemd`) for continuous uptime.

#### Issue 2.2: Implement Trend Synthesis & Slogan Prompt Engineering
* **Type:** Feature
* **Description:** Create structured prompts to ingest raw trend JSON and output non-infringing slogan concepts.
* **Tasks:**
  - [ ] Develop JSON-in / JSON-out prompt templates for trend extraction.
  - [ ] Program model to identify underlying humor mechanics (self-deprecation, occupational pride, low-effort fitness).
  - [ ] Enforce strict output constraints: under 8 words, no punctuation, no brand/celebrity/lyrics references.

---

### Milestone 3: Automated IP Clearance & Trademark Gate

#### Issue 3.1: Build USPTO TSDR API Class 025 Validation Client
* **Type:** Feature
* **Description:** Programmatically query the USPTO database for active Class 025 (Clothing) trademarks.
* **Tasks:**
  - [ ] Connect Python service to USPTO TSDR REST API / bulk database dump.
  - [ ] Filter searches strictly by **Class 025 (Clothing, Footwear, Headwear)**.
  - [ ] Implement `RapidFuzz` Levenshtein distance check for phonetic/visual similarities.

#### Issue 3.2: Build One-Click Human Approval Dashboard
* **Type:** Feature
* **Description:** Build lightweight web UI (`Streamlit` or `FastAPI` + `Vue/Tailwind`) to review auto-cleared slogans.
* **Tasks:**
  - [ ] Display cleared slogans alongside source trend data.
  - [ ] Provide one-click "Approve", "Edit", or "Reject" actions.
  - [ ] Queue approved slogans directly into mockup generator pipeline.

---

### Milestone 4: Automated Mockup Rendering & Asset Generation

#### Issue 4.1: Build High-Resolution Typography Rendering Engine
* **Type:** Feature
* **Description:** Automate creation of 300 DPI transparent PNGs formatted for Direct-to-Film (DTF) and print-on-demand.
* **Tasks:**
  - [ ] Create Python (`Pillow` / `ImageMagick`) script mapping slogans onto standardized canvas size (4500x5400px).
  - [ ] Implement automatic font-pairing logic (bold slab sans + script/accent fonts).
  - [ ] Export 300 DPI transparent PNGs and color-separated print-ready files.

---

### Milestone 5: Amazon SP-API & Multi-Channel Listing Engine

#### Issue 5.1: Integrate Amazon Selling Partner API (SP-API)
* **Type:** Feature
* **Description:** Automate draft listing creation directly in Seller Central.
* **Tasks:**
  - [ ] Authenticate with Amazon SP-API (`Listings Items API`).
  - [ ] Map generated titles, bullets, and keywords to Amazon apparel taxonomy.
  - [ ] Upload generated PNG renders to product media endpoints.

#### Issue 5.2: Multi-Channel Cross-Posting (Shopify & Etsy)
* **Type:** Feature
* **Description:** Push cleared slogans simultaneously to Shopify and Etsy to maximize organic reach.
* **Tasks:**
  - [ ] Connect Shopify REST Admin API for automated product drafting.
  - [ ] Connect Etsy v3 API endpoint for automated listing creation.

---

### Milestone 6: Orchestration, Dashboard & Monitoring

#### Issue 6.1: Celery / n8n Workflow Orchestration
* **Type:** Feature
* **Description:** Tie scraping, LLM generation, IP validation, rendering, and publishing into a scheduled pipeline.
* **Tasks:**
  - [ ] Configure `Celery` + `Redis` or `n8n` workflow engine.
  - [ ] Schedule daily 06:00 AM trend scrape and slogan processing batch job.
  - [ ] Set up notification webhooks (Discord / Slack) for pipeline execution status and daily approved list counts.