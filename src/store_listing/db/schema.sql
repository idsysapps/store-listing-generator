-- Create database
CREATE DATABASE store_listing;

-- Connect to the database
\c store_listing;

-- Table: trend_queries
-- Stores the seed keywords and generated search queries, tagged with the
-- ingester `source` (google/tiktok/pinterest/amazon/etsy)
CREATE TABLE trend_queries (
    id SERIAL PRIMARY KEY,
    seed_keyword VARCHAR(255) NOT NULL,
    query VARCHAR(500) NOT NULL,
    source VARCHAR(20) NOT NULL DEFAULT 'google',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(seed_keyword, query)
);

-- Table: trend_scores
-- Stores trend scores, deltas, and regional data for each query,
-- tagged with the ingester `source` (google/tiktok/pinterest/amazon)
CREATE TABLE trend_scores (
    id SERIAL PRIMARY KEY,
    query_id INTEGER NOT NULL REFERENCES trend_queries(id) ON DELETE CASCADE,
    score BIGINT NOT NULL,
    delta BIGINT DEFAULT 0,
    region VARCHAR(255) DEFAULT 'US',
    query_type VARCHAR(10) DEFAULT 'interest',
    source VARCHAR(20) NOT NULL DEFAULT 'google',
    trend_direction INTEGER,
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX idx_trend_scores_query_id ON trend_scores(query_id);
CREATE INDEX idx_trend_scores_fetched_at ON trend_scores(fetched_at);
CREATE INDEX idx_trend_queries_seed ON trend_queries(seed_keyword);

-- Table: seed_candidates
-- Stores every discovered query across all sources with a promotion score
-- and lifecycle status. Promotion is handled by the promote_seeds task.
CREATE TABLE seed_candidates (
    id SERIAL PRIMARY KEY,
    query VARCHAR(500) NOT NULL UNIQUE,
    source_seed VARCHAR(255) NOT NULL,
    source VARCHAR(20) NOT NULL DEFAULT 'google',
    query_type VARCHAR(10),
    score BIGINT,
    delta BIGINT,
    promotion_score BIGINT NOT NULL DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    promoted_at TIMESTAMP WITH TIME ZONE,
    archived_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_seed_candidates_status ON seed_candidates(status);
CREATE INDEX idx_seed_candidates_source ON seed_candidates(source);
CREATE INDEX idx_seed_candidates_promotion_score ON seed_candidates(promotion_score);

-- Table: active_seeds
-- Source of truth for the active seed set (max enforced by promote_seeds)
CREATE TABLE active_seeds (
    id SERIAL PRIMARY KEY,
    query VARCHAR(500) NOT NULL UNIQUE,
    promotion_score BIGINT NOT NULL DEFAULT 0,
    promoted_from_candidate_id INTEGER REFERENCES seed_candidates(id),
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    archived_at TIMESTAMP WITH TIME ZONE
);

-- Starter seeds (fallback seed set for a fresh database)
INSERT INTO active_seeds (query, promotion_score) VALUES
    ('funny t-shirt', 0),
    ('hoodie', 0),
    ('gift', 0),
    ('socks', 0),
    ('leggings', 0),
    ('custom mugs', 0),
    ('room decor', 0),
    ('wall art', 0),
    ('mom humor', 0),
    ('dad jokes', 0),
    ('gym fitness', 0),
    ('running', 0),
    ('quirky gifts', 0),
    ('novelty socks', 0),
    ('personalized gifts', 0)
ON CONFLICT (query) DO NOTHING;

-- Table: source_health
-- Per-source harvest health so the fix loop opens GitHub issues automatically
-- after repeated failures/empties (see orchestration/source_health.py).
CREATE TABLE source_health (
    source VARCHAR(20) PRIMARY KEY,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    consecutive_empty INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    last_error_at TIMESTAMP WITH TIME ZONE,
    last_success_at TIMESTAMP WITH TIME ZONE,
    issue_open BOOLEAN NOT NULL DEFAULT FALSE,
    last_issue_url TEXT,
    last_issue_number INTEGER,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table: seed_curation_log
-- Tracks LLM and calendar seed curation decisions for audit/tuning
CREATE TABLE seed_curation_log (
    id SERIAL PRIMARY KEY,
    candidate_id INTEGER REFERENCES seed_candidates(id),
    action VARCHAR(20) NOT NULL,
    reasoning TEXT,
    event_context VARCHAR(100),
    llm_model VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_curation_log_action ON seed_curation_log(action);
CREATE INDEX idx_curation_log_event ON seed_curation_log(event_context);

-- Table: seed_product_tags
-- Tags active seeds with which production methods the design fits.
-- Allows filtering seeds by what we can produce now (e.g. dtf_apparel)
-- without discarding data for future production methods.
CREATE TABLE seed_product_tags (
    id SERIAL PRIMARY KEY,
    active_seed_id INTEGER NOT NULL REFERENCES active_seeds(id) ON DELETE CASCADE,
    tag VARCHAR(30) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(active_seed_id, tag)
);

CREATE INDEX idx_seed_product_tags_tag ON seed_product_tags(tag);
CREATE INDEX idx_seed_product_tags_seed ON seed_product_tags(active_seed_id);

-- Table: design_briefs
-- Stores LLM-generated design concepts synthesized from trend signals.
CREATE TABLE design_briefs (
    id SERIAL PRIMARY KEY,
    concept TEXT NOT NULL,
    product_type VARCHAR(50) NOT NULL,
    specific_products TEXT[],
    audience TEXT,
    visual_style TEXT,
    confidence INTEGER NOT NULL DEFAULT 0,
    reasoning TEXT,
    llm_model VARCHAR(100),
    batch_id VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_design_briefs_product_type ON design_briefs(product_type);
CREATE INDEX idx_design_briefs_created_at ON design_briefs(created_at);
CREATE INDEX idx_design_briefs_batch_id ON design_briefs(batch_id);

-- Table: design_brief_sources
-- Links design briefs to the active seeds that inspired them.
CREATE TABLE design_brief_sources (
    id SERIAL PRIMARY KEY,
    brief_id INTEGER NOT NULL REFERENCES design_briefs(id) ON DELETE CASCADE,
    active_seed_id INTEGER NOT NULL REFERENCES active_seeds(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(brief_id, active_seed_id)
);

CREATE INDEX idx_design_brief_sources_brief ON design_brief_sources(brief_id);
CREATE INDEX idx_design_brief_sources_seed ON design_brief_sources(active_seed_id);
