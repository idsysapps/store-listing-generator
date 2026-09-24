-- Create database
CREATE DATABASE store_listing;

-- Connect to the database
\c store_listing;

-- Table: trend_queries
-- Stores the seed keywords and generated search queries
CREATE TABLE trend_queries (
    id SERIAL PRIMARY KEY,
    seed_keyword VARCHAR(255) NOT NULL,
    query VARCHAR(500) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(seed_keyword, query)
);

-- Table: trend_scores
-- Stores trend scores, deltas, and regional data for each query
CREATE TABLE trend_scores (
    id SERIAL PRIMARY KEY,
    query_id INTEGER NOT NULL REFERENCES trend_queries(id) ON DELETE CASCADE,
    score INTEGER NOT NULL,
    delta INTEGER DEFAULT 0,
    region VARCHAR(255) DEFAULT 'US',
    fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient querying
CREATE INDEX idx_trend_scores_query_id ON trend_scores(query_id);
CREATE INDEX idx_trend_scores_fetched_at ON trend_scores(fetched_at);
CREATE INDEX idx_trend_queries_seed ON trend_queries(seed_keyword);
