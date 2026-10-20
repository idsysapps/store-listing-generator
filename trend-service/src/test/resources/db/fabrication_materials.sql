-- Mirrors the fabrication_materials table created by the Helm chart
-- (helm/store-listing/templates/configmap.yaml). The integration tests run
-- against a real PostgreSQL container, so the schema has to be the production
-- one - H2 cannot create jsonb or text[] columns.
CREATE TABLE IF NOT EXISTS fabrication_materials (
    id SERIAL PRIMARY KEY,
    external_id VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    brand VARCHAR(100),
    substrate_type VARCHAR(50) NOT NULL,
    color VARCHAR(100),
    sizes TEXT[],
    material_composition VARCHAR(255),
    product_type VARCHAR(50) NOT NULL,
    sku VARCHAR(100),
    unit_cost DECIMAL(10,2),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    attributes JSONB,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fab_materials_product_type ON fabrication_materials(product_type);
CREATE INDEX IF NOT EXISTS idx_fab_materials_active ON fabrication_materials(active);
CREATE INDEX IF NOT EXISTS idx_fab_materials_external_id ON fabrication_materials(external_id);
