package com.trends.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.dto.BriefPricing;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

/**
 * Reads the pricing snapshot stored with a design brief.
 *
 * <p>The snapshot is written by the Python pipeline after the pricing pass. A
 * brief from before the column existed has none, and an unreadable snapshot
 * must not hide the rest of the brief (issue #295).
 */
@ApplicationScoped
public class PricingReader {

    private static final Logger LOG = Logger.getLogger(PricingReader.class);

    @Inject
    ObjectMapper objectMapper;

    /** The pricing snapshot, or null when the brief has none. */
    public BriefPricing read(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return objectMapper.readValue(json, BriefPricing.class);
        } catch (JsonProcessingException e) {
            LOG.warnf("Could not read the design brief pricing: %s", e.getMessage());
            return null;
        }
    }
}
