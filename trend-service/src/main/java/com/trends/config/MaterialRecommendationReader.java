package com.trends.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.dto.MaterialRecommendation;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.List;

/**
 * Reads the substrates a design brief was recommended.
 *
 * <p>The recommendations are stored with the brief as a JSON document, and
 * they were dropped on the way to the client until now: a brief said which
 * substrate it should be printed on, and the client never heard (issue #215).
 */
@ApplicationScoped
public class MaterialRecommendationReader {

    private static final Logger LOG = Logger.getLogger(MaterialRecommendationReader.class);

    @Inject
    ObjectMapper objectMapper;

    /** The recommended substrates, or an empty list when there are none. */
    public List<MaterialRecommendation> read(String json) {
        if (json == null || json.isBlank()) {
            return List.of();
        }
        try {
            MaterialRecommendation[] recommendations =
                    objectMapper.readValue(json, MaterialRecommendation[].class);
            if (recommendations == null) {
                return List.of();
            }
            return List.of(recommendations);
        } catch (JsonProcessingException e) {
            LOG.warnf("Could not read the recommended materials: %s", e.getMessage());
            return List.of();
        }
    }
}
