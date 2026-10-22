package com.trends.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.dto.MaterialRecommendation;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.when;

class MaterialRecommendationReaderTest {

    @Mock
    ObjectMapper objectMapper;

    @InjectMocks
    MaterialRecommendationReader reader;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testRead_WithoutAJsonDocumentIsAnEmptyList() {
        assertTrue(reader.read(null).isEmpty());
        assertTrue(reader.read("").isEmpty());
    }

    @Test
    void testRead_ReadsTheRecommendedSubstrates() throws Exception {
        String json = "[{\"external_id\":\"bc3001-white\",\"name\":\"Bella Canvas 3001\","
                + "\"color\":\"White\",\"substrate_type\":\"t-shirt\",\"unit_cost\":4.5}]";
        when(objectMapper.readValue(json, MaterialRecommendation[].class)).thenReturn(
                new MaterialRecommendation[] {
                    new MaterialRecommendation("bc3001-white", "Bella Canvas 3001", "White",
                            "t-shirt", 4.5)
                });

        List<MaterialRecommendation> result = reader.read(json);

        assertEquals(1, result.size());
        assertEquals("bc3001-white", result.get(0).getExternalId());
        assertEquals("Bella Canvas 3001", result.get(0).getName());
        assertEquals("t-shirt", result.get(0).getSubstrateType());
        assertEquals(4.5, result.get(0).getUnitCost());
    }

    @Test
    void testRead_AnUnreadableDocumentIsAnEmptyList() throws Exception {
        when(objectMapper.readValue("not json", MaterialRecommendation[].class))
                .thenThrow(new com.fasterxml.jackson.core.JsonParseException(
                        "not json", com.fasterxml.jackson.core.JsonLocation.NA));

        assertTrue(reader.read("not json").isEmpty());
    }
}
