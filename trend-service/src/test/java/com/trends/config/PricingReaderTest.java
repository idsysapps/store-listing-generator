package com.trends.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.dto.BriefPricing;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class PricingReaderTest {

    private PricingReader reader;

    @BeforeEach
    void setUp() {
        reader = new PricingReader();
        reader.objectMapper = new ObjectMapper();
    }

    @Test
    void testRead_WithoutASnapshotIsNull() {
        assertNull(reader.read(null));
        assertNull(reader.read(""));
    }

    @Test
    void testRead_ReadsThePositioningAndTheQuote() {
        String json = "{\"requested_positioning\":\"premium\","
                + "\"effective_positioning\":\"standard\","
                + "\"positioning_reason\":\"The premium price did not fit.\","
                + "\"confidence\":80,\"target_margin\":0.4,\"status\":\"downgraded\","
                + "\"currency\":\"USD\",\"policy_version\":1,\"print_rate_per_sq_in\":0.005,"
                + "\"quotes\":[{\"external_id\":\"bc3001-white\","
                + "\"requested_positioning\":\"premium\",\"positioning\":\"standard\","
                + "\"status\":\"downgraded\",\"recommended_price\":19.99,\"currency\":\"USD\","
                + "\"size_prices\":[{\"size\":\"S\",\"price\":19.99},"
                + "{\"size\":\"2XL\",\"price\":21.99}],"
                + "\"print_size\":{\"width_in\":12.0,\"height_in\":12.0,"
                + "\"placement\":\"full_front\",\"target_dpi\":300,"
                + "\"required_pixels\":{\"width\":3600,\"height\":3600},"
                + "\"source_pixels\":{\"width\":1024,\"height\":1024},"
                + "\"upscale_factor\":3.5156,\"estimated\":false},"
                + "\"price_breakdown\":{\"material_cost\":4.5,\"print_cost\":0.72,"
                + "\"print_rate_per_sq_in\":0.005,\"marketplace_fee\":3.0,"
                + "\"target_margin\":0.4,\"actual_margin\":0.42}}]}";

        BriefPricing pricing = reader.read(json);

        assertEquals("premium", pricing.getRequestedPositioning());
        assertEquals("standard", pricing.getEffectivePositioning());
        assertEquals("downgraded", pricing.getStatus());
        assertEquals(80, pricing.getConfidence());
        assertEquals(1, pricing.getQuotes().size());
        assertEquals("bc3001-white", pricing.getQuotes().get(0).getExternalId());
        assertEquals(19.99, pricing.getQuotes().get(0).getRecommendedPrice());
        assertEquals("S", pricing.getQuotes().get(0).getSizePrices().get(0).getSize());
        assertEquals(21.99, pricing.getQuotes().get(0).getSizePrices().get(1).getPrice());
        assertEquals(3600, pricing.getQuotes().get(0).getPrintSize()
                .getRequiredPixels().getWidth());
        assertEquals(0.005, pricing.getQuotes().get(0).getPriceBreakdown()
                .getPrintRatePerSqIn());
    }

    @Test
    void testRead_AnUnreadableSnapshotIsNull() {
        assertNull(reader.read("not json"));
    }
}
