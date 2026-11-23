package com.trends.subscription;

import com.trends.dto.ImageGenerationStatus;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ImageGenerationStatusesTest {

    @Test
    void parseReadsTheStatusTheWorkerPublishes() {
        String json = "{\"briefId\":12,\"phase\":\"sampling\",\"step\":5,\"totalSteps\":20,"
                + "\"elapsedSeconds\":50.0,\"estimatedTotalSeconds\":240.0,"
                + "\"remainingSeconds\":190.0}";

        ImageGenerationStatus status = ImageGenerationStatuses.parse(json);

        assertEquals(12, status.getBriefId());
        assertEquals("sampling", status.getPhase());
        assertEquals(5, status.getStep());
        assertEquals(20, status.getTotalSteps());
        assertEquals(50.0, status.getElapsedSeconds());
        assertEquals(240.0, status.getEstimatedTotalSeconds());
        assertEquals(190.0, status.getRemainingSeconds());
        assertTrue(ImageGenerationStatuses.forBrief(status, 12));
        assertFalse(ImageGenerationStatuses.forBrief(status, 4));
    }

    @Test
    void aBrokenPayloadIsSkipped() {
        assertNull(ImageGenerationStatuses.parse("not json"));
        assertNull(ImageGenerationStatuses.parse(""));
        assertFalse(ImageGenerationStatuses.forBrief(null, 12));
    }
}
