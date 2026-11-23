package com.trends.subscription;

import com.trends.dto.ImageGenerationStatus;
import io.smallrye.mutiny.Multi;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;

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

    @Test
    void theStatusKeyMatchesTheKeyTheWorkerWrites() {
        assertEquals("image-generation:status:12", ImageGenerationStatuses.statusKey(12));
    }

    @Test
    void theSnapshotIsSentBeforeLiveEvents() {
        ImageGenerationStatus snapshot = status("writing");
        ImageGenerationStatus live = status("researching");

        List<ImageGenerationStatus> seen = ImageGenerationStatuses
                .startingWith(snapshot, 12, Multi.createFrom().item(live))
                .collect().asList()
                .await().atMost(Duration.ofSeconds(2));

        assertEquals(List.of("writing", "researching"),
                seen.stream().map(ImageGenerationStatus::getPhase).toList());
    }

    @Test
    void aSnapshotForAnotherBriefIsSkipped() {
        ImageGenerationStatus other = status("writing");
        other.setBriefId(4);
        ImageGenerationStatus live = status("sampling");

        List<ImageGenerationStatus> seen = ImageGenerationStatuses
                .startingWith(other, 12, Multi.createFrom().item(live))
                .collect().asList()
                .await().atMost(Duration.ofSeconds(2));

        assertEquals(List.of("sampling"),
                seen.stream().map(ImageGenerationStatus::getPhase).toList());
    }

    private static ImageGenerationStatus status(String phase) {
        ImageGenerationStatus status = new ImageGenerationStatus();
        status.setBriefId(12);
        status.setPhase(phase);
        return status;
    }
}
