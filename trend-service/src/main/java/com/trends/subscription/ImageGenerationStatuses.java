package com.trends.subscription;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.dto.ImageGenerationStatus;
import io.smallrye.mutiny.Multi;

public final class ImageGenerationStatuses {

    public static final String CHANNEL = "image-generation";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ImageGenerationStatuses() {}

    public static String statusKey(int briefId) {
        return CHANNEL + ":status:" + briefId;
    }

    /**
     * The status stored before this subscription connected, followed by the live channel.
     * A snapshot for a different brief is ignored.
     */
    public static Multi<ImageGenerationStatus> startingWith(
            ImageGenerationStatus snapshot, int briefId, Multi<ImageGenerationStatus> live) {
        if (!forBrief(snapshot, briefId)) {
            return live;
        }
        return Multi.createFrom().item(snapshot).onCompletion().switchTo(live);
    }

    public static ImageGenerationStatus parse(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        try {
            return MAPPER.readValue(json, ImageGenerationStatus.class);
        } catch (Exception ignored) {
            return null;
        }
    }

    public static boolean forBrief(ImageGenerationStatus status, int briefId) {
        return status != null && status.getBriefId() == briefId;
    }
}
