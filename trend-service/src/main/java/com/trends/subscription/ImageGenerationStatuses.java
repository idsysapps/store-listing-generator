package com.trends.subscription;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trends.dto.ImageGenerationStatus;

public final class ImageGenerationStatuses {

    public static final String CHANNEL = "image-generation";

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private ImageGenerationStatuses() {}

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
