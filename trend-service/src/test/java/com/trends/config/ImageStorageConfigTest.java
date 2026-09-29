package com.trends.config;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.*;

class ImageStorageConfigTest {

    private ImageStorageConfig configWith(String endpointUrl, String bucket) throws Exception {
        ImageStorageConfig config = new ImageStorageConfig();
        Field endpointField = ImageStorageConfig.class.getDeclaredField("endpointUrl");
        endpointField.setAccessible(true);
        endpointField.set(config, endpointUrl);
        Field bucketField = ImageStorageConfig.class.getDeclaredField("bucket");
        bucketField.setAccessible(true);
        bucketField.set(config, bucket);
        return config;
    }

    @Test
    void testBuildUrl_WithEndpoint() throws Exception {
        ImageStorageConfig config = configWith("https://minio.local:9000", "designs");
        String url = config.buildUrl("designs/2026/09/dtf_apparel/42_raw.png");
        assertEquals("https://minio.local:9000/designs/designs/2026/09/dtf_apparel/42_raw.png", url);
    }

    @Test
    void testBuildUrl_WithEndpointTrailingSlash() throws Exception {
        ImageStorageConfig config = configWith("https://minio.local:9000/", "designs");
        String url = config.buildUrl("designs/2026/09/dtf_apparel/42_raw.png");
        assertEquals("https://minio.local:9000/designs/designs/2026/09/dtf_apparel/42_raw.png", url);
    }

    @Test
    void testBuildUrl_WithoutEndpoint_UsesAwsS3() throws Exception {
        ImageStorageConfig config = configWith("", "store-listing-designs");
        String url = config.buildUrl("designs/2026/09/dtf_apparel/42_raw.png");
        assertEquals("https://store-listing-designs.s3.amazonaws.com/designs/2026/09/dtf_apparel/42_raw.png", url);
    }

    @Test
    void testBuildUrl_NullKey_ReturnsNull() throws Exception {
        ImageStorageConfig config = configWith("https://minio.local:9000", "designs");
        assertNull(config.buildUrl(null));
    }

    @Test
    void testBuildUrl_EmptyKey_ReturnsNull() throws Exception {
        ImageStorageConfig config = configWith("https://minio.local:9000", "designs");
        assertNull(config.buildUrl(""));
    }
}
