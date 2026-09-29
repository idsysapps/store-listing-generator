package com.trends.config;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;

@ApplicationScoped
public class ImageStorageConfig {

    @ConfigProperty(name = "design.s3.endpoint-url", defaultValue = "")
    String endpointUrl;

    @ConfigProperty(name = "design.s3.bucket", defaultValue = "store-listing-designs")
    String bucket;

    public String buildUrl(String key) {
        if (key == null || key.isEmpty()) {
            return null;
        }
        if (endpointUrl != null && !endpointUrl.isEmpty()) {
            String base = endpointUrl.endsWith("/") ? endpointUrl.substring(0, endpointUrl.length() - 1) : endpointUrl;
            return base + "/" + bucket + "/" + key;
        }
        return "https://" + bucket + ".s3.amazonaws.com/" + key;
    }
}
