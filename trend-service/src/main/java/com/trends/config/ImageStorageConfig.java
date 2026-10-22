package com.trends.config;

import jakarta.enterprise.context.ApplicationScoped;
import org.eclipse.microprofile.config.inject.ConfigProperty;

import java.util.Optional;

@ApplicationScoped
public class ImageStorageConfig {

    /**
     * The endpoint of the image storage.
     *
     * <p>Optional, because an empty value is not a value: SmallRye refuses
     * to start when a property is defined as the empty string, and the chart
     * ships the key empty when a cluster does not set it (issue #181).
     */
    @ConfigProperty(name = "design.s3.endpoint-url")
    Optional<String> endpointUrl;

    @ConfigProperty(name = "design.s3.bucket", defaultValue = "store-listing-designs")
    String bucket;

    public String buildUrl(String key) {
        if (key == null || key.isEmpty()) {
            return null;
        }
        String endpoint = endpointUrl.orElse("");
        if (!endpoint.isEmpty()) {
            String base = endpoint.endsWith("/") ? endpoint.substring(0, endpoint.length() - 1) : endpoint;
            return base + "/" + bucket + "/" + key;
        }
        return "https://" + bucket + ".s3.amazonaws.com/" + key;
    }
}
