package com.trends.event;

import com.trends.domain.FabricationMaterial;
import com.trends.repository.FabricationMaterialRepository;
import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.quarkus.runtime.StartupEvent;
import io.vertx.mutiny.core.Vertx;
import io.vertx.mutiny.redis.client.Response;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.jboss.logging.Logger;

import java.math.BigDecimal;
import java.time.OffsetDateTime;

@ApplicationScoped
public class MaterialStreamConsumer {

    private static final Logger LOG = Logger.getLogger(MaterialStreamConsumer.class);
    private static final String STREAM_KEY = "materials:events";
    private static final String GROUP_NAME = "trend-service";
    private static final String CONSUMER_NAME = "consumer-1";
    private static final long POLL_INTERVAL_MS = 5000;

    @Inject
    ReactiveRedisDataSource redis;

    @Inject
    FabricationMaterialRepository materialRepository;

    @Inject
    Vertx vertx;

    void onStart(@Observes StartupEvent ev) {
        createConsumerGroup();
        vertx.setPeriodic(POLL_INTERVAL_MS, id -> pollStream());
    }

    void createConsumerGroup() {
        redis.execute("XGROUP", "CREATE", STREAM_KEY, GROUP_NAME, "$", "MKSTREAM")
                .subscribe().with(
                        r -> LOG.info("Created consumer group: " + GROUP_NAME),
                        e -> {
                            if (e.getMessage() != null && e.getMessage().contains("BUSYGROUP")) {
                                LOG.debug("Consumer group already exists: " + GROUP_NAME);
                            } else {
                                LOG.warn("Failed to create consumer group: " + e.getMessage());
                            }
                        });
    }

    void pollStream() {
        redis.execute("XREADGROUP", "GROUP", GROUP_NAME, CONSUMER_NAME,
                        "COUNT", "10", "STREAMS", STREAM_KEY, ">")
                .subscribe().with(
                        this::processStreamResponse,
                        e -> LOG.debugf("Stream poll error (may be empty): %s", e.getMessage()));
    }

    void processStreamResponse(Response response) {
        if (response == null) {
            return;
        }

        for (int s = 0; s < response.size(); s++) {
            Response streamData = response.get(s);
            Response entries = streamData.get(1);

            for (int e = 0; e < entries.size(); e++) {
                Response entry = entries.get(e);
                String messageId = entry.get(0).toString();
                Response fields = entry.get(1);

                try {
                    processMessage(messageId, fields);
                } catch (Exception ex) {
                    LOG.errorf(ex, "Failed to process message %s", messageId);
                }
            }
        }
    }

    void processMessage(String messageId, Response fields) {
        String action = getField(fields, "action");
        String externalId = getField(fields, "external_id");

        if (action == null || externalId == null) {
            LOG.warnf("Skipping message %s: missing action or external_id", messageId);
            ackMessage(messageId);
            return;
        }

        switch (action) {
            case "upsert" -> {
                processUpsertEvent(
                        externalId,
                        getField(fields, "name"),
                        getField(fields, "brand"),
                        getField(fields, "substrate_type"),
                        getField(fields, "color"),
                        getField(fields, "sizes"),
                        getField(fields, "material_composition"),
                        getField(fields, "product_type"),
                        getField(fields, "sku"),
                        getField(fields, "unit_cost"),
                        getField(fields, "active"),
                        getField(fields, "attributes"));
                LOG.infof("Upserted material: %s", externalId);
            }
            case "deactivate" -> {
                processDeactivateEvent(externalId);
                LOG.infof("Deactivated material: %s", externalId);
            }
            default -> LOG.warnf("Unknown action '%s' for message %s", action, messageId);
        }

        ackMessage(messageId);
    }

    @Transactional
    public void processUpsertEvent(
            String externalId, String name, String brand,
            String substrateType, String color, String sizesJson,
            String materialComposition, String productType,
            String sku, String unitCostStr, String activeStr,
            String attributesJson) {

        FabricationMaterial existing = materialRepository.findByExternalId(externalId);

        if (existing != null) {
            if (name != null) existing.name = name;
            if (brand != null) existing.brand = brand;
            if (substrateType != null) existing.substrateType = substrateType;
            if (color != null) existing.color = color;
            if (sizesJson != null) existing.sizes = parseSizes(sizesJson);
            if (materialComposition != null) existing.materialComposition = materialComposition;
            if (productType != null) existing.productType = productType;
            if (sku != null) existing.sku = sku;
            if (unitCostStr != null) existing.unitCost = new BigDecimal(unitCostStr);
            if (activeStr != null) existing.active = Boolean.parseBoolean(activeStr);
            if (attributesJson != null) existing.attributes = attributesJson;
            existing.updatedAt = OffsetDateTime.now();
        } else {
            FabricationMaterial mat = new FabricationMaterial();
            mat.externalId = externalId;
            mat.name = name;
            mat.brand = brand;
            mat.substrateType = substrateType;
            mat.color = color;
            mat.sizes = parseSizes(sizesJson);
            mat.materialComposition = materialComposition;
            mat.productType = productType;
            mat.sku = sku;
            mat.unitCost = unitCostStr != null ? new BigDecimal(unitCostStr) : null;
            mat.active = activeStr != null ? Boolean.parseBoolean(activeStr) : true;
            mat.attributes = attributesJson;
            mat.createdAt = OffsetDateTime.now();
            mat.updatedAt = OffsetDateTime.now();
            materialRepository.persist(mat);
        }
    }

    @Transactional
    public void processDeactivateEvent(String externalId) {
        materialRepository.deactivateByExternalId(externalId);
    }

    private String getField(Response fields, String fieldName) {
        for (int i = 0; i < fields.size() - 1; i += 2) {
            if (fieldName.equals(fields.get(i).toString())) {
                return fields.get(i + 1).toString();
            }
        }
        return null;
    }

    private void ackMessage(String messageId) {
        redis.execute("XACK", STREAM_KEY, GROUP_NAME, messageId)
                .subscribe().with(
                        r -> {},
                        e -> LOG.warnf("Failed to ACK message %s: %s", messageId, e.getMessage()));
    }

    String[] parseSizes(String sizesJson) {
        if (sizesJson == null || sizesJson.isBlank()) {
            return null;
        }
        String trimmed = sizesJson.trim();
        if (!trimmed.startsWith("[")) {
            return null;
        }
        trimmed = trimmed.substring(1, trimmed.length() - 1);
        if (trimmed.isBlank()) {
            return new String[0];
        }
        String[] parts = trimmed.split(",");
        String[] result = new String[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = parts[i].trim().replaceAll("^\"|\"$", "");
        }
        return result;
    }
}
