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
import java.util.ArrayList;
import java.util.List;

@ApplicationScoped
public class MaterialStreamConsumer {

    private static final Logger LOG = Logger.getLogger(MaterialStreamConsumer.class);
    private static final String STREAM_KEY = "materials:events";
    private static final String DEAD_STREAM_KEY = "materials:events:dead";
    private static final String GROUP_NAME = "trend-service";
    private static final String CONSUMER_NAME = "consumer-1";
    private static final long POLL_INTERVAL_MS = 5000;

    /**
     * How long a message has to sit unprocessed before it is claimed again.
     *
     * <p>A failed message was never acknowledged, so it stayed in the pending
     * list of the consumer group forever: the group only reads new messages
     * with {@code >}, and the pending entries were never picked up again
     * (issue #182).
     */
    private static final long MIN_IDLE_MS = 60_000;

    /** How often a message is tried before it is put aside. */
    private static final int MAX_DELIVERIES = 5;

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
        claimPendingMessages();
        redis.execute("XREADGROUP", "GROUP", GROUP_NAME, CONSUMER_NAME,
                        "COUNT", "10", "STREAMS", STREAM_KEY, ">")
                .subscribe().with(
                        this::processStreamResponse,
                        e -> LOG.debugf("Stream poll error (may be empty): %s", e.getMessage()));
        logPendingBacklog();
    }

    /**
     * Claims the messages that were delivered but never acknowledged.
     *
     * <p>A message that failed to process stays pending, and a pending message
     * is never delivered again by a plain read. The claim takes the entries
     * that have been idle long enough, so a transient database error does not
     * cost the material forever (issue #182).
     */
    void claimPendingMessages() {
        redis.execute("XAUTOCLAIM", STREAM_KEY, GROUP_NAME, CONSUMER_NAME,
                        Long.toString(MIN_IDLE_MS), "0-0", "COUNT", "10")
                .subscribe().with(
                        this::processClaimedResponse,
                        e -> LOG.debugf("Could not claim the pending material events: %s", e.getMessage()));
    }

    void processClaimedResponse(Response response) {
        if (response == null || response.size() < 2) {
            return;
        }
        Response entries = response.get(1);
        if (entries == null) {
            return;
        }

        for (int i = 0; i < entries.size(); i++) {
            Response entry = entries.get(i);
            String messageId = entry.get(0).toString();
            long deliveries = deliveryCount(entry);

            if (deliveries >= MAX_DELIVERIES) {
                moveToDeadLetter(messageId, entry, deliveries);
                continue;
            }

            LOG.infof("Retrying the material event %s (%d deliveries so far)", messageId, deliveries);
            processEntry(messageId, entry.get(1));
        }
    }

    /** How often the message was delivered to this consumer group. */
    long deliveryCount(Response entry) {
        if (entry.size() < 3 || entry.get(2) == null) {
            return 1;
        }
        try {
            return entry.get(2).toLong();
        } catch (Exception e) { // noqa: BLE001
            return 1;
        }
    }

    /**
     * Puts a message that keeps failing aside.
     *
     * <p>It is moved to a dead letter stream with its fields, so the event is
     * not lost and can be replayed by hand, and it is acknowledged so the
     * pending list does not grow without bound (issue #182).
     */
    void moveToDeadLetter(String messageId, Response entry, long deliveries) {
        Response fields = entry.get(1);
        List<String> args = new ArrayList<>();
        args.add(DEAD_STREAM_KEY);
        args.add("*");
        if (fields != null) {
            for (int i = 0; i < fields.size(); i++) {
                args.add(fields.get(i).toString());
            }
        }
        args.add("failed_deliveries");
        args.add(Long.toString(deliveries));
        args.add("failed_message_id");
        args.add(messageId);

        redis.execute("XADD", args.toArray(new String[0])).subscribe().with(
                r -> {
                    LOG.errorf("The material event %s failed %d times, moved to %s",
                            messageId, deliveries, DEAD_STREAM_KEY);
                    ackMessage(messageId);
                },
                e -> LOG.errorf("Could not move the material event %s aside: %s",
                        messageId, e.getMessage()));
    }

    /** Logs how many events are waiting to be processed. */
    void logPendingBacklog() {
        redis.execute("XPENDING", STREAM_KEY, GROUP_NAME).subscribe().with(
                response -> {
                    if (response == null || response.size() == 0) {
                        return;
                    }
                    long pending = response.get(0).toLong();
                    if (pending > 0) {
                        LOG.warnf("%d material events are pending and have not been processed", pending);
                    }
                },
                e -> LOG.debugf("Could not read the pending material events: %s", e.getMessage()));
    }

    void processStreamResponse(Response response) {
        if (response == null) {
            return;
        }

        Response entries = response.get(STREAM_KEY);
        if (entries == null) {
            return;
        }

        for (int e = 0; e < entries.size(); e++) {
            Response entry = entries.get(e);
            String messageId = entry.get(0).toString();
            Response fields = entry.get(1);

            vertx.executeBlocking(() -> {
                try {
                    processMessage(messageId, fields);
                } catch (Exception ex) {
                    LOG.errorf(ex, "Failed to process message %s", messageId);
                }
                return null;
            }).subscribe().with(
                    v -> {},
                    ex -> LOG.errorf(ex, "Failed to execute blocking for message %s", messageId));
        }
    }

    /** Processes one entry of the stream and acknowledges it when it worked. */
    void processEntry(String messageId, Response fields) {
        vertx.executeBlocking(() -> {
            try {
                processMessage(messageId, fields);
            } catch (Exception ex) {
                LOG.errorf(ex, "Failed to process message %s", messageId);
            }
            return null;
        }).subscribe().with(
                v -> {},
                ex -> LOG.errorf(ex, "Failed to execute blocking for message %s", messageId));
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
            if (sizesJson != null) existing.sizes = parseStringArray(sizesJson);
            if (materialComposition != null) existing.materialComposition = materialComposition;
            if (productType != null) existing.productType = parseStringArray(productType);
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
            mat.sizes = parseStringArray(sizesJson);
            mat.materialComposition = materialComposition;
            mat.productType = parseStringArray(productType);
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
        Response value = fields.get(fieldName);
        return value != null ? value.toString() : null;
    }

    private void ackMessage(String messageId) {
        redis.execute("XACK", STREAM_KEY, GROUP_NAME, messageId)
                .subscribe().with(
                        r -> {},
                        e -> LOG.warnf("Failed to ACK message %s: %s", messageId, e.getMessage()));
    }

    /**
     * The values of a JSON array field of the event.
     *
     * <p>Cost-book publishes the sizes and the product types of a material as
     * JSON arrays. A field that carries a single value instead of an array is
     * read as a one-element array, so the column always holds the list the
     * producer meant (issue #186).
     */
    String[] parseStringArray(String json) {
        if (json == null || json.isBlank()) {
            return null;
        }
        String trimmed = json.trim();
        if (!trimmed.startsWith("[")) {
            String single = trimmed.replaceAll("^\"|\"$", "").trim();
            return single.isEmpty() ? null : new String[] { single };
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
