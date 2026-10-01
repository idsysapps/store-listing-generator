package com.trends.event;

import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.Base64;
import java.util.UUID;

@ApplicationScoped
public class CeleryTaskDispatcher {

    private static final Logger LOG = Logger.getLogger(CeleryTaskDispatcher.class);
    private static final String CELERY_QUEUE = "celery";

    @Inject
    ReactiveRedisDataSource redis;

    public Uni<String> dispatchTask(String taskName) {
        String taskId = UUID.randomUUID().toString();
        String correlationId = UUID.randomUUID().toString();

        String bodyJson = "[[], {}, {\"callbacks\": null, \"errbacks\": null, \"chain\": null, \"chord\": null}]";
        String bodyEncoded = Base64.getEncoder().encodeToString(bodyJson.getBytes());

        String message = """
            {
              "body": "%s",
              "content-encoding": "utf-8",
              "content-type": "application/json",
              "headers": {
                "lang": "py",
                "task": "%s",
                "id": "%s",
                "shadow": null,
                "eta": null,
                "expires": null,
                "group": null,
                "group_index": null,
                "retries": 0,
                "timelimit": [null, null],
                "root_id": "%s",
                "parent_id": null,
                "argsrepr": "()",
                "kwargsrepr": "{}",
                "origin": "trend-service"
              },
              "properties": {
                "correlation_id": "%s",
                "reply_to": "",
                "delivery_mode": 2,
                "delivery_info": {
                  "exchange": "",
                  "routing_key": "celery"
                },
                "priority": 0,
                "body_encoding": "base64",
                "delivery_tag": "%s"
              }
            }
            """.formatted(bodyEncoded, taskName, taskId, taskId, correlationId, taskId);

        LOG.infof("Dispatching Celery task %s (taskId=%s)", taskName, taskId);

        return redis.execute("LPUSH", CELERY_QUEUE, message)
                .replaceWith(taskId)
                .onFailure().invoke(e -> LOG.errorf(e, "Failed to dispatch Celery task %s", taskName));
    }

    public Uni<Void> dispatchGenerateSingleBriefImage(int briefId) {
        String taskName = "store_listing.orchestration.tasks.generate_single_brief_image_task";
        String taskId = UUID.randomUUID().toString();
        String correlationId = UUID.randomUUID().toString();

        String bodyJson = "[[" + briefId + "], {}, {\"callbacks\": null, \"errbacks\": null, \"chain\": null, \"chord\": null}]";
        String bodyEncoded = Base64.getEncoder().encodeToString(bodyJson.getBytes());

        String message = """
            {
              "body": "%s",
              "content-encoding": "utf-8",
              "content-type": "application/json",
              "headers": {
                "lang": "py",
                "task": "%s",
                "id": "%s",
                "shadow": null,
                "eta": null,
                "expires": null,
                "group": null,
                "group_index": null,
                "retries": 0,
                "timelimit": [null, null],
                "root_id": "%s",
                "parent_id": null,
                "argsrepr": "(%d,)",
                "kwargsrepr": "{}",
                "origin": "trend-service"
              },
              "properties": {
                "correlation_id": "%s",
                "reply_to": "",
                "delivery_mode": 2,
                "delivery_info": {
                  "exchange": "",
                  "routing_key": "celery"
                },
                "priority": 0,
                "body_encoding": "base64",
                "delivery_tag": "%s"
              }
            }
            """.formatted(bodyEncoded, taskName, taskId, taskId, briefId, correlationId, taskId);

        LOG.infof("Dispatching Celery task %s for briefId=%d (taskId=%s)", taskName, briefId, taskId);

        return redis.execute("LPUSH", CELERY_QUEUE, message)
                .replaceWithVoid()
                .onFailure().invoke(e -> LOG.errorf(e, "Failed to dispatch Celery task for briefId=%d", briefId));
    }
}
