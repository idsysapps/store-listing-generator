package com.trends.event;

import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.smallrye.mutiny.Uni;
import io.vertx.mutiny.redis.client.Response;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class CeleryTaskDispatcherTest {

    @Mock
    ReactiveRedisDataSource redis;

    @InjectMocks
    CeleryTaskDispatcher celeryTaskDispatcher;

    @SuppressWarnings("unchecked")
    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        when(redis.execute(anyString(), anyString(), anyString()))
                .thenReturn(Uni.createFrom().item(Response.newInstance(null)));
    }

    @Test
    void testDispatchTask_PushesToCeleryQueue() {
        String taskName = "store_listing.orchestration.tasks.curate_seeds_task";

        Uni<String> result = celeryTaskDispatcher.dispatchTask(taskName);
        String taskId = result.await().indefinitely();

        assertNotNull(taskId);
        assertFalse(taskId.isEmpty());

        ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
        verify(redis).execute(eq("LPUSH"), eq("celery"), messageCaptor.capture());

        String message = messageCaptor.getValue();
        assertTrue(message.contains("\"task\": \"" + taskName + "\""));
        assertTrue(message.contains("\"origin\": \"trend-service\""));
    }

    @Test
    void testDispatchTask_BodyContainsEmptyArgs() {
        String taskName = "store_listing.orchestration.tasks.generate_design_briefs_task";

        celeryTaskDispatcher.dispatchTask(taskName).await().indefinitely();

        ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
        verify(redis).execute(eq("LPUSH"), eq("celery"), messageCaptor.capture());

        String message = messageCaptor.getValue();
        assertTrue(message.contains("\"body\""));
        assertTrue(message.contains("\"body_encoding\": \"base64\""));
        assertTrue(message.contains("\"argsrepr\": \"()\""));
    }

    @Test
    void testDispatchTask_ReturnsUniqueTaskIds() {
        String taskName = "store_listing.orchestration.tasks.curate_seeds_task";

        String id1 = celeryTaskDispatcher.dispatchTask(taskName).await().indefinitely();
        String id2 = celeryTaskDispatcher.dispatchTask(taskName).await().indefinitely();

        assertNotEquals(id1, id2);
    }
}
