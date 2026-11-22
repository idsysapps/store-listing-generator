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

import java.util.Base64;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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
    void testDispatchRegenerateListingDescription_SendsTheBriefAndTheGuidance() {
        celeryTaskDispatcher
                .dispatchRegenerateListingDescription(12, "Mention Sunday mornings")
                .await().indefinitely();

        ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
        verify(redis).execute(eq("LPUSH"), eq("celery"), messageCaptor.capture());

        String message = messageCaptor.getValue();
        assertTrue(message.contains(
                "store_listing.orchestration.tasks.regenerate_listing_description_task"));
        Matcher matcher = Pattern.compile("\"body\": \"([^\"]+)\"").matcher(message);
        assertTrue(matcher.find());
        String body = new String(Base64.getDecoder().decode(matcher.group(1)));
        assertTrue(body.startsWith("[[12, \"Mention Sunday mornings\"]"));
    }

    @Test
    void testDispatchRegenerateListingDescription_EscapesQuotesInGuidance() {
        celeryTaskDispatcher
                .dispatchRegenerateListingDescription(12, "say \"hello\"")
                .await().indefinitely();

        ArgumentCaptor<String> messageCaptor = ArgumentCaptor.forClass(String.class);
        verify(redis).execute(eq("LPUSH"), eq("celery"), messageCaptor.capture());
        Matcher matcher = Pattern.compile("\"body\": \"([^\"]+)\"").matcher(messageCaptor.getValue());
        assertTrue(matcher.find());
        String body = new String(Base64.getDecoder().decode(matcher.group(1)));
        assertTrue(body.contains("say \\\"hello\\\""));
    }

    @Test
    void testDispatchTask_ReturnsUniqueTaskIds() {
        String taskName = "store_listing.orchestration.tasks.curate_seeds_task";

        String id1 = celeryTaskDispatcher.dispatchTask(taskName).await().indefinitely();
        String id2 = celeryTaskDispatcher.dispatchTask(taskName).await().indefinitely();

        assertNotEquals(id1, id2);
    }
}
