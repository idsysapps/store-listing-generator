package com.trends.config;

import jakarta.inject.Inject;
import jakarta.ws.rs.*;
import jakarta.ws.rs.core.MediaType;
import jakarta.ws.rs.core.Response;

import com.trends.domain.TrendQuery;
import com.trends.dto.TrendUpdate;
import com.trends.repository.TrendQueryRepository;
import com.trends.event.TrendingEventPublisher;

import java.time.OffsetDateTime;

@Path("/api/events")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class EventResource {

    @Inject
    TrendQueryRepository trendQueryRepository;

    @Inject
    TrendingEventPublisher eventPublisher;

    @POST
    @Path("/trend-updated")
    public Response triggerTrendUpdated(TrendUpdateRequest request) {
        if (request.query() == null || request.query().isBlank()) {
            return Response.status(Response.Status.BAD_REQUEST)
                    .entity("{\"error\": \"query is required\"}")
                    .build();
        }

        TrendQuery trendQuery = trendQueryRepository.findByQuery(request.query());
        if (trendQuery == null) {
            return Response.status(Response.Status.NOT_FOUND)
                    .entity("{\"error\": \"query not found\"}")
                    .build();
        }

        int score = request.score() != null ? request.score() : 0;
        int delta = request.delta() != null ? request.delta() : 0;

        TrendUpdate update = new TrendUpdate(
                request.query(),
                delta,
                score,
                OffsetDateTime.now()
        );

        eventPublisher.publish(update);

        return Response.ok()
                .entity("{\"status\": \"published\"}")
                .build();
    }

    public record TrendUpdateRequest(String query, Integer score, Integer delta) {}
}
