package com.trends.subscription;

import com.trends.dto.ImageGenerationStatus;
import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.quarkus.security.identity.SecurityIdentity;
import io.smallrye.graphql.api.Subscription;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.GraphQLException;

@GraphQLApi
@ApplicationScoped
public class ImageGenerationSubscription {

    @Inject
    ReactiveRedisDataSource redis;

    @Inject
    SecurityIdentity securityIdentity;

    @Subscription
    public Multi<ImageGenerationStatus> imageGeneration(int briefId) throws GraphQLException {
        if (securityIdentity == null || securityIdentity.isAnonymous()) {
            throw new GraphQLException("Authentication required");
        }
        return redis.pubsub(String.class)
                .subscribe(ImageGenerationStatuses.CHANNEL)
                .map(ImageGenerationStatuses::parse)
                .select().where(status -> ImageGenerationStatuses.forBrief(status, briefId));
    }
}
