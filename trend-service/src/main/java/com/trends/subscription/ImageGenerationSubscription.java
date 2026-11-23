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
        Multi<ImageGenerationStatus> live = redis.pubsub(String.class)
                .subscribe(ImageGenerationStatuses.CHANNEL)
                .map(ImageGenerationStatuses::parse)
                .select().where(status -> ImageGenerationStatuses.forBrief(status, briefId));
        return redis.value(String.class)
                .get(ImageGenerationStatuses.statusKey(briefId))
                .onFailure().recoverWithItem((String) null)
                .onItem().transformToMulti(json -> ImageGenerationStatuses.startingWith(
                        ImageGenerationStatuses.parse(json), briefId, live));
    }
}
