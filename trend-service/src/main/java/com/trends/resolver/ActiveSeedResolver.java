package com.trends.resolver;

import com.trends.domain.ActiveSeed;
import com.trends.domain.SeedProductTag;
import com.trends.dto.ActiveSeedSummary;
import com.trends.repository.ActiveSeedRepository;
import io.quarkus.security.Authenticated;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.stream.Collectors;

@GraphQLApi
@ApplicationScoped
public class ActiveSeedResolver {

    private static final Logger LOG = Logger.getLogger(ActiveSeedResolver.class);

    @Inject
    ActiveSeedRepository activeSeedRepository;

    @Authenticated
    @Query("activeSeeds")
    public List<ActiveSeedSummary> getActiveSeeds(String tag, int limit) {
        LOG.debugf("Fetching active seeds: tag=%s, limit=%d", tag, limit);

        List<ActiveSeed> seeds;
        if (tag != null && !tag.isEmpty()) {
            seeds = activeSeedRepository.findActiveSeedsByTag(tag, limit);
        } else {
            seeds = activeSeedRepository.findActiveSeeds(limit);
        }

        return seeds.stream()
                .map(this::toSummary)
                .collect(Collectors.toList());
    }

    @Authenticated
    @Query("productTags")
    public List<String> getProductTags() {
        LOG.debug("Fetching distinct product tags");
        return activeSeedRepository.findDistinctTags();
    }

    private ActiveSeedSummary toSummary(ActiveSeed seed) {
        List<String> tags = seed.productTags != null
                ? seed.productTags.stream().map(t -> t.tag).collect(Collectors.toList())
                : List.of();
        return new ActiveSeedSummary(seed.id, seed.query, seed.promotionScore, tags);
    }
}
