package com.trends.resolver;

import com.trends.domain.DesignBrief;
import com.trends.domain.DesignBriefSource;
import com.trends.dto.DesignBriefSummary;
import com.trends.repository.DesignBriefRepository;
import io.quarkus.security.Authenticated;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

@GraphQLApi
@ApplicationScoped
public class DesignBriefResolver {

    private static final Logger LOG = Logger.getLogger(DesignBriefResolver.class);

    @Inject
    DesignBriefRepository designBriefRepository;

    @Authenticated
    @Query("designBriefs")
    public List<DesignBriefSummary> getDesignBriefs(String productType, String batchId, int limit) {
        LOG.debugf("Fetching design briefs: productType=%s, batchId=%s, limit=%d",
                productType, batchId, limit);

        List<DesignBrief> briefs;
        if (batchId != null && !batchId.isEmpty()) {
            briefs = designBriefRepository.findByBatchId(batchId);
        } else if (productType != null && !productType.isEmpty()) {
            briefs = designBriefRepository.findByProductType(productType, limit);
        } else {
            briefs = designBriefRepository.findRecent(limit);
        }

        return briefs.stream()
                .map(this::toSummary)
                .collect(Collectors.toList());
    }

    private DesignBriefSummary toSummary(DesignBrief brief) {
        List<String> products = brief.specificProducts != null
                ? Arrays.asList(brief.specificProducts)
                : List.of();

        List<String> sourceSeeds = brief.sources != null
                ? brief.sources.stream()
                        .map(s -> s.activeSeed.query)
                        .collect(Collectors.toList())
                : List.of();

        return new DesignBriefSummary(
                brief.id, brief.concept, brief.productType,
                products, brief.audience, brief.visualStyle,
                brief.confidence, brief.reasoning, brief.llmModel,
                brief.batchId, brief.createdAt, sourceSeeds);
    }
}
