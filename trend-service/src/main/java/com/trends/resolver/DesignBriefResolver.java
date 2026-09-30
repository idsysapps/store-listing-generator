package com.trends.resolver;

import com.trends.config.ImageStorageConfig;
import com.trends.domain.DesignBrief;
import com.trends.domain.DesignBriefSource;
import com.trends.dto.DesignBriefSummary;
import com.trends.event.CeleryTaskDispatcher;
import com.trends.repository.DesignBriefRepository;
import io.quarkus.security.Authenticated;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.Mutation;
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

    @Inject
    ImageStorageConfig imageStorageConfig;

    @Inject
    CeleryTaskDispatcher celeryTaskDispatcher;

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

    @Authenticated
    @Mutation("requestRegeneration")
    @Transactional
    public DesignBriefSummary requestRegeneration(int briefId, String feedback) {
        DesignBrief brief = designBriefRepository.findById((long) briefId);
        if (brief == null) {
            return null;
        }

        brief.imageKeyRaw = null;
        brief.imageKeyTransparent = null;
        brief.regenerationFeedback = feedback;
        designBriefRepository.persist(brief);

        celeryTaskDispatcher.dispatchGenerateSingleBriefImage(briefId)
                .subscribe().with(
                        v -> LOG.infof("Dispatched image generation for briefId=%d", briefId),
                        e -> LOG.errorf(e, "Failed to dispatch image generation for briefId=%d", briefId)
                );

        return toSummary(brief);
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

        DesignBriefSummary summary = new DesignBriefSummary(
                brief.id, brief.concept, brief.productType,
                products, brief.audience, brief.visualStyle,
                brief.confidence, brief.reasoning, brief.llmModel,
                brief.batchId, brief.createdAt, sourceSeeds,
                brief.imageKeyRaw, brief.imageKeyTransparent);
        summary.setImageUrl(imageStorageConfig.buildUrl(brief.imageKeyRaw));
        summary.setImageTransparentUrl(imageStorageConfig.buildUrl(brief.imageKeyTransparent));
        return summary;
    }
}
