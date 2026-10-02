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
    public List<DesignBriefSummary> getDesignBriefs(
            String productType, String batchId,
            String startDate, String endDate,
            Integer minConfidence, Integer maxConfidence,
            List<String> sourceSeeds, List<String> specificProducts,
            String audienceContains, int limit) {

        String pt = nullIfEmpty(productType);
        String bi = nullIfEmpty(batchId);
        String sd = nullIfEmpty(startDate);
        String ed = nullIfEmpty(endDate);
        String ac = nullIfEmpty(audienceContains);
        List<String> ss = nullIfEmptyList(sourceSeeds);
        List<String> sp = nullIfEmptyList(specificProducts);

        LOG.debugf("Fetching design briefs: productType=%s, batchId=%s, limit=%d", pt, bi, limit);

        boolean hasAdvancedFilters = sd != null || ed != null
                || minConfidence != null || maxConfidence != null
                || (ss != null && !ss.isEmpty())
                || (sp != null && !sp.isEmpty())
                || ac != null;

        List<DesignBrief> briefs;
        if (hasAdvancedFilters || (pt != null && bi != null)) {
            briefs = designBriefRepository.findWithFilters(
                    pt, bi, sd, ed, minConfidence, maxConfidence,
                    ss, sp, ac, limit);
        } else if (bi != null) {
            briefs = designBriefRepository.findByBatchId(bi);
        } else if (pt != null) {
            briefs = designBriefRepository.findByProductType(pt, limit);
        } else {
            briefs = designBriefRepository.findRecent(limit);
        }

        return briefs.stream()
                .map(this::toSummary)
                .collect(Collectors.toList());
    }

    private static String nullIfEmpty(String value) {
        return (value != null && !value.isEmpty()) ? value : null;
    }

    private static List<String> nullIfEmptyList(List<String> values) {
        if (values == null) {
            return null;
        }
        List<String> filtered = values.stream()
                .filter(s -> s != null && !s.isEmpty())
                .collect(Collectors.toList());
        return filtered.isEmpty() ? null : filtered;
    }

    @Authenticated
    @Mutation("createCustomBrief")
    public String createCustomBrief(String description, String productType) {
        LOG.infof("Creating custom brief: description=%s, productType=%s", description, productType);
        return celeryTaskDispatcher.dispatchCreateCustomBrief(description, productType)
                .await().indefinitely();
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
