package com.trends.resolver;

import com.trends.config.ImageStorageConfig;
import com.trends.domain.DesignBrief;
import com.trends.domain.DesignBriefSource;
import com.trends.dto.DesignBriefConnection;
import com.trends.dto.DesignBriefOrderBy;
import com.trends.dto.DesignBriefSummary;
import com.trends.dto.PageInfo;
import com.trends.event.CeleryTaskDispatcher;
import com.trends.repository.DesignBriefRepository;
import io.quarkus.security.identity.SecurityIdentity;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.transaction.Transactional;
import org.eclipse.microprofile.graphql.DefaultValue;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.GraphQLException;
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

    @Inject
    SecurityIdentity securityIdentity;

    private static final int MAX_LIMIT = 100;

    @Query("designBriefs")
    public DesignBriefConnection getDesignBriefs(
            String productType, String batchId,
            String startDate, String endDate,
            Integer minConfidence, Integer maxConfidence,
            List<String> sourceSeeds, List<String> specificProducts,
            String audienceContains, int limit, @DefaultValue("0") int offset,
            @DefaultValue("CREATED_AT_DESC") DesignBriefOrderBy orderBy) throws GraphQLException {
        requireAuthentication();

        int effectiveLimit = Math.max(0, Math.min(limit, MAX_LIMIT));
        int effectiveOffset = Math.max(0, offset);

        String pt = nullIfEmpty(productType);
        String bi = nullIfEmpty(batchId);
        String sd = nullIfEmpty(startDate);
        String ed = nullIfEmpty(endDate);
        String ac = nullIfEmpty(audienceContains);
        List<String> ss = nullIfEmptyList(sourceSeeds);
        List<String> sp = nullIfEmptyList(specificProducts);

        LOG.debugf("Fetching design briefs: productType=%s, batchId=%s, limit=%d, offset=%d", pt, bi, effectiveLimit, effectiveOffset);

        if (effectiveLimit == 0) {
            long totalCount = designBriefRepository.countWithFilters(pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac);
            PageInfo pageInfo = new PageInfo(totalCount > 0, false, (int) totalCount);
            return new DesignBriefConnection(List.of(), pageInfo);
        }

        List<DesignBrief> briefs;
        long totalCount;

        boolean hasAdvancedFilters = sd != null || ed != null
                || minConfidence != null || maxConfidence != null
                || (ss != null && !ss.isEmpty())
                || (sp != null && !sp.isEmpty())
                || ac != null;

        DesignBriefOrderBy effectiveOrderBy = orderBy != null ? orderBy : DesignBriefOrderBy.CREATED_AT_DESC;

        if (hasAdvancedFilters || (pt != null && bi != null) || pt != null || bi != null) {
            briefs = designBriefRepository.findWithFilters(
                    pt, bi, sd, ed, minConfidence, maxConfidence,
                    ss, sp, ac, effectiveLimit, effectiveOffset, effectiveOrderBy);
            totalCount = designBriefRepository.countWithFilters(
                    pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac);
        } else {
            briefs = designBriefRepository.findRecentPaginated(effectiveLimit, effectiveOffset, effectiveOrderBy);
            totalCount = designBriefRepository.countAll();
        }

        List<DesignBriefSummary> items = briefs.stream()
                .map(this::toSummary)
                .collect(Collectors.toList());

        PageInfo pageInfo = new PageInfo(
                effectiveOffset + effectiveLimit < totalCount,
                effectiveOffset > 0,
                (int) totalCount
        );

        return new DesignBriefConnection(items, pageInfo);
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

    @Mutation("createCustomBrief")
    public String createCustomBrief(String description, String productType, boolean personalUse) throws GraphQLException {
        requireAuthentication();
        LOG.infof("Creating custom brief: description=%s, productType=%s, personalUse=%s", description, productType, personalUse);
        return celeryTaskDispatcher.dispatchCreateCustomBrief(description, productType, personalUse)
                .await().indefinitely();
    }

    @Mutation("requestRegeneration")
    @Transactional
    public DesignBriefSummary requestRegeneration(int briefId, String feedback) throws GraphQLException {
        requireAuthentication();
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

    private void requireAuthentication() throws GraphQLException {
        if (securityIdentity == null || securityIdentity.isAnonymous()) {
            throw new GraphQLException("Authentication required");
        }
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
        summary.setLayoutType(brief.layoutType);
        summary.setHeadlineText(brief.headlineText);
        summary.setTaglineText(brief.taglineText);
        summary.setFontColor(brief.fontColor);
        summary.setSceneDescription(brief.sceneDescription);
        summary.setPersonalUse(brief.personalUse);
        return summary;
    }
}
