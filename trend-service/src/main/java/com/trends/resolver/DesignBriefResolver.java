package com.trends.resolver;

import com.trends.config.GarmentPreviewColor;
import com.trends.config.ImageStorageConfig;
import com.trends.config.MaterialRecommendationReader;
import com.trends.config.PricingReader;
import com.trends.domain.DesignBrief;
import com.trends.domain.DesignBriefSource;
import com.trends.dto.DesignBriefConnection;
import com.trends.dto.DesignBriefOrderBy;
import com.trends.dto.DesignBriefSummary;
import com.trends.dto.DesignReviewFilter;
import com.trends.dto.PageInfo;
import com.trends.dto.StylePresetInfo;
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

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
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
    MaterialRecommendationReader materialRecommendationReader;

    @Inject
    GarmentPreviewColor garmentPreviewColor;

    @Inject
    PricingReader pricingReader;

    @Inject
    CeleryTaskDispatcher celeryTaskDispatcher;

    @Inject
    SecurityIdentity securityIdentity;

    private static final int MAX_LIMIT = 100;

    public DesignBriefConnection getDesignBriefs(
            String productType, String batchId,
            String startDate, String endDate,
            Integer minConfidence, Integer maxConfidence,
            List<String> sourceSeeds, List<String> specificProducts,
            String audienceContains, int limit, @DefaultValue("0") int offset,
            @DefaultValue("CREATED_AT_DESC") DesignBriefOrderBy orderBy) throws GraphQLException {
        return getDesignBriefs(
                productType, batchId, startDate, endDate, minConfidence, maxConfidence,
                sourceSeeds, specificProducts, audienceContains, null, null, limit, offset, orderBy);
    }

    @Query("designBriefs")
    public DesignBriefConnection getDesignBriefs(
            String productType, String batchId,
            String startDate, String endDate,
            Integer minConfidence, Integer maxConfidence,
            List<String> sourceSeeds, List<String> specificProducts,
            String audienceContains, String silhouette,
            @DefaultValue("ALL") DesignReviewFilter reviewStatus,
            int limit, @DefaultValue("0") int offset,
            @DefaultValue("CREATED_AT_DESC") DesignBriefOrderBy orderBy) throws GraphQLException {
        requireAuthentication();

        int effectiveLimit = Math.max(0, Math.min(limit, MAX_LIMIT));
        int effectiveOffset = Math.max(0, offset);

        String pt = nullIfEmpty(productType);
        String bi = nullIfEmpty(batchId);
        String sd = nullIfEmpty(startDate);
        String ed = nullIfEmpty(endDate);
        String ac = nullIfEmpty(audienceContains);
        String sil = nullIfEmpty(silhouette);
        List<String> ss = nullIfEmptyList(sourceSeeds);
        List<String> sp = nullIfEmptyList(specificProducts);
        DesignReviewFilter review = reviewStatus == null ? DesignReviewFilter.ALL : reviewStatus;
        boolean filtersByReview = review != DesignReviewFilter.ALL;

        LOG.debugf("Fetching design briefs: productType=%s, batchId=%s, limit=%d, offset=%d", pt, bi, effectiveLimit, effectiveOffset);

        if (effectiveLimit == 0) {
            long totalCount = filtersByReview
                    ? designBriefRepository.countWithFilters(
                            pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac, sil, review)
                    : sil == null
                    ? designBriefRepository.countWithFilters(pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac)
                    : designBriefRepository.countWithFilters(pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac, sil);
            PageInfo pageInfo = new PageInfo(totalCount > 0, false, (int) totalCount);
            return new DesignBriefConnection(List.of(), pageInfo);
        }

        List<DesignBrief> briefs;
        long totalCount;

        boolean hasAdvancedFilters = sd != null || ed != null
                || minConfidence != null || maxConfidence != null
                || (ss != null && !ss.isEmpty())
                || (sp != null && !sp.isEmpty())
                || ac != null
                || sil != null;

        DesignBriefOrderBy effectiveOrderBy = orderBy != null ? orderBy : DesignBriefOrderBy.CREATED_AT_DESC;

        if (filtersByReview || hasAdvancedFilters || (pt != null && bi != null) || pt != null || bi != null) {
            if (filtersByReview) {
                briefs = designBriefRepository.findWithFilters(
                        pt, bi, sd, ed, minConfidence, maxConfidence,
                        ss, sp, ac, effectiveLimit, effectiveOffset, effectiveOrderBy, sil, review);
                totalCount = designBriefRepository.countWithFilters(
                        pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac, sil, review);
            } else if (sil == null) {
                briefs = designBriefRepository.findWithFilters(
                        pt, bi, sd, ed, minConfidence, maxConfidence,
                        ss, sp, ac, effectiveLimit, effectiveOffset, effectiveOrderBy);
                totalCount = designBriefRepository.countWithFilters(
                        pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac);
            } else {
                briefs = designBriefRepository.findWithFilters(
                        pt, bi, sd, ed, minConfidence, maxConfidence,
                        ss, sp, ac, effectiveLimit, effectiveOffset, effectiveOrderBy, sil);
                totalCount = designBriefRepository.countWithFilters(
                        pt, bi, sd, ed, minConfidence, maxConfidence, ss, ac, sil);
            }
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

    private static final List<StylePresetInfo> STYLE_PRESETS = List.of(
            new StylePresetInfo("vintage_screenprint", "Vintage screenprint"),
            new StylePresetInfo("seventies_retro", "Seventies retro"),
            new StylePresetInfo("halftone_comic", "Comic book"),
            new StylePresetInfo("riso_two_color", "Riso two colour"),
            new StylePresetInfo("hand_inked_woodcut", "Hand inked woodcut"),
            new StylePresetInfo("kawaii_flat", "Kawaii flat"),
            new StylePresetInfo("art_nouveau", "Art nouveau"),
            new StylePresetInfo("distressed_band_tee", "Distressed band tee"),
            new StylePresetInfo("photorealistic", "Photorealistic"),
            new StylePresetInfo("3d", "3D"),
            new StylePresetInfo("2d", "2D"),
            new StylePresetInfo("flat_3d", "Flat 3D")
    );

    @Query("customBriefStylePresets")
    public List<StylePresetInfo> customBriefStylePresets() throws GraphQLException {
        requireAuthentication();
        return STYLE_PRESETS;
    }

    @Mutation("createCustomBrief")
    public String createCustomBrief(
            String description,
            String productType,
            boolean personalUse,
            @org.eclipse.microprofile.graphql.DefaultValue("qwen_image_2.1") String imageModel,
            String stylePreset
    ) throws GraphQLException {
        requireAuthentication();
        LOG.infof(
                "Creating custom brief: description=%s, productType=%s, personalUse=%s, imageModel=%s, stylePreset=%s",
                description, productType, personalUse, imageModel, stylePreset);
        return celeryTaskDispatcher
                .dispatchCreateCustomBrief(description, productType, personalUse, imageModel, stylePreset)
                .await().indefinitely();
    }

    @Mutation("requestRegeneration")
    @Transactional
    public DesignBriefSummary requestRegeneration(int briefId, String feedback, String stylePreset,
            String voice) throws GraphQLException {
        requireAuthentication();
        DesignBrief brief = designBriefRepository.findById((long) briefId);
        if (brief == null) {
            return null;
        }

        brief.imageKeyRaw = null;
        brief.imageKeyTransparent = null;
        brief.regenerationFeedback = feedback;
        brief.imageStatus = "pending";
        brief.imageError = null;

        // The same design in another look is a render away, so the curation
        // can compare them side by side (issue #228).
        if (stylePreset != null && !stylePreset.isBlank()) {
            brief.stylePreset = stylePreset;
            // The type follows the look, so the family that was stored is not
            // the family that will be drawn.
            brief.fontFamily = null;
        }
        if (voice != null && !voice.isBlank()) {
            brief.voice = voice;
            brief.fontFamily = null;
        }

        designBriefRepository.persist(brief);

        celeryTaskDispatcher.dispatchGenerateSingleBriefImage(briefId)
                .subscribe().with(
                        v -> LOG.infof("Dispatched image generation for briefId=%d", briefId),
                        e -> LOG.errorf(e, "Failed to dispatch image generation for briefId=%d", briefId)
                );

        return toSummary(brief);
    }

    @Mutation("rejectDesign")
    @Transactional
    public DesignBriefSummary rejectDesign(int briefId, String reason) throws GraphQLException {
        requireAuthentication();
        DesignBrief brief = designBriefRepository.findById((long) briefId);
        if (brief == null) {
            return null;
        }

        String trimmed = reason == null ? "" : reason.trim();
        brief.reviewStatus = "rejected";
        brief.rejectionReason = trimmed.isEmpty() ? null : trimmed;
        brief.rejectedAt = OffsetDateTime.now(ZoneOffset.UTC);
        designBriefRepository.persist(brief);
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
        summary.setListingDescription(brief.listingDescription);
        summary.setImageStatus(brief.imageStatus);
        summary.setImageError(brief.imageError);
        summary.setReviewStatus(brief.reviewStatus);
        summary.setRejectionReason(brief.rejectionReason);
        summary.setVoice(brief.voice);
        summary.setStylePreset(brief.stylePreset);
        summary.setFontFamily(brief.fontFamily);
        summary.setSilhouette(brief.silhouette);
        summary.setPersonalUse(brief.personalUse);
        summary.setRecommendedMaterials(
                materialRecommendationReader.read(brief.recommendedMaterials));
        summary.setPreviewBackgroundColor(
                garmentPreviewColor.forMaterials(summary.getRecommendedMaterials()));
        summary.setPricing(pricingReader.read(brief.pricing));
        return summary;
    }
}
