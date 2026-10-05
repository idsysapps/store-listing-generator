package com.trends.resolver;

import com.trends.domain.SourceHealth;
import com.trends.dto.SourceHealthSummary;
import com.trends.dto.SystemStatus;
import com.trends.event.CeleryTaskDispatcher;
import com.trends.repository.SourceHealthRepository;
import io.quarkus.security.identity.SecurityIdentity;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import jakarta.persistence.EntityManager;
import org.eclipse.microprofile.graphql.GraphQLApi;
import org.eclipse.microprofile.graphql.GraphQLException;
import org.eclipse.microprofile.graphql.Mutation;
import org.eclipse.microprofile.graphql.Query;
import org.jboss.logging.Logger;

import java.util.List;
import java.util.stream.Collectors;

@GraphQLApi
@ApplicationScoped
public class SystemResolver {

    private static final Logger LOG = Logger.getLogger(SystemResolver.class);

    @Inject
    EntityManager entityManager;

    @Inject
    SourceHealthRepository sourceHealthRepository;

    @Inject
    CeleryTaskDispatcher celeryTaskDispatcher;

    @Inject
    SecurityIdentity securityIdentity;

    @Query("systemStatus")
    public SystemStatus getSystemStatus() throws GraphQLException {
        requireAuthentication();
        int pending = countQuery("SELECT COUNT(*) FROM seed_candidates WHERE status = 'pending'");
        int promoted = countQuery("SELECT COUNT(*) FROM seed_candidates WHERE status = 'promoted'");
        int archived = countQuery("SELECT COUNT(*) FROM seed_candidates WHERE status = 'archived'");
        int activeSeeds = countQuery("SELECT COUNT(*) FROM active_seeds WHERE archived_at IS NULL");
        int briefsTotal = countQuery("SELECT COUNT(*) FROM design_briefs");
        int briefsWithImages = countQuery("SELECT COUNT(*) FROM design_briefs WHERE image_key_raw IS NOT NULL");
        int briefsPendingImages = countQuery("SELECT COUNT(*) FROM design_briefs WHERE image_key_raw IS NULL");
        int briefsPendingRegen = countQuery("SELECT COUNT(*) FROM design_briefs WHERE regeneration_feedback IS NOT NULL AND image_key_raw IS NULL");

        return new SystemStatus(pending, promoted, archived, activeSeeds,
                briefsTotal, briefsWithImages, briefsPendingImages, briefsPendingRegen);
    }

    @Query("sourceHealth")
    public List<SourceHealthSummary> getSourceHealth() throws GraphQLException {
        requireAuthentication();
        return sourceHealthRepository.listAll().stream()
                .map(this::toSummary)
                .collect(Collectors.toList());
    }

    @Mutation("triggerCuration")
    public String triggerCuration() throws GraphQLException {
        requireAuthentication();
        String taskId = celeryTaskDispatcher
                .dispatchTask("store_listing.orchestration.tasks.curate_seeds_task")
                .await().indefinitely();
        LOG.infof("Triggered curation task: %s", taskId);
        return taskId;
    }

    @Mutation("triggerDesignBriefs")
    public String triggerDesignBriefs() throws GraphQLException {
        requireAuthentication();
        String taskId = celeryTaskDispatcher
                .dispatchTask("store_listing.orchestration.tasks.generate_design_briefs_task")
                .await().indefinitely();
        LOG.infof("Triggered design briefs task: %s", taskId);
        return taskId;
    }

    @Mutation("triggerImageGeneration")
    public String triggerImageGeneration() throws GraphQLException {
        requireAuthentication();
        String taskId = celeryTaskDispatcher
                .dispatchTask("store_listing.orchestration.tasks.generate_design_images_task")
                .await().indefinitely();
        LOG.infof("Triggered image generation task: %s", taskId);
        return taskId;
    }

    private void requireAuthentication() throws GraphQLException {
        if (securityIdentity == null || securityIdentity.isAnonymous()) {
            throw new GraphQLException("Authentication required");
        }
    }

    private int countQuery(String sql) {
        Object result = entityManager.createNativeQuery(sql).getSingleResult();
        return ((Number) result).intValue();
    }

    private SourceHealthSummary toSummary(SourceHealth sh) {
        return new SourceHealthSummary(
                sh.source, sh.consecutiveFailures, sh.consecutiveEmpty,
                sh.lastError, sh.lastErrorAt, sh.lastSuccessAt,
                sh.issueOpen, sh.lastIssueUrl);
    }
}
