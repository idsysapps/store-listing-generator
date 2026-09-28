package com.trends.repository;

import com.trends.domain.ActiveSeed;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;

@ApplicationScoped
public class ActiveSeedRepository implements PanacheRepository<ActiveSeed> {

    public List<ActiveSeed> findActiveSeeds(int limit) {
        return find("archivedAt IS NULL ORDER BY promotionScore DESC")
                .page(0, limit)
                .list();
    }

    public List<ActiveSeed> findActiveSeedsByTag(String tag, int limit) {
        return getEntityManager()
                .createQuery(
                        "SELECT DISTINCT a FROM ActiveSeed a JOIN a.productTags t "
                                + "WHERE a.archivedAt IS NULL AND t.tag = :tag "
                                + "ORDER BY a.promotionScore DESC",
                        ActiveSeed.class)
                .setParameter("tag", tag)
                .setMaxResults(limit)
                .getResultList();
    }

    public List<String> findDistinctTags() {
        return getEntityManager()
                .createQuery(
                        "SELECT DISTINCT t.tag FROM SeedProductTag t ORDER BY t.tag",
                        String.class)
                .getResultList();
    }
}
