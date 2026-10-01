package com.trends.repository;

import com.trends.domain.DesignBrief;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@ApplicationScoped
public class DesignBriefRepository implements PanacheRepository<DesignBrief> {

    public List<DesignBrief> findByBatchId(String batchId) {
        return find("batchId", batchId).list();
    }

    public List<DesignBrief> findByProductType(String productType, int limit) {
        return find("productType ORDER BY createdAt DESC", productType)
                .page(0, limit)
                .list();
    }

    public List<DesignBrief> findRecent(int limit) {
        return find("ORDER BY createdAt DESC")
                .page(0, limit)
                .list();
    }

    public List<DesignBrief> findBySeedKeyword(String seedKeyword) {
        return getEntityManager()
                .createQuery(
                        "SELECT DISTINCT db FROM DesignBrief db "
                                + "JOIN db.sources dbs "
                                + "JOIN dbs.activeSeed a "
                                + "WHERE a.query = :seedKeyword "
                                + "ORDER BY db.createdAt DESC",
                        DesignBrief.class)
                .setParameter("seedKeyword", seedKeyword)
                .setMaxResults(5)
                .getResultList();
    }

    public List<DesignBrief> findWithFilters(
            String productType, String batchId,
            String startDate, String endDate,
            Integer minConfidence, Integer maxConfidence,
            List<String> sourceSeeds, List<String> specificProducts,
            String audienceContains, int limit) {

        boolean needsSourceJoin = sourceSeeds != null && !sourceSeeds.isEmpty();

        StringBuilder jpql = new StringBuilder();
        if (needsSourceJoin) {
            jpql.append("SELECT DISTINCT db FROM DesignBrief db JOIN db.sources dbs JOIN dbs.activeSeed a");
        } else {
            jpql.append("SELECT db FROM DesignBrief db");
        }

        List<String> conditions = new ArrayList<>();
        Map<String, Object> params = new HashMap<>();

        if (productType != null) {
            conditions.add("db.productType = :productType");
            params.put("productType", productType);
        }
        if (batchId != null) {
            conditions.add("db.batchId = :batchId");
            params.put("batchId", batchId);
        }
        if (startDate != null) {
            OffsetDateTime start = LocalDate.parse(startDate).atStartOfDay().atOffset(ZoneOffset.UTC);
            conditions.add("db.createdAt >= :startDate");
            params.put("startDate", start);
        }
        if (endDate != null) {
            OffsetDateTime end = LocalDate.parse(endDate).plusDays(1).atStartOfDay().atOffset(ZoneOffset.UTC);
            conditions.add("db.createdAt < :endDate");
            params.put("endDate", end);
        }
        if (minConfidence != null) {
            conditions.add("db.confidence >= :minConfidence");
            params.put("minConfidence", minConfidence);
        }
        if (maxConfidence != null) {
            conditions.add("db.confidence <= :maxConfidence");
            params.put("maxConfidence", maxConfidence);
        }
        if (needsSourceJoin) {
            conditions.add("a.query IN :sourceSeeds");
            params.put("sourceSeeds", sourceSeeds);
        }
        if (audienceContains != null) {
            conditions.add("LOWER(db.audience) LIKE :audienceContains");
            params.put("audienceContains", "%" + audienceContains.toLowerCase() + "%");
        }

        if (!conditions.isEmpty()) {
            jpql.append(" WHERE ");
            jpql.append(String.join(" AND ", conditions));
        }

        jpql.append(" ORDER BY db.createdAt DESC");

        var query = getEntityManager().createQuery(jpql.toString(), DesignBrief.class);
        for (var entry : params.entrySet()) {
            query.setParameter(entry.getKey(), entry.getValue());
        }
        query.setMaxResults(limit);

        List<DesignBrief> results = query.getResultList();

        if (specificProducts != null && !specificProducts.isEmpty()) {
            results = results.stream()
                    .filter(db -> db.specificProducts != null && hasOverlap(db.specificProducts, specificProducts))
                    .toList();
        }

        return results;
    }

    private boolean hasOverlap(String[] array, List<String> filter) {
        for (String item : array) {
            for (String f : filter) {
                if (item.equalsIgnoreCase(f)) {
                    return true;
                }
            }
        }
        return false;
    }
}
