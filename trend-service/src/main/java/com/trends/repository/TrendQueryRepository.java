package com.trends.repository;

import com.trends.domain.TrendQuery;
import com.trends.domain.TrendScore;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.persistence.EntityManager;
import jakarta.persistence.TypedQuery;
import java.util.Base64;
import java.util.List;

@ApplicationScoped
public class TrendQueryRepository implements PanacheRepository<TrendQuery> {

    public List<TrendQuery> findPaginated(int limit, int offset) {
        return find("ORDER BY createdAt DESC")
                .page(offset / limit, limit)
                .list();
    }

    public long countAll() {
        return count();
    }

    public TrendQuery findByQuery(String query) {
        return find("query", query).firstResult();
    }

    public List<TrendScore> findScoresByQueryId(Integer queryId, int limit, int offset) {
        return getEntityManager()
                .createQuery(
                        "SELECT ts FROM TrendScore ts WHERE ts.trendQuery.id = :queryId ORDER BY ts.fetchedAt DESC",
                        TrendScore.class)
                .setParameter("queryId", queryId)
                .setFirstResult(offset)
                .setMaxResults(limit)
                .getResultList();
    }

    public List<TrendScore> findLatestScoresByQueryId(Integer queryId, int limit) {
        return getEntityManager()
                .createQuery(
                        "SELECT ts FROM TrendScore ts WHERE ts.trendQuery.id = :queryId ORDER BY ts.fetchedAt DESC",
                        TrendScore.class)
                .setParameter("queryId", queryId)
                .setFirstResult(0)
                .setMaxResults(limit)
                .getResultList();
    }

    public List<TrendScore> findScoresByQuery(String query, int days) {
        return getEntityManager()
                .createQuery(
                        "SELECT ts FROM TrendScore ts WHERE ts.trendQuery.query = :query AND ts.fetchedAt >= CURRENT_TIMESTAMP - :days ORDER BY ts.fetchedAt ASC",
                        TrendScore.class)
                .setParameter("query", query)
                .setParameter("days", days)
                .getResultList();
    }

    public String encodeCursor(Integer id) {
        return Base64.getEncoder().encodeToString(("cursor:" + id).getBytes());
    }

    public Integer decodeCursor(String cursor) {
        if (cursor == null || cursor.isEmpty()) {
            return null;
        }
        String decoded = new String(Base64.getDecoder().decode(cursor));
        return Integer.parseInt(decoded.replace("cursor:", ""));
    }
}
