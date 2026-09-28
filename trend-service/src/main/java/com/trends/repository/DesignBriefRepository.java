package com.trends.repository;

import com.trends.domain.DesignBrief;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;

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
}
