package com.trends.repository;

import com.trends.domain.FabricationMaterial;
import io.quarkus.hibernate.orm.panache.PanacheRepository;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.transaction.Transactional;

import java.time.OffsetDateTime;
import java.util.List;

@ApplicationScoped
public class FabricationMaterialRepository implements PanacheRepository<FabricationMaterial> {

    public List<FabricationMaterial> findByProductType(String productType) {
        return find("productType = ?1 AND active = true ORDER BY name, color", productType).list();
    }

    public FabricationMaterial findByExternalId(String externalId) {
        return find("externalId", externalId).firstResult();
    }

    public List<FabricationMaterial> findAllActive() {
        return find("active = true ORDER BY productType, name, color").list();
    }

    @Transactional
    public void deactivateByExternalId(String externalId) {
        update("active = false, updatedAt = ?1 WHERE externalId = ?2",
                OffsetDateTime.now(), externalId);
    }
}
