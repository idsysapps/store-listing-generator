package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.OffsetDateTime;

@Entity
@Table(name = "fabrication_materials")
public class FabricationMaterial extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @Column(name = "external_id", nullable = false, unique = true, length = 100)
    public String externalId;

    @Column(name = "name", nullable = false, length = 255)
    public String name;

    @Column(name = "brand", length = 100)
    public String brand;

    @Column(name = "substrate_type", nullable = false, length = 50)
    public String substrateType;

    @Column(name = "color", length = 100)
    public String color;

    @Column(name = "sizes", columnDefinition = "TEXT[]")
    public String[] sizes;

    @Column(name = "material_composition", length = 255)
    public String materialComposition;

    @Column(name = "product_type", nullable = false, length = 50)
    public String productType;

    @Column(name = "sku", length = 100)
    public String sku;

    @Column(name = "unit_cost", precision = 10, scale = 2)
    public BigDecimal unitCost;

    @Column(name = "active", nullable = false)
    public Boolean active = true;

    @Column(name = "attributes", columnDefinition = "JSONB")
    public String attributes;

    @Column(name = "updated_at")
    public OffsetDateTime updatedAt;

    @Column(name = "created_at")
    public OffsetDateTime createdAt;

    public FabricationMaterial() {}
}
