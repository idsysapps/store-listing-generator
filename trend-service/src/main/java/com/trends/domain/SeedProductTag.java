package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "seed_product_tags")
public class SeedProductTag extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "active_seed_id", nullable = false)
    public ActiveSeed activeSeed;

    @Column(name = "tag", nullable = false, length = 30)
    public String tag;

    @Column(name = "created_at")
    public OffsetDateTime createdAt;

    public SeedProductTag() {}
}
