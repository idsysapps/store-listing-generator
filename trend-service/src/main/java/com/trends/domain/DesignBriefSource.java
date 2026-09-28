package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "design_brief_sources")
public class DesignBriefSource extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "brief_id", nullable = false)
    public DesignBrief designBrief;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "active_seed_id", nullable = false)
    public ActiveSeed activeSeed;

    @Column(name = "created_at")
    public OffsetDateTime createdAt;

    public DesignBriefSource() {}
}
