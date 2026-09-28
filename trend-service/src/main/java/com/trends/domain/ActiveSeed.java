package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.List;

@Entity
@Table(name = "active_seeds")
public class ActiveSeed extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @Column(name = "query", nullable = false, length = 500)
    public String query;

    @Column(name = "promotion_score", nullable = false)
    public Long promotionScore;

    @Column(name = "promoted_from_candidate_id")
    public Integer promotedFromCandidateId;

    @Column(name = "added_at")
    public OffsetDateTime addedAt;

    @Column(name = "archived_at")
    public OffsetDateTime archivedAt;

    @OneToMany(mappedBy = "activeSeed", fetch = FetchType.LAZY)
    public List<SeedProductTag> productTags;

    public ActiveSeed() {}
}
