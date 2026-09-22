package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;
import java.util.List;

@Entity
@Table(name = "trend_queries")
public class TrendQuery extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @Column(name = "seed_keyword", nullable = false, length = 255)
    public String seedKeyword;

    @Column(name = "query", nullable = false, length = 500)
    public String query;

    @Column(name = "created_at")
    public OffsetDateTime createdAt;

    @OneToMany(mappedBy = "trendQuery", fetch = FetchType.LAZY)
    public List<TrendScore> scores;

    public TrendQuery() {}

    public TrendQuery(String seedKeyword, String query) {
        this.seedKeyword = seedKeyword;
        this.query = query;
        this.createdAt = OffsetDateTime.now();
    }
}
