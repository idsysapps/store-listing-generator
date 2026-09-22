package com.trends.domain;

import io.quarkus.hibernate.orm.panache.PanacheEntityBase;
import jakarta.persistence.*;
import java.time.OffsetDateTime;

@Entity
@Table(name = "trend_scores")
public class TrendScore extends PanacheEntityBase {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    public Integer id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "query_id", nullable = false)
    public TrendQuery trendQuery;

    @Column(name = "score", nullable = false)
    public Integer score;

    @Column(name = "delta")
    public Integer delta;

    @Column(name = "region", length = 10)
    public String region;

    @Column(name = "fetched_at")
    public OffsetDateTime fetchedAt;

    public TrendScore() {}

    public TrendScore(TrendQuery trendQuery, Integer score, Integer delta, String region) {
        this.trendQuery = trendQuery;
        this.score = score;
        this.delta = delta;
        this.region = region;
        this.fetchedAt = OffsetDateTime.now();
    }
}
