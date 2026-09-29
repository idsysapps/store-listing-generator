package com.trends.dto;

import java.time.OffsetDateTime;
import java.util.List;

public class TrendUpdate {
    private String query;
    private Long delta;
    private Long score;
    private OffsetDateTime timestamp;

    public TrendUpdate() {}

    public TrendUpdate(String query, Long delta, Long score, OffsetDateTime timestamp) {
        this.query = query;
        this.delta = delta;
        this.score = score;
        this.timestamp = timestamp;
    }

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public Long getDelta() {
        return delta;
    }

    public void setDelta(Long delta) {
        this.delta = delta;
    }

    public Long getScore() {
        return score;
    }

    public void setScore(Long score) {
        this.score = score;
    }

    public OffsetDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(OffsetDateTime timestamp) {
        this.timestamp = timestamp;
    }
}
