package com.trends.dto;

import java.time.OffsetDateTime;
import java.util.List;

public class TrendUpdate {
    private String query;
    private Integer delta;
    private Integer score;
    private OffsetDateTime timestamp;

    public TrendUpdate() {}

    public TrendUpdate(String query, Integer delta, Integer score, OffsetDateTime timestamp) {
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

    public Integer getDelta() {
        return delta;
    }

    public void setDelta(Integer delta) {
        this.delta = delta;
    }

    public Integer getScore() {
        return score;
    }

    public void setScore(Integer score) {
        this.score = score;
    }

    public OffsetDateTime getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(OffsetDateTime timestamp) {
        this.timestamp = timestamp;
    }
}
