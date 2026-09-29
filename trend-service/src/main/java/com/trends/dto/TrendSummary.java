package com.trends.dto;

import java.util.List;

public class TrendSummary {
    private String query;
    private Long latestScore;
    private Long velocity;
    private List<RegionScore> topRegions;
    private String source;

    public TrendSummary() {}

    public TrendSummary(String query, Long latestScore, Long velocity, List<RegionScore> topRegions) {
        this(query, latestScore, velocity, topRegions, null);
    }

    public TrendSummary(String query, Long latestScore, Long velocity, List<RegionScore> topRegions, String source) {
        this.query = query;
        this.latestScore = latestScore;
        this.velocity = velocity;
        this.topRegions = topRegions;
        this.source = source;
    }

    public String getQuery() {
        return query;
    }

    public void setQuery(String query) {
        this.query = query;
    }

    public Long getLatestScore() {
        return latestScore;
    }

    public void setLatestScore(Long latestScore) {
        this.latestScore = latestScore;
    }

    public Long getVelocity() {
        return velocity;
    }

    public void setVelocity(Long velocity) {
        this.velocity = velocity;
    }

    public List<RegionScore> getTopRegions() {
        return topRegions;
    }

    public void setTopRegions(List<RegionScore> topRegions) {
        this.topRegions = topRegions;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }
}
