package com.trends.dto;

import java.util.List;

public class TrendSummary {
    private String query;
    private String seedKeyword;
    private Long latestScore;
    private Long velocity;
    private List<RegionScore> topRegions;
    private String source;
    private Integer normalizedScore;
    private List<DesignBriefSummary> designBriefs;

    public TrendSummary() {}

    public TrendSummary(String query, Long latestScore, Long velocity, List<RegionScore> topRegions) {
        this(query, null, latestScore, velocity, topRegions, null, null, List.of());
    }

    public TrendSummary(String query, Long latestScore, Long velocity, List<RegionScore> topRegions, String source) {
        this(query, null, latestScore, velocity, topRegions, source, null, List.of());
    }

    public TrendSummary(String query, String seedKeyword, Long latestScore, Long velocity,
            List<RegionScore> topRegions, String source, Integer normalizedScore,
            List<DesignBriefSummary> designBriefs) {
        this.query = query;
        this.seedKeyword = seedKeyword;
        this.latestScore = latestScore;
        this.velocity = velocity;
        this.topRegions = topRegions;
        this.source = source;
        this.normalizedScore = normalizedScore;
        this.designBriefs = designBriefs;
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

    public String getSeedKeyword() {
        return seedKeyword;
    }

    public void setSeedKeyword(String seedKeyword) {
        this.seedKeyword = seedKeyword;
    }

    public String getSource() {
        return source;
    }

    public void setSource(String source) {
        this.source = source;
    }

    public Integer getNormalizedScore() {
        return normalizedScore;
    }

    public void setNormalizedScore(Integer normalizedScore) {
        this.normalizedScore = normalizedScore;
    }

    public List<DesignBriefSummary> getDesignBriefs() {
        return designBriefs;
    }

    public void setDesignBriefs(List<DesignBriefSummary> designBriefs) {
        this.designBriefs = designBriefs;
    }
}
