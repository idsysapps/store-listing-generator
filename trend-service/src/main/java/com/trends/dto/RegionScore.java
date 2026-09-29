package com.trends.dto;

public class RegionScore {
    private String region;
    private Long score;

    public RegionScore() {}

    public RegionScore(String region, Long score) {
        this.region = region;
        this.score = score;
    }

    public String getRegion() {
        return region;
    }

    public void setRegion(String region) {
        this.region = region;
    }

    public Long getScore() {
        return score;
    }

    public void setScore(Long score) {
        this.score = score;
    }
}
