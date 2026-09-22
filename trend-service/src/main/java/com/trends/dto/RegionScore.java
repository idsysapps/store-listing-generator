package com.trends.dto;

public class RegionScore {
    private String region;
    private Integer score;

    public RegionScore() {}

    public RegionScore(String region, Integer score) {
        this.region = region;
        this.score = score;
    }

    public String getRegion() {
        return region;
    }

    public void setRegion(String region) {
        this.region = region;
    }

    public Integer getScore() {
        return score;
    }

    public void setScore(Integer score) {
        this.score = score;
    }
}
