package com.trends.dto;

import java.util.List;

public class ActiveSeedSummary {
    private Integer id;
    private String query;
    private Long promotionScore;
    private List<String> productTags;

    public ActiveSeedSummary() {}

    public ActiveSeedSummary(Integer id, String query, Long promotionScore, List<String> productTags) {
        this.id = id;
        this.query = query;
        this.promotionScore = promotionScore;
        this.productTags = productTags;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }

    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }

    public Long getPromotionScore() { return promotionScore; }
    public void setPromotionScore(Long promotionScore) { this.promotionScore = promotionScore; }

    public List<String> getProductTags() { return productTags; }
    public void setProductTags(List<String> productTags) { this.productTags = productTags; }
}
