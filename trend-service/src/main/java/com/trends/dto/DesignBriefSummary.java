package com.trends.dto;

import java.time.OffsetDateTime;
import java.util.List;

public class DesignBriefSummary {
    private Integer id;
    private String concept;
    private String productType;
    private List<String> specificProducts;
    private String audience;
    private String visualStyle;
    private Integer confidence;
    private String reasoning;
    private String llmModel;
    private String batchId;
    private OffsetDateTime createdAt;
    private List<String> sourceSeeds;
    private String imageKeyRaw;
    private String imageKeyTransparent;
    private String imageUrl;
    private String imageTransparentUrl;
    private String layoutType;
    private String headlineText;
    private String taglineText;
    private String fontColor;
    private String sceneDescription;
    private Boolean personalUse;

    public DesignBriefSummary() {}

    public DesignBriefSummary(Integer id, String concept, String productType,
            List<String> specificProducts, String audience, String visualStyle,
            Integer confidence, String reasoning, String llmModel, String batchId,
            OffsetDateTime createdAt, List<String> sourceSeeds,
            String imageKeyRaw, String imageKeyTransparent) {
        this.id = id;
        this.concept = concept;
        this.productType = productType;
        this.specificProducts = specificProducts;
        this.audience = audience;
        this.visualStyle = visualStyle;
        this.confidence = confidence;
        this.reasoning = reasoning;
        this.llmModel = llmModel;
        this.batchId = batchId;
        this.createdAt = createdAt;
        this.sourceSeeds = sourceSeeds;
        this.imageKeyRaw = imageKeyRaw;
        this.imageKeyTransparent = imageKeyTransparent;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }

    public String getConcept() { return concept; }
    public void setConcept(String concept) { this.concept = concept; }

    public String getProductType() { return productType; }
    public void setProductType(String productType) { this.productType = productType; }

    public List<String> getSpecificProducts() { return specificProducts; }
    public void setSpecificProducts(List<String> specificProducts) { this.specificProducts = specificProducts; }

    public String getAudience() { return audience; }
    public void setAudience(String audience) { this.audience = audience; }

    public String getVisualStyle() { return visualStyle; }
    public void setVisualStyle(String visualStyle) { this.visualStyle = visualStyle; }

    public Integer getConfidence() { return confidence; }
    public void setConfidence(Integer confidence) { this.confidence = confidence; }

    public String getReasoning() { return reasoning; }
    public void setReasoning(String reasoning) { this.reasoning = reasoning; }

    public String getLlmModel() { return llmModel; }
    public void setLlmModel(String llmModel) { this.llmModel = llmModel; }

    public String getBatchId() { return batchId; }
    public void setBatchId(String batchId) { this.batchId = batchId; }

    public OffsetDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(OffsetDateTime createdAt) { this.createdAt = createdAt; }

    public List<String> getSourceSeeds() { return sourceSeeds; }
    public void setSourceSeeds(List<String> sourceSeeds) { this.sourceSeeds = sourceSeeds; }

    public String getImageKeyRaw() { return imageKeyRaw; }
    public void setImageKeyRaw(String imageKeyRaw) { this.imageKeyRaw = imageKeyRaw; }

    public String getImageKeyTransparent() { return imageKeyTransparent; }
    public void setImageKeyTransparent(String imageKeyTransparent) { this.imageKeyTransparent = imageKeyTransparent; }

    public String getImageUrl() { return imageUrl; }
    public void setImageUrl(String imageUrl) { this.imageUrl = imageUrl; }

    public String getImageTransparentUrl() { return imageTransparentUrl; }
    public void setImageTransparentUrl(String imageTransparentUrl) { this.imageTransparentUrl = imageTransparentUrl; }

    public String getLayoutType() { return layoutType; }
    public void setLayoutType(String layoutType) { this.layoutType = layoutType; }

    public String getHeadlineText() { return headlineText; }
    public void setHeadlineText(String headlineText) { this.headlineText = headlineText; }

    public String getTaglineText() { return taglineText; }
    public void setTaglineText(String taglineText) { this.taglineText = taglineText; }

    public String getFontColor() { return fontColor; }
    public void setFontColor(String fontColor) { this.fontColor = fontColor; }

    public String getSceneDescription() { return sceneDescription; }
    public void setSceneDescription(String sceneDescription) { this.sceneDescription = sceneDescription; }

    public Boolean getPersonalUse() { return personalUse; }
    public void setPersonalUse(Boolean personalUse) { this.personalUse = personalUse; }
}
