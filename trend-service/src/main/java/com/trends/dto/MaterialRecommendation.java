package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * A substrate the LLM picked for a design brief.
 *
 * <p>The recommendations are stored with the brief as a JSON document, so the
 * fields are read by the names the document uses.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class MaterialRecommendation {

    @JsonProperty("external_id")
    private String externalId;

    private String name;

    private String color;

    @JsonProperty("substrate_type")
    private String substrateType;

    @JsonProperty("unit_cost")
    private Double unitCost;

    public MaterialRecommendation() {}

    public MaterialRecommendation(
            String externalId, String name, String color, String substrateType, Double unitCost) {
        this.externalId = externalId;
        this.name = name;
        this.color = color;
        this.substrateType = substrateType;
        this.unitCost = unitCost;
    }

    public String getExternalId() { return externalId; }
    public void setExternalId(String externalId) { this.externalId = externalId; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getColor() { return color; }
    public void setColor(String color) { this.color = color; }

    public String getSubstrateType() { return substrateType; }
    public void setSubstrateType(String substrateType) { this.substrateType = substrateType; }

    public Double getUnitCost() { return unitCost; }
    public void setUnitCost(Double unitCost) { this.unitCost = unitCost; }
}
