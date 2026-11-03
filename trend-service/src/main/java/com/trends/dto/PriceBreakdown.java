package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The costs behind one recommended price.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class PriceBreakdown {

    @JsonProperty("material_cost")
    private Double materialCost;

    @JsonProperty("print_cost")
    private Double printCost;

    @JsonProperty("print_rate_per_sq_in")
    private Double printRatePerSqIn;

    @JsonProperty("marketplace_fee")
    private Double marketplaceFee;

    @JsonProperty("target_margin")
    private Double targetMargin;

    @JsonProperty("actual_margin")
    private Double actualMargin;

    public Double getMaterialCost() { return materialCost; }
    public void setMaterialCost(Double materialCost) { this.materialCost = materialCost; }

    public Double getPrintCost() { return printCost; }
    public void setPrintCost(Double printCost) { this.printCost = printCost; }

    public Double getPrintRatePerSqIn() { return printRatePerSqIn; }
    public void setPrintRatePerSqIn(Double printRatePerSqIn) {
        this.printRatePerSqIn = printRatePerSqIn;
    }

    public Double getMarketplaceFee() { return marketplaceFee; }
    public void setMarketplaceFee(Double marketplaceFee) { this.marketplaceFee = marketplaceFee; }

    public Double getTargetMargin() { return targetMargin; }
    public void setTargetMargin(Double targetMargin) { this.targetMargin = targetMargin; }

    public Double getActualMargin() { return actualMargin; }
    public void setActualMargin(Double actualMargin) { this.actualMargin = actualMargin; }
}
