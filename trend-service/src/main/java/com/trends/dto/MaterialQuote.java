package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * The recommended price and DTF print for one recommended material.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class MaterialQuote {

    @JsonProperty("external_id")
    private String externalId;

    @JsonProperty("requested_positioning")
    private String requestedPositioning;

    private String positioning;

    private String status;

    @JsonProperty("recommended_price")
    private Double recommendedPrice;

    private String currency;

    @JsonProperty("size_prices")
    private List<SizePrice> sizePrices = List.of();

    @JsonProperty("print_size")
    private PrintSize printSize;

    @JsonProperty("price_breakdown")
    private PriceBreakdown priceBreakdown;

    public String getExternalId() { return externalId; }
    public void setExternalId(String externalId) { this.externalId = externalId; }

    public String getRequestedPositioning() { return requestedPositioning; }
    public void setRequestedPositioning(String requestedPositioning) {
        this.requestedPositioning = requestedPositioning;
    }

    public String getPositioning() { return positioning; }
    public void setPositioning(String positioning) { this.positioning = positioning; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public Double getRecommendedPrice() { return recommendedPrice; }
    public void setRecommendedPrice(Double recommendedPrice) {
        this.recommendedPrice = recommendedPrice;
    }

    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }

    public List<SizePrice> getSizePrices() { return sizePrices; }
    public void setSizePrices(List<SizePrice> sizePrices) {
        this.sizePrices = sizePrices == null ? List.of() : sizePrices;
    }

    public PrintSize getPrintSize() { return printSize; }
    public void setPrintSize(PrintSize printSize) { this.printSize = printSize; }

    public PriceBreakdown getPriceBreakdown() { return priceBreakdown; }
    public void setPriceBreakdown(PriceBreakdown priceBreakdown) {
        this.priceBreakdown = priceBreakdown;
    }
}
