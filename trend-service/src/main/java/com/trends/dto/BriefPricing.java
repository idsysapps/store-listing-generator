package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * The pricing snapshot stored with a design brief.
 *
 * <p>The positioning is the model's decision; the per-material quotes are the
 * deterministic prices the pipeline calculated (issue #295).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class BriefPricing {

    @JsonProperty("requested_positioning")
    private String requestedPositioning;

    @JsonProperty("effective_positioning")
    private String effectivePositioning;

    @JsonProperty("positioning_reason")
    private String positioningReason;

    private Integer confidence;

    @JsonProperty("target_margin")
    private Double targetMargin;

    private String status;

    private String currency;

    @JsonProperty("policy_version")
    private Integer policyVersion;

    @JsonProperty("print_rate_per_sq_in")
    private Double printRatePerSqIn;

    private List<MaterialQuote> quotes = List.of();

    public String getRequestedPositioning() { return requestedPositioning; }
    public void setRequestedPositioning(String requestedPositioning) {
        this.requestedPositioning = requestedPositioning;
    }

    public String getEffectivePositioning() { return effectivePositioning; }
    public void setEffectivePositioning(String effectivePositioning) {
        this.effectivePositioning = effectivePositioning;
    }

    public String getPositioningReason() { return positioningReason; }
    public void setPositioningReason(String positioningReason) {
        this.positioningReason = positioningReason;
    }

    public Integer getConfidence() { return confidence; }
    public void setConfidence(Integer confidence) { this.confidence = confidence; }

    public Double getTargetMargin() { return targetMargin; }
    public void setTargetMargin(Double targetMargin) { this.targetMargin = targetMargin; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }

    public Integer getPolicyVersion() { return policyVersion; }
    public void setPolicyVersion(Integer policyVersion) { this.policyVersion = policyVersion; }

    public Double getPrintRatePerSqIn() { return printRatePerSqIn; }
    public void setPrintRatePerSqIn(Double printRatePerSqIn) {
        this.printRatePerSqIn = printRatePerSqIn;
    }

    public List<MaterialQuote> getQuotes() { return quotes; }
    public void setQuotes(List<MaterialQuote> quotes) {
        this.quotes = quotes == null ? List.of() : quotes;
    }
}
