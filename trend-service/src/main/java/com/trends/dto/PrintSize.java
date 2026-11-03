package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * The suggested physical DTF print and the upscale target for it.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class PrintSize {

    @JsonProperty("width_in")
    private Double widthIn;

    @JsonProperty("height_in")
    private Double heightIn;

    private String placement;

    @JsonProperty("target_dpi")
    private Integer targetDpi;

    @JsonProperty("required_pixels")
    private PixelSize requiredPixels;

    @JsonProperty("source_pixels")
    private PixelSize sourcePixels;

    @JsonProperty("upscale_factor")
    private Double upscaleFactor;

    private Boolean estimated;

    public Double getWidthIn() { return widthIn; }
    public void setWidthIn(Double widthIn) { this.widthIn = widthIn; }

    public Double getHeightIn() { return heightIn; }
    public void setHeightIn(Double heightIn) { this.heightIn = heightIn; }

    public String getPlacement() { return placement; }
    public void setPlacement(String placement) { this.placement = placement; }

    public Integer getTargetDpi() { return targetDpi; }
    public void setTargetDpi(Integer targetDpi) { this.targetDpi = targetDpi; }

    public PixelSize getRequiredPixels() { return requiredPixels; }
    public void setRequiredPixels(PixelSize requiredPixels) { this.requiredPixels = requiredPixels; }

    public PixelSize getSourcePixels() { return sourcePixels; }
    public void setSourcePixels(PixelSize sourcePixels) { this.sourcePixels = sourcePixels; }

    public Double getUpscaleFactor() { return upscaleFactor; }
    public void setUpscaleFactor(Double upscaleFactor) { this.upscaleFactor = upscaleFactor; }

    public Boolean getEstimated() { return estimated; }
    public void setEstimated(Boolean estimated) { this.estimated = estimated; }
}
