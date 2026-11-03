package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * A pixel width and height.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class PixelSize {

    private Integer width;
    private Integer height;

    public Integer getWidth() { return width; }
    public void setWidth(Integer width) { this.width = width; }

    public Integer getHeight() { return height; }
    public void setHeight(Integer height) { this.height = height; }
}
