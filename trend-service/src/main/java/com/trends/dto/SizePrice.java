package com.trends.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * The price for one garment size.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class SizePrice {

    private String size;
    private Double price;

    public SizePrice() {}

    public SizePrice(String size, Double price) {
        this.size = size;
        this.price = price;
    }

    public String getSize() { return size; }
    public void setSize(String size) { this.size = size; }

    public Double getPrice() { return price; }
    public void setPrice(Double price) { this.price = price; }
}
