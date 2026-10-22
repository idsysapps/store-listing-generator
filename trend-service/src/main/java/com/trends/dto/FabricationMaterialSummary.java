package com.trends.dto;

import java.math.BigDecimal;
import java.util.List;

/**
 * A material the cost book offers.
 *
 * <p>A material can serve several product types, so the product types are a
 * list (issue #186).
 */
public class FabricationMaterialSummary {

    private Integer id;
    private String externalId;
    private String name;
    private String brand;
    private String substrateType;
    private String color;
    private List<String> sizes;
    private String materialComposition;
    private List<String> productTypes;
    private BigDecimal unitCost;

    public FabricationMaterialSummary() {}

    public FabricationMaterialSummary(
            Integer id, String externalId, String name, String brand, String substrateType,
            String color, List<String> sizes, String materialComposition,
            List<String> productTypes, BigDecimal unitCost) {
        this.id = id;
        this.externalId = externalId;
        this.name = name;
        this.brand = brand;
        this.substrateType = substrateType;
        this.color = color;
        this.sizes = sizes;
        this.materialComposition = materialComposition;
        this.productTypes = productTypes;
        this.unitCost = unitCost;
    }

    public Integer getId() { return id; }
    public void setId(Integer id) { this.id = id; }

    public String getExternalId() { return externalId; }
    public void setExternalId(String externalId) { this.externalId = externalId; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getBrand() { return brand; }
    public void setBrand(String brand) { this.brand = brand; }

    public String getSubstrateType() { return substrateType; }
    public void setSubstrateType(String substrateType) { this.substrateType = substrateType; }

    public String getColor() { return color; }
    public void setColor(String color) { this.color = color; }

    public List<String> getSizes() { return sizes; }
    public void setSizes(List<String> sizes) { this.sizes = sizes; }

    public String getMaterialComposition() { return materialComposition; }
    public void setMaterialComposition(String materialComposition) {
        this.materialComposition = materialComposition;
    }

    public List<String> getProductTypes() { return productTypes; }
    public void setProductTypes(List<String> productTypes) { this.productTypes = productTypes; }

    public BigDecimal getUnitCost() { return unitCost; }
    public void setUnitCost(BigDecimal unitCost) { this.unitCost = unitCost; }
}
