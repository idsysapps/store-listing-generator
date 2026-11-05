package com.trends.config;

import com.trends.dto.MaterialRecommendation;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class GarmentPreviewColorTest {

    private final GarmentPreviewColor previewColour = new GarmentPreviewColor();

    private static MaterialRecommendation garment(String colour) {
        return new MaterialRecommendation("shirt-1", "Core Adult", colour, "t-shirt", 2.39);
    }

    @Test
    void mapsTheDarkCatalogueColours() {
        assertEquals("#0B0B0B", previewColour.forMaterials(List.of(garment("Jet Black"))));
        assertEquals("#3A3A3A", previewColour.forMaterials(List.of(garment("Dark Heather"))));
        assertEquals("#111111", previewColour.forMaterials(List.of(garment("Black"))));
    }

    @Test
    void mapsTheLightCatalogueColours() {
        assertEquals("#FFFFFF", previewColour.forMaterials(List.of(garment("White"))));
        assertEquals("#FAFAFA", previewColour.forMaterials(List.of(garment("Event White"))));
        assertEquals("#A8C6E0", previewColour.forMaterials(List.of(garment("Carolina Blue Mist"))));
        assertEquals("#F2C6D4", previewColour.forMaterials(List.of(garment("Cherry Blossom"))));
        assertEquals("#8FBFA8", previewColour.forMaterials(List.of(garment("True Celadon"))));
    }

    @Test
    void ignoresCaseAndSurroundingWhitespace() {
        assertEquals("#0B0B0B", previewColour.forMaterials(List.of(garment("  jet black  "))));
    }

    @Test
    void usesTheFirstRecommendedMaterialAsTheGarment() {
        List<MaterialRecommendation> materials = List.of(
                garment("Dark Heather"),
                garment("White"));

        assertEquals("#3A3A3A", previewColour.forMaterials(materials));
    }

    @Test
    void returnsNullWhenThereIsNoMaterialOrTheColourIsUnknown() {
        assertNull(previewColour.forMaterials(List.of()));
        assertNull(previewColour.forMaterials(null));
        assertNull(previewColour.forMaterials(List.of(garment("Mystery Fuchsia"))));
        assertNull(previewColour.forMaterials(List.of(garment(null))));
    }
}
