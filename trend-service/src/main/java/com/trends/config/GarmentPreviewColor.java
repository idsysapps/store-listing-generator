package com.trends.config;

import com.trends.dto.MaterialRecommendation;
import jakarta.enterprise.context.ApplicationScoped;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The background colour a transparent design should be previewed on.
 *
 * <p>The first recommended material is the garment the design is printed on
 * (see the DTF design guidelines). The catalogue stores the colour by name, so
 * the client cannot use it as a background directly. This maps the catalogue
 * names to hex colours, and returns {@code null} rather than inventing a
 * colour the catalogue does not offer.
 */
@ApplicationScoped
public class GarmentPreviewColor {

    private static final Map<String, String> COLOURS = Map.ofEntries(
            Map.entry("black", "#111111"),
            Map.entry("jet black", "#0B0B0B"),
            Map.entry("dark heather", "#3A3A3A"),
            Map.entry("white", "#FFFFFF"),
            Map.entry("event white", "#FAFAFA"),
            Map.entry("carolina blue mist", "#A8C6E0"),
            Map.entry("cherry blossom", "#F2C6D4"),
            Map.entry("true celadon", "#8FBFA8"),
            // Colours the cost book may add later. The map is explicit so an
            // unknown name is a null preview, not a guess.
            Map.entry("charcoal", "#36454F"),
            Map.entry("navy", "#1F2A44"),
            Map.entry("sport grey", "#A6A6A6"),
            Map.entry("heather grey", "#9E9E9E"),
            Map.entry("ash", "#D9D9D9"),
            Map.entry("sand", "#D9C7A7"),
            Map.entry("natural", "#EFE6D5"),
            Map.entry("cream", "#F2E8D5"),
            Map.entry("red", "#B3261E"),
            Map.entry("maroon", "#6B1F2A"),
            Map.entry("forest green", "#2F4F3E"),
            Map.entry("royal blue", "#1F4E9C"));

    /** The preview background for the garment, or null when it is unknown. */
    public String forMaterials(List<MaterialRecommendation> materials) {
        if (materials == null || materials.isEmpty()) {
            return null;
        }
        String colour = materials.get(0).getColor();
        if (colour == null || colour.isBlank()) {
            return null;
        }
        String key = colour.trim().replaceAll("\\s+", " ").toLowerCase(Locale.ROOT);
        return COLOURS.get(key);
    }
}
